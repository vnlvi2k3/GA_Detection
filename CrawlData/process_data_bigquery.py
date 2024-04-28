""" Process hive blocks for bigquery """
import ndjson
import logging
import json
import re
from json.decoder import JSONDecodeError
from copy import deepcopy
from argparse import ArgumentParser
from genson import SchemaBuilder
from genson.schema.strategies import Object, List
from bigquery_schema_generator.generate_schema import SchemaGenerator


SUBSTITUTE_PATTERN = re.compile("[^a-zA-Z0-9_]")

xp_found = False

class NoAdditionalPropertiesObject(Object):
    KEYWORDS = (*Object.KEYWORDS, "additionalProperties")

    def to_schema(self):
        schema = super().to_schema()
        schema["additionalProperties"] = False
        return schema


class CountEmptyList(List):
    KEYWORDS = (*List.KEYWORDS, "maxItems")

    def __init__(self, node_class):
        super().__init__(node_class)
        self.empty = True

    def add_schema(self, schema):
        super().add_schema(schema)
        self.empty = bool(schema.get("maxItems", False))

    def add_object(self, obj):
        super().add_object(obj)
        self.empty = not bool(len(obj))

    def to_schema(self):
        schema = super().to_schema()
        if self.empty:
            schema["maxItems"] = 0
        return schema


class ConservativeSchemaBuilder(SchemaBuilder):
    EXTRA_STRATEGIES = (NoAdditionalPropertiesObject, CountEmptyList)


def parse_args():
    parser = ArgumentParser(description="Process dataset file into bigquery-friendly way")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file")
    parser.add_argument("-o", "--output", required=True, type=str, help="Outut file")
    parser.add_argument("-s", "--schema", required=True, type=str, help="Bigquery schema file path")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output")
    return parser.parse_known_args()


def sanitize_name(value):
    if not value:
        return "__empty_key_json"
    if value[0].isdigit():
        value = f"n{value}"
    return SUBSTITUTE_PATTERN.sub("_", value[:100]).lower()


def nested_arr_process(nested_arrays, cast_type=None):
    flattened = []
    for i, o in enumerate(nested_arrays):
        for ii in o:
            if not cast_type:
                if ii is None:
                    flattened.append({"index": i, "value_str": ""})
                else:
                    flattened.append({"index": i, f"value_{type(ii).__name__}": ii})
            else:
                flattened.append({"index": i, "value": cast_type(ii)})
    return flattened


def ensure_homogenous(data):
    if isinstance(data, list):
        type_mapping = [ensure_homogenous(i) for i in data]
        if len(set(t[1] for t in type_mapping)) == 1:
            if "list" in type_mapping[0][1]:
                return [
                        {
                            "index": i,
                            f"value_{t[1]}": t[0]
                        }
                        for i, t in enumerate(type_mapping)], ("list_" + type_mapping[0][1])
            return [t[0] for t in type_mapping], ("list_" + type_mapping[0][1])
        else:
            for i, t in enumerate(type_mapping):
                value_name = f"value_{t[1]}"
                data[i] = {value_name: t[0]}
            return data, "list_dict"
    if isinstance(data, dict):
        if len(data) == 0:
            return "", str.__name__
        obj_keys = list(data.keys())
        for k in obj_keys:
            if k.isdigit() and isinstance(data[k], list) and all(isinstance(t, dict) for t in data[k]):
                game_items = []
                for d in data[k]:
                    for kk in d:
                        if d[kk]:
                            game_items.append({
                                "key": kk,
                                "value": d[kk]})
                        else:
                            game_items.append({"key": kk})
                item_result = (game_items, "list_str")
            else:
                item_result = ensure_homogenous(data[k])
            data[f"{sanitize_name(k)}_{item_result[1]}"] = item_result[0]
            del data[k]
        return data, "dict"
    if data is None:
        return "", str.__name__
    return data, type(data).__name__


def resolve_json(value_dict, field_name, block_id):
    try:
        result = None
        if not isinstance(value_dict[field_name], str):
            result = ensure_homogenous(value_dict[field_name])
        else:
            if value_dict[field_name]:
                result = json.loads(value_dict[field_name], strict=False)
                result = ensure_homogenous(result)
                global xp_found
                if xp_found:
                    xp_found = False
            else:
                result = ensure_homogenous(value_dict[field_name])
        value_dict[f"{field_name}_{result[1]}"] = result[0]
    except JSONDecodeError as err:
        logging.debug(f"Encountered error {str(err)} on block {block_id} with value of field '{field_name}' {value_dict}")
        value_dict[f"{field_name}_err"] = value_dict[field_name]
    finally:
        del value_dict[field_name]


def process_blocks(input_file, output_file):
    b = {}
    counter = 1
    logging.info("Starting processing...")
    for block in input_file:
        if "extensions" in block:
            extensions_value, extensions_type = ensure_homogenous(block["extensions"])
            block[f"extensions_{extensions_type}"] = extensions_value
            del block["extensions"]
        for t in block["transactions"]:
            for o in t["operations"]:
                if "extensions" in o:
                    extensions_value, extensions_type = ensure_homogenous(o["value"]["extensions"])
                    o["value"][f"extensions_{extensions_type}"] = extensions_value
                    del o["value"]["extensions"]
                value_dict = o.get("value", b)
                if not value_dict:
                    break
                if "props" in value_dict:
                    props_value, props_type = ensure_homogenous(o["value"]["props"])
                    o["value"][f"props_{props_type}"] = props_value
                    del o["value"]["props"]
                if "owner" in value_dict:
                    owner_value, owner_type = ensure_homogenous(o["value"]["owner"])
                    o["value"][f"owner_{owner_type}"] = owner_value
                    del o["value"]["owner"]
                if "active" in value_dict:
                    active_value, active_type = ensure_homogenous(o["value"]["active"])
                    o["value"][f"active_{active_type}"] = active_value
                    del o["value"]["active"]
                if "posting" in value_dict:
                    posting_value, posting_type = ensure_homogenous(o["value"]["posting"])
                    o["value"][f"posting_{posting_type}"] = posting_value
                    del o["value"]["posting"]
                if "id" in value_dict:
                    o["value"]["id"] = str(o["value"]["id"])
                if "json" in value_dict:
                    resolve_json(o["value"], "json", block["id"])
                if "posting_json_metadata" in value_dict:
                    resolve_json(o["value"], "posting_json_metadata", block["id"])
                if "json_metadata" in value_dict:
                    resolve_json(o["value"], "json_metadata", block["id"])
                if "new_owner_authority" in value_dict:
                    new_owner_authority_value, new_owner_authority_type = ensure_homogenous(o["value"]["new_owner_authority"])
                    o["value"][f"new_owner_authority_{new_owner_authority_type}"] = new_owner_authority_value
                    del o["value"]["new_owner_authority"]
                if "recent_owner_authority" in value_dict:
                    recent_owner_authority_value, recent_owner_authority_type = ensure_homogenous(o["value"]["recent_owner_authority"])
                    o["value"][f"recent_owner_authority_{recent_owner_authority_type}"] = recent_owner_authority_value
                    del o["value"]["recent_owner_authority"]
        if counter == 100000:
            counter = 1
            logging.info("Processed row: " + str(block["id"]))
        else:
            counter += 1
        output_file.writerow(block)


if __name__ == "__main__":
    args = parse_args()[0]
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    schema = None
    with open(args.input, "r+") as i, open(args.output, "w+") as o:
        reader = ndjson.reader(i)
        output_writer = ndjson.writer(o)
        process_blocks(reader, output_writer)
    with open(args.schema, "w+") as f, open(args.output, "r") as r:
        generator = SchemaGenerator(quoted_values_are_strings=True, input_format="json", keep_nulls=True)
        schema_map, error_logs = generator.deduce_schema(input_data=r)
        if error_logs:
            logging.error(args.output)
            raise Exception(f"Error generating schema from data: {str(error_logs)}")
        json.dump(generator.flatten_schema(schema_map), f, indent=2)
