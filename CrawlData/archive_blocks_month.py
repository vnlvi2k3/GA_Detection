""" Script to split scraped blocks to month """
from datetime import datetime
from dateutil import parser
from dateutil.rrule import rrule, MONTHLY
from shutil import copyfileobj
import operator
import os
import gzip
import re
from argparse import ArgumentParser
from json import loads

arg_parser = ArgumentParser("Split and archive given block files")
arg_parser.add_argument("-s", "--start_month", type=lambda d: datetime.strptime(d, "%m.%Y"), required=True)
arg_parser.add_argument("-e", "--end_month", type=lambda d: datetime.strptime(d, "%m.%Y"), required=True)
arg_parser.add_argument("-o", "--output", type=str, required=True)
args = arg_parser.parse_args()
start_month = args.start_month
end_month = args.end_month
output = args.output

matcher = re.compile("^block_data_([0-9]+)_([0-9]+)-([0-9]+).json$")

block_mapping = []

def map_folder(folder):
    for file in os.listdir(folder):
        res = matcher.match(file)
        if not res:
            continue
        block_mapping.append({
            "name": folder + res.group(0),
            "start_block": int(res.group(1)),
            "end_block": int(res.group(2)),
            "block_number": int(res.group(3))
        })

def monthdiff(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def archive_iterator(start_month, end_month):
    for m in rrule(dtstart=start_month, until=end_month, freq=MONTHLY):
        with gzip.open(output + f"block_data_{m.year}-{m.month:02}-{m.day:02}.json.gz", "wb") as a:
            yield m, a

def get_first_timestamp(file):
    with open(file["name"], "r+") as f:
        first_line = f.readline()
    first_line = loads(first_line)
    return parser.parse(first_line["timestamp"])

def files_iterator(start_index, end_index):
    current_file = None
    first_file = None
    append_files = []
    last_file = None
    file_iter = (block_mapping[i] for i in range(start_index, end_index - 1, -1))
    while True:
        month = yield (first_file, append_files, last_file)
        if current_file is not None:
            first_file = last_file
            append_files = [current_file]
        for f in file_iter:
            current_file = f
            file_timestamp = get_first_timestamp(current_file)
            if monthdiff(file_timestamp, month) == -1:
                first_file = current_file
            if monthdiff(file_timestamp, month) == 0:
                append_files.append(current_file)
            if monthdiff(file_timestamp, month) == 1:
                last_file = append_files.pop()
                break

map_folder("/home/admin/bigquery-data/")

block_mapping = sorted(block_mapping, key=operator.itemgetter("start_block", "block_number"), reverse=True)

# find passing month interval

start_file = None
end_file = None

block_iter = enumerate(block_mapping)

for i, file in block_iter:
    file_timestamp = get_first_timestamp(file)
    if monthdiff(file_timestamp, end_month) == 0:
        if i == 0:
            end_file = i - 1
        else:
            end_file = i
        break

inside_month = False

for i, file in block_iter:
    file_timestamp = get_first_timestamp(file)
    if monthdiff(file_timestamp, start_month) == 0 or inside_month:
        if monthdiff(file_timestamp, start_month) == -1:
            start_file = i + 1
            break
        else:
            inside_month = True
            continue

if not start_file:
    start_file = len(block_mapping) - 1

files_iter = files_iterator(start_file, end_file)
month_iter = archive_iterator(start_month, end_month)
files_iter.send(None)
temp_block = None
timestamp = None
for m, w in month_iter:
    print(f"Writing archive for month: {m}")
    month_files = files_iter.send(m)
    print(f"Writing first file: {month_files[0]}")
    with open(month_files[0]["name"], "r") as f:
        for line in f:
            temp_block = loads(line)
            timestamp = parser.parse(temp_block["timestamp"])
            if monthdiff(timestamp, m) == 0:
                w.write(line.encode())
    for af in month_files[1]:
        print(f"Appending intermediate file: {af}")
        with open(af["name"], "rb") as oaf:
            copyfileobj(oaf, w)
    print(f"Writing last file: {month_files[2]}")
    with open(month_files[2]["name"], "r") as f:
        for line in f:
            temp_block = loads(line)
            timestamp = parser.parse(temp_block["timestamp"])
            if monthdiff(timestamp, m) == 0:
                w.write(line.encode())

