""" Check scraped files """
import ndjson
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Check scraped file integrity")
    parser.add_argument("-f", "--files", type=str, required=True, action="append", help="Files to check integrity")
    return parser.parse_known_args()


def read_files(files):
    file_ranges = {}
    for f in files:
        with open(f, "r") as of:
            print("checking file: " + f)
            file_ranges[f] = []
            temp_first = 0
            temp_last = 0
            reader = ndjson.reader(of)
            temp_first = next(reader)["id"]
            temp_last = temp_first
            for b in reader:
                block_id = b["id"]
                if block_id == temp_last + 1:
                    temp_last = block_id
                elif block_id > temp_last:
                    file_ranges[f].append((temp_first, temp_last, "consecutive blocks"))
                    file_ranges[f].append((temp_last + 1, block_id - 1, "gap"))
                    temp_first = block_id
                    temp_last = block_id
                if block_id < temp_first:
                    file_ranges[f].append((temp_first, temp_last, "consecutive blocks"))
                    temp_first = block_id
                    temp_last = block_id
            file_ranges[f].append((temp_first, temp_last, "consecutive blocks"))
    return file_ranges



if __name__ == "__main__":
    args = parse_args()[0]
    ranges = read_files(args.files)
    for r in ranges:
        print("-----------------------")
        print(r)
        print("-----------------------")
        for rr in ranges[r]:
            print(rr)
