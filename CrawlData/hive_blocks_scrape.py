""" Scrape blockchain blocks from hive blockchain """
import ndjson
from argparse import ArgumentParser
from beem import Hive
from beem.blockchain import Blockchain

def parse_args():
    """ Parse scraping arguments """
    parser = ArgumentParser(description="Scrape blocks from hive API")
    parser.add_argument("-a", "--api", action="append", required=True, type=str, help="API to crawl")
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", default=0)
    parser.add_argument("-o", "--output-file", type=str, required=True, help="File to store the blocks")
    parser.add_argument("-s", "--start-block", type=int, required=True, help="Starting block")
    parser.add_argument("-f", "--finish-block", type=int, required=True, help="End block to scrape")
    return parser.parse_known_args()


def setup_node(endpoints):
    """ Configure hive node """
    return Hive(node=endpoints, num_retries=15)


def setup_block_scrape(node: Hive, threads: int, filepath: str, start: int, finish: int):
    """ Scrape blocks from endpoint into a file """
    blockchain = Blockchain(node)
    with open(filepath, "w+") as f:
        writer = ndjson.writer(f, ensure_ascii=False)
        threading = bool(threads)
        for b in blockchain.blocks(start, finish, threading=threading, thread_num=threads):
            print("Writing block: " + str(b.block_num))
            writer.writerow(b.json())


if __name__ == "__main__":
    args = parse_args()[0]
    node = setup_node(args.api)
    setup_block_scrape(node, args.threads, args.output_file, args.start_block, args.finish_block)
    print("Done scraping!")
