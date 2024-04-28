from beem import Hive
from beem.blockchain import Blockchain
from beem.nodelist import NodeList
nodelist = NodeList()
nodelist.update_nodes()
nodes = nodelist.get_hive_nodes()

hive = Hive(nodes=nodes, num_retries=15)
blockchain = Blockchain(hive)

print(f"Current block is: {blockchain.get_current_block_num()}")
