#!/bin/bash
###############################
# CONFIG
###############################

# Bigquery json key
export GOOGLE_APPLICATION_CREDENTIALS="/home/admin/blockchain-data/Steemit-e706b5b8cead.json"
# Folder with scripts
ROOT_FOLDER="/home/admin/blockchain-data/"

###############################
# CONFIG END
###############################
###############################
# UPDATE
###############################

echo "Waiting for table list..."

python "${ROOT_FOLDER}get_sochain_tables.py" | sed -E -n "/^block_data_[0-9]+_[0-9]+_[0-9]+$/p" > "${ROOT_FOLDER}sochain_table_list.txt"

echo "Table list updated"

###############################
# UPDATE END
###############################
