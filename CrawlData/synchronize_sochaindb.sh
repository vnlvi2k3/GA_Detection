#!/bin/bash

###############################
# CONFIG
###############################
ROOT_DIR="/home/admin/blockchain-data/"
BIGQUERY_DIR="/home/admin/bigquery-data/"
SCRAPE_DIR="/home/admin/scrape-data/"
ARCHIVE_DIR="/home/admin/host/"
###############################
# CONFIG END
###############################
last_processed_month () {
	LAST_PROCESSED_DATE=$(tail -n 1 $(find $BIGQUERY_DIR -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+-[0-9]+\.json' | sort -n | tail -n 1) | sed -E -n 's/^.*"timestamp": "([0-9]{4})-([0-9]{2})-[0-9]{2}.*$/\2.\1/p')
}
last_archived_month () {
	LAST_ARCHIVED_DATE=$(find $ARCHIVE_DIR -maxdepth 1 -type f -regex '.*block_data_[0-9]+-[0-9]+-[0-9]+\.json\.gz' | sort -n | tail -n 1 | sed -E -n 's/^.*block_data_([0-9]{4})-([0-9]{2})-[0-9]{2}\.json\.gz$/\2.\1/p')
}
scraped_files () {
	SCRAPE_BLOCK_FILES=$(find $SCRAPE_DIR -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+\.json' | sort -n)

}


last_processed_month
last_archived_month
scraped_files
CURRENT_MONTH=$(date +%m.%y)

echo $SCRAPE_BLOCK_FILES
echo $LAST_PROCESSED_DATE
echo $LAST_ARCHIVED_DATE
echo $CURRENT_MONTH
echo "~~~~~~~~~"

if [ -z "$SCRAPE_BLOCK_FILES" ]
then
	echo ${ROOT_DIR}scrape_blocks.sh
	SCRAPE_BLOCK_FILES=$(find $SCRAPE_DIR -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+\.json' | sort -n)
fi

for file in "$SCRAPE_BLOCK_FILES"
do 
	echo ${ROOT_DIR}process_data_parallel.sh 
done

echo python ${ROOT_DIR}archive_blocks_month.py
