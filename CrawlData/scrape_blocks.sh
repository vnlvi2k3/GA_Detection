#!/usr/bin/env bash

###############################
# CONFIG
###############################
# Folder for block NDJSON files that are going to be uploaded to bigquery
BIGQUERY_DATA_FOLDER="/home/admin/bigquery-data/"
# Folder for scraped block NDJSON files that need to be preprocessed
SCRAPED_DATA_FOLDER="/home/admin/scrape-data/"
# Hive API endpoints to be used for downloading the data 
# Formatting:
# {url} {name}
SOURCE_ENDPOINTS=$(cat << EOF
https://api.deathwing.me deathwing
https://hived.emre.sh emre
https://rpc.ausbit.dev ausbit
https://api.c0ff33a.uk c0ff33a
EOF
)
###############################
# CONFIG END
###############################
###############################
# LOAD DATA
###############################
SCRAPED_FILES=""
CHILD_PROCESSES=""

cleanup() {
	kill ${CHILD_PROCESSES}
	rm ${SCRAPED_FILES}
}

trap cleanup EXIT

DOWNLOAD_SOURCE_AMOUNT=$(echo "$SOURCE_ENDPOINTS" | wc -l)
LATEST_BLOCK=$(python get_latest_block.py)
LATEST_BLOCK=${LATEST_BLOCK:18}
LATEST_STORED_BLOCK=$(ls ${BIGQUERY_DATA_FOLDER}block_data_*.json | sed -n -E "s/^.*block_data_[0-9]+_([0-9]+)-[0-9]+\.json$/\1/p" | tail -n 1)
BLOCK_STEP=$((($LATEST_BLOCK - $LATEST_STORED_BLOCK)/$DOWNLOAD_SOURCE_AMOUNT))
BLOCK_STEPS=$(seq $LATEST_STORED_BLOCK $BLOCK_STEP $LATEST_BLOCK)
START_BLOCKS=$(echo "$BLOCK_STEPS" | sed "1d;")
END_BLOCKS=$(echo "$BLOCK_STEPS" | sed "1d; $ d;")
FIRST_BLOCK=$(echo "$BLOCK_STEPS" | head -n 1)
((FIRST_BLOCK++))


for i in $(seq "$(($DOWNLOAD_SOURCE_AMOUNT - 1))")
do
	temp=$(echo "$START_BLOCKS" | sed -n "$i p")
	((temp++))
	START_BLOCKS=$(echo "$START_BLOCKS" | sed "$i s/[0-9]*/$temp/")
done

START_BLOCKS=$(echo "$START_BLOCKS" | sed "1i $FIRST_BLOCK")
END_BLOCKS=$(echo "$END_BLOCKS" | sed "$ a $LATEST_BLOCK")


for interval in $(seq $DOWNLOAD_SOURCE_AMOUNT)
do
	INTERVAL_START=$(echo "$START_BLOCKS" | sed -n "$interval p")
	INTERVAL_END=$(echo "$END_BLOCKS" | sed -n "$interval p")
	API_NAME=$(echo "$SOURCE_ENDPOINTS" | sed -n "s/^.* //g; $interval p;")
	API_LINK=$(echo "$SOURCE_ENDPOINTS" | sed -n "s/ [a-z0-9A-Z]*$//g; $interval p;")
	echo "Scraping blocks $INTERVAL_START - $INTERVAL_END from: $API_NAME using link: $API_LINK"
	SCRAPE_FILE="${SCRAPED_DATA_FOLDER}block_data_${API_NAME}.json"
	LOG_FILE="${SCRAPED_DATA_FOLDER}${API_NAME}.log"
	python hive_blocks_scrape.py -a $API_LINK -o $SCRAPE_FILE -s $INTERVAL_START -f $INTERVAL_END > $LOG_FILE 2> $LOG_FILE &
	SCRAPED_FILES="$SCRAPED_FILES $SCRAPE_FILE"
	CHILD_PROCESSES="$CHILD_PROCESSES $!"
done

echo "Waiting scraping to finish..."

wait

SCRAPE_OUTPUT="${SCRAPED_DATA_FOLDER}block_data_${FIRST_BLOCK}_${LATEST_BLOCK}.json"

cat $SCRAPED_FILES > $SCRAPE_OUTPUT


###############################
# LOAD DATA END
###############################
