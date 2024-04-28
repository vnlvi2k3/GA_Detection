#! /bin/bash
###############################
# CONFIG
###############################

# Folder for block NDJSON files that are going to be uploaded to bigquery
BIGQUERY_DATA_FOLDER="/home/admin/bigquery-data/"
# Folder for scraped block NDJSON files that need to be preprocessed
SCRAPED_DATA_FOLDER="/home/admin/scrape-data/"
# Python script for processing scraped data
PROCESS_SCRIPT="/home/admin/blockchain-data/process_data_bigquery.py"
# Bigquery server to be used
BIGQUERY_PLACE="steemit-307308.hive_zurich"
# Amount of cores on host machine / how many files will data be split to
CORES=48

###############################
# CONFIG END
###############################
# PROCESSING OF FILES
###############################

CORE_COUNT_DIGITS=$(printf "$CORES" | wc -m)
DIGIT_FORMAT="%0${CORE_COUNT_DIGITS}g"

WHOLE_FILES="$(find $SCRAPED_DATA_FOLDER -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+\.json' -printf '%f\n' | sort -n)"
PROCESSED_FILES="$(find $BIGQUERY_DATA_FOLDER -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+-[0-9]+\.json' -printf '%f\n' | sort -n)"
DOWNLOADED_FILES="$(find $SCRAPED_DATA_FOLDER -maxdepth 1 -type f -regex '.*block_data_[0-9]+_[0-9]+-[0-9]+\.json' -printf '%f\n' | sort -n)"
SPLIT_PROCESS_FILES=$(comm -23 <(echo "$DOWNLOADED_FILES") <(echo "$PROCESSED_FILES"))

echo "$WHOLE_FILES"
echo "~~~~~~~~~~~~"

RUNNING_PROCESSES=""
CLEAN_FILES=""

cleanup() {
	kill ${RUNNING_PROCESSES}
	echo "rm ${CLEAN_FILES}"
}

trap cleanup KILL

for file in ${WHOLE_FILES}
do
	split_file=$(echo "$file" | sed "s/\.json$/-/")
	echo "split ${SCRAPED_DATA_FOLDER}${file} $split_file -n l/$CORES --numeric-suffixes=1 --additional-suffix='.json'"
	for c in $(seq -f $DIGIT_FORMAT 1 $CORES)
	do
		split_part="${split_file}${c}.json"
		SPLIT_PROCESS_FILES="${SPLIT_PROCESS_FILES}"$'\n'"${split_part}"
	done
	CLEAN_FILES="$CLEAN_FILES $file"
done

echo "$SPLIT_PROCESS_FILES"

for c in $(seq -f $DIGIT_FORMAT 1 $CORES)
do

	PROCESSING_FILE="${BIGQUERY_DATA_FOLDER}${INPUT_FILE}_processing-${c}.log"
	echo 'python "$PROCESS_SCRIPT" \
		-i "${SCRAPED_DATA_FOLDER}${INPUT_FILE}-${c}${FILE_EXTENSION}" \
		-o "${BIGQUERY_DATA_FOLDER}${INPUT_FILE}-${c}${FILE_EXTENSION}" \
		-s "${BIGQUERY_DATA_FOLDER}schemas/${INPUT_FILE}-${c}${FILE_EXTENSION}" > "$PROCESSING_FILE" 2> "$PROCESSING_FILE"' &
	RUNNING_PROCESSES="$RUNNING_PROCESSES $!"
	echo $(echo " 12 15 44 30" | grep -o -E "[0-9]+" | wc -l)
done

echo "Waiting for jobs to finish..."

wait

###############################
# PROCESSING OF FILES END
###############################
