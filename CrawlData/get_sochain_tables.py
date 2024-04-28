from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT DISTINCT "block_data_" || _TABLE_SUFFIX AS path
FROM `steemit-307308.hive_zurich.block_data_*`
ORDER BY path ASC;
"""

job = client.query(query)

rows = job.result()

for row in rows:
    table_name = row["path"]
    print(f"{table_name}")
