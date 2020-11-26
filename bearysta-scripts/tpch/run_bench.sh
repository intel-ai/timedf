rm -rf data
rm -rf runs
rm -rf generated
mkdir data
../../../omniscidb/build/bin/initdb --data data

bash generate_tables.sh $1
bash prepare_tables.sh


python -m bearysta.run --bench-path run-tpc-h-bench.yml
python -m bearysta.aggregate tpc-h-query-times.yml -P
