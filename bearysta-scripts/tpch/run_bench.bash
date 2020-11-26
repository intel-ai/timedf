rm -rf data
rm -rf runs
rm -rf generated
mkdir data
../../../omniscidb/build/bin/initdb --data data

bash generate_tables.bash $1
bash prepare_tables.bash


python -m bearysta.run --bench-path run-tpc-h-bench.yml
python -m bearysta.aggregate tpc-h-query-times.yml -P
