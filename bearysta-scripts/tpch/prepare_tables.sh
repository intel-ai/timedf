if [ ! -d "generated/data-$1-$2" ]; then

    bash generate_tables.sh $1

    mkdir -p generated/data-$1-$2 && ../../../omniscidb/build/bin/initdb --data generated/data-$1-$2

    python3 prepare_tables.py $2

    ../../../omniscidb/build/bin/omnisci_server --db-query-list=generated/create_tables.sql --exit-after-warmup --data generated/data-$1-$2 --log-severity-clog=INFO
fi

