git submodule sync
mkdir -p generated
rm -rf generated/*

pushd tpch-dbgen
make
./dbgen -s $1 -f
sed -i 's/.$//' *.tbl
mv *.tbl ../generated

for q in 1 5 11; do 
	DSS_QUERY=../query_templates ./qgen $q -s $1 > ../generated/$q.sql
	dos2unix -o ../generaged/$q.sql
done;

popd

echo "USER admin omnisci {" > generated/queries.sql

for q in 1 5 11; do
	cat generated/$q.sql >> generated/queries.sql
	cat generated/$q.sql >> generated/queries.sql
	cat generated/$q.sql >> generated/queries.sql
done;

echo "}" >> generated/queries.sql
