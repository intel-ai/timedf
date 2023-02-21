#!/bin/sh

set -vxe

conda install -n omnisci-dev mysql mysql-connector-python sqlalchemy>=1.4 -c conda-forge


# Modin installation
git clone https://github.com/modin-project/modin.git
cd modin && pip install -e .[ray] && cd ..

# ASV installation
git clone https://github.com/airspeed-velocity/asv.git
cd asv
git checkout ef016e233cb9a0b19d517135104f49e0a3c380e9
git apply ../omniscripts/docker/microbenchmarks-hdk/asv-default-timeout.patch
cd ..
cd asv && pip install -e . && cd ..

cd modin
cat << EOF >> groupby.patch
diff --git a/asv_bench/benchmarks/hdk/benchmarks.py b/asv_bench/benchmarks/hdk/benchmarks.py
index 2ba83c03..0bdd1279 100644
--- a/asv_bench/benchmarks/hdk/benchmarks.py
+++ b/asv_bench/benchmarks/hdk/benchmarks.py
@@ -501,3 +501,10 @@ class TimeGroupByMultiColumn(BaseTimeGroupBy):
                 {col: "mean" for col in self.non_groupby_columns}
             )
         )
+
+    def time_groupby_agg_nunique_dict(self, *args, **kwargs):
+        execute(
+            self.df.groupby(by=self.groupby_columns).agg(
+                {col: "nunique" for col in self.non_groupby_columns}
+            )
+        )
EOF
git apply groupby.patch
cd ..

echo "asv script launch"
chmod +x ./omniscripts/docker/microbenchmarks-hdk/asv-runner.sh
./omniscripts/docker/microbenchmarks-hdk/asv-runner.sh

