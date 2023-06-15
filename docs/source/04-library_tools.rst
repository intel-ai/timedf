Benchmarking utilities
======================

Library
-------

When you are writing new benchmark you can use several tools, provided by timedf library.

pandas backend
^^^^^^^^^^^^^^

Pandas is the most popular python library for data processing and we support modin backend for it. You can automatically use either pandas or modin with this import::

    from timedf.backend import pd

The actual backend will be picked depending on ``-backend``  parameter.

Currently supported values: ``"Pandas"`` , ``"Modin_on_ray"``, ``"Modin_on_hdk"``

timer
^^^^^

When writing benchmarks we often need to measure time it takes to perform particular block of code. The library contains
a tool for that purpose that can perform this task well, using context manager::

    from timedf import TimerManager

    tm = TimerManager()


    with tm.timeit('heavy_call'):
        # this call takes 11 seconds
        heavy_call()

    with tm.timeit('easy_call'):
        c = 1 + 2


    print(tm.get_results())
    # {'heavy_call': 11.0, 'easy_call': 0.0001}

Timer manager supports nested measurements like this::

    from timedf import TimerManager

    tm = TimerManager()


    def load_data():
        with tm.timeit('load_data'):
            df = pd.read_csv('dataset.csv')

    def append_feature1(df):
        with tm.timeit('feature_c'):
            df['c'] = 12
        return df

    def append_feature2(df):
        with tm.timeit('feature_d'):
            df['d'] = 12
        return df

    def append_feature3(df):
        with tm.timeit('feature_e'):
            df['e'] = 12
        return df

    def feature_engineering(df):
        with tm.timeit('fe'):
            df = append_feature1(df)
        df = append_feature2(df)
            df = append_feature3(df)
        return df

    def main():
        with tm.timeit('total'):
            df = load_data()
            df = feature_engineering(df)

    main()

    print(tm.get_results())
    # {'total': 11.0, 'total.load_data': 2.0, 'total.fe': 9.0, 'total.fe.feature_c': 3.0, 'total.fe.feature_d': 3.0, 'total.fe.feature_e': 3.0}

If you want to use ``TimerManager``  across several files you can do that,
it will maintain nested measurements.
You just need to make sure that the same instance is used across all
files (by defining one instance in utility file, for instance).
It's important to know that TimerManager is not thread-safe,
so use it in main thread only.

Benchmark
^^^^^^^^^

Benchmark class provides interface for timedf library to interact with your benchmark::

    from timedf import BaseBenchmark, BenchmarkResults

    # You need to call your benchmark class exactly "Benchmark"
    class Benchmark(BaseBenchmark):

        # Write your payload in this function
        def run_benchmark(self, params) -> BenchmarkResults:
            pass
            return BenchmarkResults({'load_data': 11.0})

Visualization
-------------

There are tools to help you visualize experiment results

xlsx generation
^^^^^^^^^^^^^^^

To generate xlsx table with experiment results run (from library root)::

    report-xlsx -report_path RESULT_FILE_PATH.xlsx -agg median $DB_OPTIONS

``$DB_OPTIONS`` stand for connection parameters for your database.
In case of sqlite database it's enough to provide path to sqlite file like this: ``db_name PATH.sqlite``.

notebook
^^^^^^^^

There is a notebook with result visualization, located in https://github.com/intel-ai/benchmarks_tutorials/blob/main/visualization/reporter.ipynb

It can be used if you want to visualize benchmark results in jupyter notebook, which should be useful for developers who want to get benchmark results quickly and without using additional infrastructure.

The simplest way to use it is to save your benchmark results in an sqlite table and then visualize there results with notebook.
