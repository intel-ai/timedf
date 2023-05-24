Arbitrary workload
==================

If you are a developer working with your own benchmark (maybe from customer workload)
and you don't want to implement your benchmark as part of ``timedf``  you can,
nonetheless, use timedf library to simplify benchmarking for you.

The basic pipeline is:

#. You make timedf library available for your own benchmark script
#. You gather benchmark measurements by yourself or with ``TimerManager``
#. You import ``BenchmarkDd``, connect to intermediate database (easiest DB to run by yourself if sqlite, that is essentially just a file) and report results like this::

    from sqlalchemy import create_engine

    from timedf.report import BenchmarkDb


    # your benchmark code
    ...
    ...
    # example of results, can be anything in a form Dict[str, float]
    benchmark_results = {'load_data': 11.1, 'etl': 12.0, ...}


    # you can connect to other database, but this is the easiest option
    engine = create_engine('sqlite+pysqlite:///db_file.sql')

    db = BenchmarkDb(engine)

    db.report_arbitrary(
            benchmark='my_benchmark',
            run_i = 122, # needs to be unique for this run, if you manually run this benchmark multiple times (iterations), then provide the same value for each iteration
            backend = 'my_backend', # this will be stored in `iteration.pandas_mode`
            iteration_no= 1, # if you perform multiple iterations manually, then provide correct counter, this will be stored in `iteration.iteration_no`
            name2time =  benchmark_results,
            params = {'n_cpu': 32}, # you can report whatever you want to later use this for analysis, this data is stored in `iteration.params`
        ):

#. Now this database contains your results. You can use already available tools like xlsx generation or jupyter notebook visualization (recommended, the link is https://github.com/intel-ai/benchmarks_tutorials/blob/main/visualization/reporter.ipynb) to visualize your results. The notebook contains manual for result loading and interpretation.