import os
import warnings

from .pandas_backend import PandasBackend, pandas_backends

__all__ = ["create_backend", "HdkBackend", "PandasBackend", "PolarsBackend"]


def create_backend(name, params):
    name2backend = {
        # TODO: will be replaced to pandas in the future, since that's the backend
        **{b: PandasBackend for b in pandas_backends},
        "polars": PolarsBackend,
        "hdk": HdkBackend,
    }

    if name in name2backend:
        return name2backend[name](params)
    else:
        raise ValueError(f"Unrecognized backend: {name}")


class PolarsBackend:
    def __init__(self, params) -> None:
        n = params.get("num_threads", None)
        if n is not None:
            os.environ["POLARS_MAX_THREADS"] = str(n)

    # TODO: just rewrite benchmark to trigger with polars tools and remove this function
    def trigger_loading(self, *dfs):
        return self.trigger_execution(*dfs)

    def trigger_execution(self, *dfs):
        results = [d.collect() if hasattr(d, "collect") else d for d in dfs]

        if len(dfs) == 1:
            return dfs[0]
        else:
            return dfs


# TODO: maybe we need general notification about unexpected params
class HdkBackend:
    # We expect HDK to trigger execution manually in benchmark source code with *.run()
    # The reason is that checks such as
    # `from pyhdk.hdk import QueryNode; isinstance(df, QueryNode)` can take ~0.5s to run
    # significantly affecting performance of small queries.
    # Just running *.run() here doesn't work because it can lead to double execution.
    def __init__(self, params) -> None:
        warnings.warn(
            "HDK currently doesn't have control for number of CPU cores used, "
            "'num_threads' will be ignored"
        )
        import pyhdk

        # This will be used to configure HDK when options will be relevant
        pyhdk.init()
