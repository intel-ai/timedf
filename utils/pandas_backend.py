# from utils import import_pandas_into_module_namespace

from ast import Module
import importlib
from types import ModuleType

import pandas as pd

from utils import import_pandas_into_module_namespace


class PB:
    """Holder of pandas backend. Intended use:
    1. Use the same instance across different modules.
    2. Get pandas in each module with `pd = pb.get_pd(__name__)`, this automatically subscribes to change of backend.
    3. Call `set_backend` once to update pd variable across all the subscribed modules."""

    def __init__(self) -> None:
        self.subscribed_modules = []
        self.pd_store = {"pd": pd}

    def register_pd_user(self, module_name):
        self.subscribed_modules.append(module_name)
        # return self.pd_store['pd']

    def set_backend(self, pandas_mode, ray_tmpdir, ray_memory):
        import_pandas_into_module_namespace(
            namespace=self.pd_store,
            mode=pandas_mode,
            ray_tmpdir=ray_tmpdir,
            ray_memory=ray_memory,
        )

        for module_name in self.subscribed_modules:
            module = importlib.import_module(module_name)
            module.__dict__["pd"] = self.pd_store["pd"]


pb = PB()
