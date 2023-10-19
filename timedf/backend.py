# TODO: legacy for backward-compatibility
# This will be replaced by modin.pandas after Backend.init() call
import pandas as pd  # noqa: F401 this import exists to provide vscode support for backend users


class Backend:
    """Singleton storing backend utilities and configurations"""

    # Backend was initalized and ready for work
    _ready = False
    # Backend name
    _name = None

    @classmethod
    def init(cls, backend_name: str, backend_params: dict):
        from timedf.backends import create_backend

        cls._name = backend_name
        cls._backend = create_backend(backend_name, backend_params)

        cls._ready = True

    @classmethod
    def _check_ready(cls):
        if not cls._ready:
            raise ValueError("Attempting to use unitialized backend")

    # Legacy call
    @classmethod
    def get_name(cls):
        cls._check_ready()
        if cls.get_backend_name() == "Pandas":
            return cls.get_backend_impl().pandas_mode
        else:
            return cls.get_backend_name()

    @classmethod
    def get_backend_name(cls):
        cls._check_ready()
        return cls._name

    @classmethod
    def get_backend_impl(cls):
        cls._check_ready()
        return cls._backend

    # TODO LEGACY to be removed
    @classmethod
    def get_modin_cfg(cls):
        # TODO: deprecated legacy for old modin interface
        # Just use Backend.get_backend_impl().params
        if cls.get_name().lower() != "pandas":
            return None
        return cls.get_backend_impl().params

    # TODO LEGACY to be removed, we expect backends to trigger by themselves
    @classmethod
    def trigger_loading(cls, *dfs):
        if cls.get_name().lower() in ["polars", "pandas"]:
            cls.get_backend_impl().trigger_loading(*dfs)
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    # TODO LEGACY to be removed
    @classmethod
    def trigger_execution(cls, *dfs):
        if cls.get_name().lower() in ["polars", "pandas"]:
            cls.get_backend_impl().trigger_execution(*dfs)
        if len(dfs) == 1:
            return dfs[0]
        return dfs
