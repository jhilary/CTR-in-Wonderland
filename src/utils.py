import resource
import importlib
import platform


def str_to_function(some_str):
    module_name, _, function_name = some_str.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def get_memory_usage():
    if platform.system() == u'Linux':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

