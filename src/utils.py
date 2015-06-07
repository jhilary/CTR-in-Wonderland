import resource
import importlib
import platform
import subprocess


def str_to_function(some_str):
    module_name, _, function_name = some_str.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def get_memory_usage():
    if platform.system() == u'Linux':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return None


def du(path):
    if platform.system() == u'Linux':
        return int(subprocess.check_output(['du','-s', path]).split()[0].decode('utf-8'))
    return None


def format_kb_to_gb(num):
    return "%.4f" % (num / float(1024 * 1024))