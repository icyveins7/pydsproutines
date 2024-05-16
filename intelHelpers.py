import os


def include_ipp():
    """
    Extracts the IPP redist/bin directory defined in PATH by
    the environment variable setter in oneAPI, then adds it as
    a dll directory in Python.
    """
    ipp_paths = [
        x for x in os.environ['PATH'].split(";") if 'ipp' in x
    ]
    for ipp_path in ipp_paths:
        os.add_dll_directory(ipp_path)
