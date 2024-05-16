if os.name == 'nt': # Load the directory on windows
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist', 'intel64')) # Configure IPP dll reliance

# Import the cythonized pyd directly into namespace
try:
    from .PySampledLinearInterpolator import PySampledLinearInterpolator # Don't forget the . because we are name mangling with the folder
except Exception as e:
    print("Cythonised module PySampledLinearInterpolator not found. Please compile it: %s" % str(e))