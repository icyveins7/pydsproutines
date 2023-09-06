import os
if os.name == 'nt': # Load the directory on windows
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist', 'intel64')) # Configure IPP dll reliance

# Import the cythonized pyd directly into namespace
try:
    from .pbIppGroupXcorrCZT import pbIppGroupXcorrCZT # Don't forget the . because we are name mangling with the folder
except Exception as e:
    print("pybind module pbIppGroupXcorrCZT not found. Please compile it: %s" % str(e))
