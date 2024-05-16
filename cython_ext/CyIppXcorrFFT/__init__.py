from intelHelpers import include_ipp
import os
if os.name == 'nt': # Load the directory on windows
    include_ipp()

# Import the cythonized pyd directly into namespace
try:
    from .CyIppXcorrFFT import CyIppXcorrFFT # Don't forget the . because we are name mangling with the folder
except Exception as e:
    print("Cythonised module CyIppXcorrFFT not found. Please compile it: %s" % str(e))
