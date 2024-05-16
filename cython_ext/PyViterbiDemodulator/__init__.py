import os
from intelHelpers import include_ipp
if os.name == 'nt': # Load the directory on windows
    include_ipp()

# Import the cythonized pyd directly into namespace
try:
    from .PyViterbiDemodulator import PyViterbiDemodulator # Don't forget the . because we are name mangling with the folder
except Exception as e:
    print("Cythonised module PyViterbiDemodulator not found. Please compile it: %s" % str(e))
