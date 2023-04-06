import os
if os.name == 'nt': # Load the directory on windows
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist', 'intel64')) # Configure IPP dll reliance

# Import the cythonized pyd directly into namespace
from .CyIppXcorrFFT import CyIppXcorrFFT # Don't forget the . because we are name mangling with the folder
