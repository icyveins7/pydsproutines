import os
from intelHelpers import include_ipp

if os.name == 'nt': # Load the directory on windows
    include_ipp()

# Import the cythonized pyd directly into namespace
try:
    from .PySampledLinearInterpolator import PySampledLinearInterpolator_64f, PyConstAmpSigLerp_64f
    from .PySampledLinearInterpolator import PySampledLinearInterpolatorWorkspace_64f, PyConstAmpSigLerpBursty_64f, PyConstAmpSigLerpBurstyMulti_64f
except Exception as e:
    print("Cythonised module PySampledLinearInterpolator not found. Please compile it: %s" % str(e))
