from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


#%%
ext_modules = [
    Pybind11Extension(
        "ippCZT.pbIppCZT32fc", # Using dot places it into the folder?
        ["ippCZT/CZT.cpp", "ippCZT/pbCZT.cpp"],
        include_dirs=["../ipp_ext/include"],
        libraries=["ippcore", "ipps"],
        # Define COMPILE_FOR_PYBIND
        extra_compile_args=["-DCOMPILE_FOR_PYBIND"]
    ),
    Pybind11Extension(
        "ippGroupXcorrCZT.pbIppGroupXcorrCZT",
        ["ippCZT/CZT.cpp", 
         "ippGroupXcorrCZT/GroupXcorrCZT.cpp", 
         "ippGroupXcorrCZT/pbGroupXcorrCZT.cpp"],
        include_dirs=["../ipp_ext/include"],
        libraries=["ippcore", "ipps"],
        # Define COMPILE_FOR_PYBIND
        extra_compile_args=["-DCOMPILE_FOR_PYBIND"]
    ),
    Pybind11Extension(
        "frequencyAdjuster.pbffs",
        ["frequencyAdjuster/pbffs.cpp"],
        include_dirs=["frequencyAdjuster/ffs/include"],
        # Define COMPILE_FOR_PYBIND
        extra_compile_args=["-DCOMPILE_FOR_PYBIND"]
    ),
]

setup(
    name="pydspPybinds", 
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
