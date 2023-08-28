from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pbIppCZT32fc",
        ["CZT.cpp", "pbCZT.cpp"],
        include_dirs=["../../ipp_ext/include"],
        libraries=["ippcore", "ipps"],
        # Define COMPILE_FOR_PYBIND
        extra_compile_args=["-DCOMPILE_FOR_PYBIND"]
    ),
]

setup(
    name="pbIppCZT32fc", 
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)