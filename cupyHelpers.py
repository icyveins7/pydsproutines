
import cupy as cp
import os

#%% Convenience functions
def cupyModuleToKernelsLoader(modulefilename: str, kernelNames: list, options: tuple=('-std=c++17',)):
    """
    Helper function to generate the CuPy kernel objects from a module.
    The module is expected to reside in the custom_kernels folder.

    Examples:
        kernel1, kernel2 = cupyModuleToKernelsLoader("mymodule.cu", ["mykern1","mykern2"])
        kernel1, = cupyModuleToKernelsLoader("mymodule.cu", "mykern1")

    Parameters
    ----------
    modulefilename : str
        Name of the module file.
    kernelNames : list
        List of kernel names. These are to include the templated types if the kernels are templated.
        Example: ["mykern1", "mykern2<float>", "mykern2<double>"]
    options: tuple
        Compiler options as a tuple of arguments.

    Returns
    -------
    kernels : list
        List of kernels that can be invoked directly. You should probably unpack it and name them in the same
        order as was input in kernelNames.
    _module : cupy.RawModule
        The RawModule object. Can be used to access global symbols if necessary (usually not).
    """
    if isinstance(kernelNames, str):
        kernelNames = [kernelNames]
    kernels = []
    with open(os.path.join(os.path.dirname(__file__), "custom_kernels", modulefilename), "r") as fid:
        _module = cp.RawModule(code=fid.read(), options=options,
                               name_expressions=kernelNames)
        for kernelName in kernelNames:
            kernels.append(_module.get_function(kernelName))

    return kernels, _module

def cupyRequireDtype(dtype: type, var: cp.ndarray):
    """
    Example: cupyRequireDtype(cp.uint32, myarray)
    """
    if var.dtype != dtype:
        raise TypeError("Must be %s, found %s" % (dtype, var.dtype))
    
def cupyCheckExceedsSharedMem(requestedBytes: int, maximumBytes: int=48000):
    if requestedBytes > maximumBytes:
        raise MemoryError("Shared memory requested %d bytes exceeds maximum %d bytes" % (requestedBytes, maximumBytes))

def requireCupyArray(var: cp.ndarray):
    if not isinstance(var, cp.ndarray):
        raise TypeError("Must be cupy array.")
    
def cupyGetEnoughBlocks(length: int, computedPerBlock: int):
    """
    Gets just enough blocks to cover a certain length.
    Assumes every block will compute 'computedPerBlock' elements.
    """
    NUM_BLKS = length // computedPerBlock
    NUM_BLKS = NUM_BLKS if NUM_BLKS % computedPerBlock == 0 else NUM_BLKS + 1
    return NUM_BLKS

