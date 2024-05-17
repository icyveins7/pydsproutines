# pydsproutines
Python Routines/Classes/Functions for DSP-related things. Includes Ctypes DLLs and Cythonized Functions.

# Installation
Please install via pip in editable mode.

```
git clone https://github.com/icyveins7/pydsproutines.git
cd pydsproutines
pip install -e .
```

## Submodules
When on a local network, all submodules should be cloned and pushed to the internal git server. For now, the following submodules are required:

1. ipp_ext
2. ffs

Then make sure to re-set the submodule url with, for example,

```
git submodule set-url ipp_ext git@mygitserver:myuser/ipp_ext.git
git submodule update --init # and other related commands
```

You can also modify ```.gitmodules``` directly instead of using ```set-url```.

## Cython extensions

The ```cython_ext``` subfolder contains a few extensions built using cython.

IMPORTANT: This requires you to ensure the submodules are set correctly in the previous step.

If you haven't yet, install cython and numpy:

```
pip install cython numpy
```

Then compile all of them by running the build script (only for Windows)

```
build_all.bat
```

## pybinds

TO BE COMPLETED