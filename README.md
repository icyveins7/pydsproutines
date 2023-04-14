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

Then make sure to re-set the submodule url with, for example,

```
git submodule set-url ipp_ext git@mygitserver:myuser/ipp_ext.git
git submodule update --init # and other related commands
```
