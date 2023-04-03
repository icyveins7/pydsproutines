Structure of this subfolder is as follows:

1. Create subfolder for your .c (ctypes) or .cpp (Cython) module.
2. Place the contents of everything inside this subfolder, including any compilation batch scripts or setup.py.
3. Place a '\_\_init\_\_.py' inside this subfolder. Within it, reference the .pyd or .dll module and import the name of the class/function directly, e.g.

```python
# Inside cython_ext/subfolder/__init__.py
from .MyModuleName import MyModuleName # This is the usual scenario where the .pyd shares the same name as the class for example
```

4. Then the module should be accessible by doing the following:

```python
from cython_ext.subfolder import MyModuleName

a = MyModuleName(...constructor arguments...)
```