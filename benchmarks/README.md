# benchmark_xcorrs

```bash
python benchmark_xcorrs.py 1000000 128
```

1. Pythonic loops+Numpy: 15.9s.
2. Cython (4 threads): 3.9s.
3. Cupy: 0.29s.

This length of 1M is about where Cupy starts to become more worth it than Cython.

