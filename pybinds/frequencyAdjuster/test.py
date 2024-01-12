from pbffs import ffs1_64fc, ffs4_64fc
import numpy as np

x = np.random.randn(32) + 1j*np.random.randn(32)
y = x.copy()
# print(x)

freq = np.random.rand()
phi = np.random.rand()

ffs4_64fc(y, freq, phi)
# print(y)

manual = x * np.exp(1j*(2*np.pi*freq*np.arange(x.size) + phi))
np.testing.assert_allclose(manual, y)