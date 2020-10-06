import numpy as np

a = np.array((0, 1,31,0,0,0,31,31,0,0,0,0,0,0,0,0))
print(np.trim_zeros(a,'b'))


localization = np.zeros((10, 3))
localization[0,:] = 1
print(localization)

int32[] data