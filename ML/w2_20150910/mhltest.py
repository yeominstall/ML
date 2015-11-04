import numpy as np
from scipy import linalg

x = [[0.96,0.69], [0.69, 0.96]]
xcov = linalg.inv(np.array(x).T)
print xcov

