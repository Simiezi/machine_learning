from itertools import combinations
import numpy as np


comb = combinations([1, 2, 3], 2)
print(list(comb))
x = np.array()
y = np.array()
mask = np.array(list(range(len(x))))
np.random.shuffle(mask)
x[mask]
y[mask]