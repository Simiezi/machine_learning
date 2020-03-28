import numpy as np
import matplotlib.pyplot as plt

width = 0.3
conts = [63.34243, 92.428756, 86.52342]
weights = [3.65672, 120.23482, 299.087659]
func_names = ['sin(x)', 'e^x']
for i, j in zip(weights, func_names):
    print(f'{i} \n')
    print(f'{j} \n')
reg1 = ''.join([f'{w:.2f} * {name}+' for w, name in zip(weights, func_names)]) + f'{weights[-1]:.2f}'
print(reg1)
labels = [reg1, 'Регр2', 'Регр3']
bin_positions = np.array(list(range(len(conts))))
bins_avt = plt.bar(bin_positions, conts, width, label='Точность на обуч выборке')
plt.show()