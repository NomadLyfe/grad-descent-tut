import numpy as np

x = np.random.random(100)
y = 2*x + 6

m = 0
b = 0
n = x.size
print(n)
learning_rate = 0.01

def descend(x, y, m, b, learning_rate):
    ddm = 0
    ddb = 0
    for xi, yi in zip(x, y):
        ddm += -2*xi*(yi-m*xi-b)
        ddb += -2*(yi-m*xi-b)
    m = m - learning_rate*(ddm)*(1/n)
    b = b - learning_rate*(ddb)*(1/n)
    return m, b

for epoch in range(3000):
    m, b = descend(x, y, m, b, learning_rate)
    print(f'm: {m}, b: {b}')