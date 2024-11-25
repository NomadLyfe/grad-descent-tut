import numpy as np

x = np.random.random(100) #array
y = 2*x + 0.1 #array
# cost_func = sum((y - yhat)**2) / n
m = 0
b = 0
n = x.size
learning_rate = 0.1

def descend(x, y, m, b, learning_rate):
    ddm = 0
    ddb = 0
    for xi, yi in zip(x, y):
        ddm += -2*xi*(yi-m*xi-b)
        ddb += -2*(yi-m*xi-b)
    m = m - learning_rate*(ddm)*(1/n)
    b = b - learning_rate*(ddb)*(1/n)
    return m, b

for epoch in range(200):
    m, b = descend(x, y, m, b, learning_rate)
    yhat = m*x + b #array
    cost_func = (y-yhat)**2 #array
    loss = np.sum(cost_func) / n
    print(f'Epoch {epoch} results: loss of {loss}, m of {m}, and b of {b}')