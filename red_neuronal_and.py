import numpy as np


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])
y = np.array([0, 0, 0, 1])


epochs = 10
lr = 0.1 



Xb = np.hstack([X, np.ones((X.shape[0], 1))])



np.random.seed(42)
w = np.random.uniform(-0.5, 0.5, size=(Xb.shape[1],))



def step(z):
    return 1 if z >= 0 else 0

for epoch in range(epochs):
    errors = 0
    for xi, yi in zip(Xb, y):
        z = np.dot(xi, w)
        yout = step(z)
        delta = yi - yout
        if delta != 0:
            w += lr * delta * xi
            errors += 1

    print(f"Epoch {epoch+1}/{epochs}, Errores: {errors}")


def predict(x):
    xb = np.append(x,1)
    z = np.dot(xb, w)
    return step(z)

for x in X:
    print(f"Entrada: {x}, Predicción: {predict(x)}")