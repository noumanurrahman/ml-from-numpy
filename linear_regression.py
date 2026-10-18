import numpy as np

N_SAMPLES = 5000
N_FEATURES = 10
NOISE_STD = 0.5
SEED = 42

rng = np.random.default_rng(SEED)

X = rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, N_FEATURES))

true_w = rng.uniform(-5, 5, size=(N_FEATURES,))
true_b = 3.0

noise = rng.normal(0.0, NOISE_STD, size=(N_SAMPLES,))
Y = X @ true_w + true_b + noise

def model(theta):
    return X @ theta[1:] + theta[0]

def loss(theta):
    Y_pred = model(theta)
    Y_loss = np.square(Y_pred - Y)
    return np.sum(Y_loss) / N_SAMPLES

theta = np.zeros((N_FEATURES+1,), dtype=float)
print(f"RMSE before: {np.sqrt(loss(theta))}")

def gradient(theta):
    h = model(theta)
    dw = 2 / N_SAMPLES * (X.T @ (h - Y))
    db = 2 / N_SAMPLES * (np.sum(h - Y))
    gradient = np.r_[db, dw]
    return gradient

def descent(learning_rate, initial_params, steps):
    theta = initial_params
    for i in range(steps + 1):
        theta -= learning_rate * gradient(theta)
    return theta

descent(0.1, theta, 2000)
print(f"RMSE after: {np.sqrt(loss(theta))}")
print(f"Actual bias: {true_b}")
print(f"Actual weights: ")
print(true_w)
print(f"Gradient bias: {theta[0]}")
print(f"Gradient weights: ")
print(theta[1:])
print(f"Change in bias: {true_b - theta[0]}")
print(f"Change in weights: ")
print(true_w - theta[1:])