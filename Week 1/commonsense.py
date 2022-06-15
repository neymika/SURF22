import numpy as np

np.random.seed(2022)
dataxis = np.random.normal(0.0, 1.0, size=(20, 1000))
truebetas = .5*np.ones(shape=(20,))
trueyis = np.matmul(truebetas, dataxis)
noisyyi = trueyis + np.random.normal(0.0, .01, size=(1000,))
estbetas = np.matmul(np.matmul(np.linalg.inv(np.matmul(dataxis, np.transpose(dataxis))), dataxis), noisyyi)
npbetas = np.linalg.lstsq(np.transpose(dataxis), noisyyi)

# common sense check
print(estbetas - npbetas[0])
print(npbetas)
print()
print("True Betas")
print(truebetas)
