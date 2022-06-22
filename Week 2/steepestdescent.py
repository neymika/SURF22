import pandas as pd
import numpy as np
from scipy import optimize
import timeit

normtable = pd.read_csv("data_normalized.csv", header=None)
xi = np.array([normtable.iloc[0, 0:48]])
yi = np.array([normtable.iloc[0, 48]])
for i in range(1, 200):
    xi = np.append(xi, [normtable.iloc[i, 0:48]], axis=0)
    yi = np.append(yi, [normtable.iloc[i, 48]], axis=0)

def jacob(sig, lam=10e-2):
    c = (lam/2)*np.matmul(np.transpose(sig), sig)
    sumi = c
    netsum = 0

    for i in range(xi.shape[0]):
        ho = np.matmul(np.transpose(sig), np.append(xi[i], 1))
        netsum += (ho - yi[i])**2
    sumi += netsum/(2*xi.shape[0])

    return sumi

def dfjacob(sig, lam=10e-2):
    derivs = np.zeros(shape=(sig.shape[0],))

    for j in range(xi.shape[0]):
        largei = np.append(xi[j], 1)
        derivs += np.matmul(np.transpose(sig), largei)*largei - yi[j]*largei

    derivs += lam*np.transpose(sig)

    return derivs/xi.shape[0]

def fi(sig, i, lam=10e-2):
    c = lam*np.matmul(np.transpose(sig), sig)
    sumi = c + (np.matmul(np.transpose(sig), np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dffi(sig, j, lam=10e-2):
    derivs = np.zeros(shape=(sig.shape[0],))
    largei = np.append(xi[j], 1)
    derivs = np.matmul(np.transpose(sig), largei)*largei - yi[j]*largei

    derivs += lam*np.transpose(sig)

    return derivs

def phi(f, xk, pk, a):
    return f(xk+a*pk)

def phipr(df, xk, pk, a):
    return np.matmul(np.transpose(df(xk + a*pk)), pk)

def backtracing(f, df, xk, pk, mu1, alf0, rho):
    alf = alf0
    ahist = alf0

    for i in range(100):
        if phi(f, xk, pk, alf) <= phi(f, xk, pk, 0) + mu1*alf*phipr(df, xk, pk, 0):
            break
        alf = rho*alf
        ahist = np.append(ahist, alf)
        if i == 99:
            print('Backtracking exited without satsifying Armijo condition.')
            return alf

    return alf

def steepestdescent(f, df, x0, tau, alf0, mu1, rho):
    k = 0
    xk = x0
    xhist = np.array([xk])
    ahist = f(xk)
    alfk = alf0
    dk = df(xk)
    pk = -dk
    reset = False
    losses = np.array([np.linalg.norm(df(xk), ord=2)])


    while np.linalg.norm(df(xk), ord=2)/np.linalg.norm(xk, ord=2) > tau:
        if k != 0:
            dk = df(xk)
            pk = -dk
        alfk = backtracing(f, df, xk, pk, mu1, alfk, rho)
        xk = xk + alfk*pk

        xhist = np.append(xhist, [xk], axis=0)
        ahist = np.append(ahist, [f(xk)])
        losses = np.append(losses, [np.linalg.norm(df(xk), ord=2)/np.linalg.norm(xk, ord=2)])

        k += 1
        if k == 1100:
            print("Failed to converge")
            break

    return xhist, losses, ahist

def main():
    test_start_iter = timeit.default_timer()
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    sigmafound, siglosses, sigfuncs = steepestdescent(jacob, dfjacob, initialguess, .01, 1, 1e-5, .5)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Steepest Descent Values")
    print(sigmafound[-1])
    print(sigfuncs[-1])
    print(siglosses[-1])

    xib = np.concatenate((xi, np.ones((xi.shape[0], 1))), 1)
    d = np.matmul(np.transpose(xib), xib)+ 0.0001*np.identity(xib.shape[1])
    theta_star =np.linalg.lstsq(d, np.matmul(np.transpose(xib), yi), rcond=None)
    true_objective = jacob(theta_star[0])

    print("A\\b Values")
    print(theta_star[0])
    print(true_objective)

if __name__ == "__main__":
    main()
