import pandas as pd
import numpy as np
from scipy import optimize

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

def steepestdescent(f, df, x0, tau, alf0, mu1, mu2, sigma):
    k = 0
    xk = x0
    xhist = np.array([xk])
    alfinit = alf0
    ahist = alf0
    alfk = [0]
    dk = df(xk)
    pk = -dk/np.linalg.norm(dk)
    phist = np.array([pk])
    fhist = np.array([dk])
    reset = False
    losses = np.array([np.linalg.norm(df(xk), ord=2)])


    while np.linalg.norm(df(xk), ord=2) > tau:
        if k != 0:
            dk = df(xk)
            pk = -dk/np.linalg.norm(dk)
#             print(np.matmul(np.transpose(fhist[k-1]), phist[k-1]))
#             print(alfk)
#             alfinit = alfk[0]*(np.matmul(np.transpose(fhist[k-1]), phist[k-1]))/(np.matmul(np.transpose(dk), pk))
#             dphiinit = phipr(df, xk, pk, alfinit)
            # DO LATER
#             bk = (dfk(xko)*dfk(xko))/(dfk(xko)*dfk(xko))
#             pkn = bk*pko - (j+1)*dfk(xko)/np.abs((j+1)*dfk(xko))
#             pko = pkn
#         it, alfk, alfhist = bracketing(f, df, xk, pk, mu1, alfinit, phi0, \
#                                        dphi0, mu2, sigma)
#             alfk = backtracing(f, df, xk, pk, mu1, alfinit, rho)
        alfk = optimize.line_search(f, df, xk, pk, gfk=None, old_fval=None, old_old_fval=None, \
                                    args=(), c1=mu1, c2=mu2, amax=sigma, extra_condition=None, maxiter=20)
        if alfk[0] != None:
            xk = xk + alfk[0]*pk

            xhist = np.append(xhist, [xk], axis=0)

            ahist = np.append([alfk[0]], ahist)
            reset = False
        else:
            alfk = [alf0*.001]
            xk = xk + alfk[0]*pk

            xhist = np.append(xhist, [xk], axis=0)

            ahist = np.append([alfk[0]], ahist)
            reset = True
#             print("WHOOPS")
#             print(alfk)

        if k != 0:
            phist = np.append(phist, [pk], axis=0)
            fhist = np.append(fhist, [dk], axis=0)

        losses = np.append(losses, [np.linalg.norm(df(xk), ord=2)])

        k += 1
        if k == 500:
            print("Failed to converge")
            break

    return xhist, losses

def main():
    normtable = pd.read_csv("data_normalized.csv", header=None)
    xi = np.array([normtable.iloc[0, 0:48]])
    yi = np.array([normtable.iloc[0, 48]])
    for i in range(1, 200):
        xi = np.append(xi, [normtable.iloc[i, 0:48]], axis=0)
        yi = np.append(yi, [normtable.iloc[i, 48]], axis=0)

    test_start_iter = timeit.default_timer()
    sigmafound, siglosses = steepestdescent(jacob, dfjacob, np.ones(shape=(xi[1].shape[0]+1,)), 1e-5, 1, .0001, .9, 2)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    testthetas = np.zeros(shape=(48,))

    for i in range(48):
        testthetas[i] = np.linalg.norm(sigmafound[i], ord=2)*.01

if __name__ == "__main__":
    main()
