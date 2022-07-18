import pandas as pd
import numpy as np
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.io import output_notebook, export_png
from bokeh.palettes import linear_palette as palette
from bokeh.layouts import gridplot
import itertools
import sys
import ray
plt.style.use('seaborn-white')
# Set up basic style of bokeh plotting
# output_notebook()
def style(p, autohide=False):
    p.title.text_font="Helvetica"
    p.title.text_font_size="13px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"

    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p
# ray.init()


xi = np.random.normal(0.0, 1.0, size=(10000, 20))
truebetas = .5*np.ones(shape=(20,))
trueyis = np.matmul(truebetas, np.transpose(xi)) + 2
yi = trueyis + np.random.normal(0.0, .01, size=(10000,))

def loss(sig, lam=1e-4):
    sumi = (lam/2)*(sig.T @ sig)
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    netsum = np.mean((xi_til @ sig - yi)**2)/2
    sumi += netsum

    return sumi

def dfloss(sig, lam=1e-4):
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    derivs  = (xi_til @ sig - yi) @ xi_til

    derivs += lam * sig.T

    return derivs/xi.shape[0]

def psi(sig, i, lam=1e-4):
    sumi = lam*(sig.T @ sig) + ((sig.T @ np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dfpsi(sig, j, lam=1e-4):
    largei = np.append(xi[j], 1)
    derivs = (sig.T @ largei - yi[j])*largei
    derivs += lam * sig.T

    return derivs

# @ray.remote
def sgdescent(f, df, x0, etait, \
epochs=200, miter=100, tau=1e-4, fixval = 1/100, fixiter=100):
    k = 0
    maxes = 100
    xk = np.copy(x0)
    xhist = np.array([xk])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    ahist = np.array([loss(xk)])
    olosses = np.array([np.linalg.norm(dfloss(xk), ord=2)/np.linalg.norm(xk, ord=2)])

    while losses[-1] > tau:
        change = np.copy(xk)
        # alfk = fixval

        if k >= fixiter:
                alfk = etait(k)
        else:
            alfk = fixval

        for m in range(maxes):
            chosen = np.random.choice(xi.shape[0], maxes, replace=False)
            minibatch = 0
            for j in range(miter):
                minibatch += alfk*df(xk, chosen[m])

            xk -= minibatch/miter

        # print(change)
        # print(xk)
        # print(dfloss(xk))
        # print(np.linalg.norm(dfloss(xk), ord=2))

        xhist = np.append(xhist, [xk], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])
        olosses = np.append(olosses, [np.linalg.norm(dfloss(xk), ord=2)/np.linalg.norm(xk, ord=2)])
        ahist = np.append(ahist, [loss(xk)])

        change -= xk

        k += 1
        if k == epochs or np.log10(xk[0]) > 20:
            print("Failed to converge")
            break

    return xhist, losses, ahist, olosses

def firsteta(it):
    return 1/it

# @ray.remote
def svrg(xk, num_epochs, epoch_size, step_size, batch_size, \
minibatch_size, tau=1e-4):
    xs = np.copy(xk)
    change = xk+1
    mu = 0
    olosses = np.array([np.linalg.norm(dfpsi(xs, np.random.choice(int(epoch_size))), ord=2)])
    losses = np.array([np.linalg.norm(change, ord=2)])
    ahist = np.array([loss(xs)])
    xhist = np.array([xs])

    for s in range(num_epochs):
        change = np.copy(xs)
        epoch_choice = np.random.choice(xi.shape[0], batch_size, replace=False)

        mu *= 0
        saved = np.array([xs - step_size*mu])
        for i in range(1, epoch_choice.shape[0]):
            mu += dfpsi(xs, i)
        mu /= epoch_choice.shape[0]

        for iter in range(epoch_size):
            select = np.random.choice(xi.shape[0], minibatch_size, replace=False)
            bmu = 0
            for i in range(1, minibatch_size):
                bmu += dfpsi(saved[iter-1], select[i]) - dfpsi(xs, select[i])
            bmu /= minibatch_size
            diff = mu + bmu

            saved = np.append(saved, [saved[iter-1]-diff], axis=0)

        olosses = np.append(olosses, [np.linalg.norm(mu, ord=2)/np.linalg.norm(xs, ord=2)])

        xs = saved[np.random.choice(int(epoch_size))]

        xhist = np.append(xhist, [xs], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xs, ord=2)])
        ahist = np.append(ahist, [loss(xs)])

        if losses[-1] < tau:
            break

    return xhist, losses, ahist, olosses

def phi(f, xk, pk, a):
    return f(xk+a*pk)

def phipr(df, xk, pk, a):
    return (df(xk + a*pk).T @ pk)

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

# @ray.remote
def steepestdescent(f, df, x0, tau, alf0, mu1, rho, \
fixval = 1/100, fixiter = 100, epochs =1100):
    k = 0
    xk = x0
    xhist = np.array([xk])
    ahist = np.array([f(xk)])
    alfk = alf0
    dk = df(xk)
    pk = -dk
    olosses = np.array([np.linalg.norm(df(xk), ord=2)])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])

    # while np.linalg.norm(df(xk), ord=2)/np.linalg.norm(xk, ord=2) > tau:
    while np.linalg.norm(dfloss(xk), ord=2)/np.linalg.norm(xk, ord=2) > tau:
        change = np.copy(xk)

        if k != 0:
            dk = df(xk)
            pk = -dk
        # alfk = backtracing(f, df, xk, pk, mu1, alfk, rho)
        # print(alfk)
        # alfk = fixval
        if k > fixiter:
            alfk = 1/k
        else:
            alfk = fixval
        xk = xk + alfk*pk

        xhist = np.append(xhist, [xk], axis=0)
        ahist = np.append(ahist, [f(xk)])
        olosses = np.append(olosses, [np.linalg.norm(df(xk), ord=2)/np.linalg.norm(xk, ord=2)])
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])

        k += 1
        if k == epochs:
            print("Failed to converge")
            break
    return xhist, losses, ahist, olosses

def optimize_schedule():
    bestconverge = 1000
    bestloss = 1
    bestschedule = 1/1000
    bestdenom = 1000
    bestiter = 100
    initialguess = np.ones(shape=(xi[1].shape[0]+1,))
    results = {}


    for i in range(1, 101):
        test = 1/i
        results[i] = []
        for j in range(1, 20):
            results[i].append(sgdescent.remote(psi, dfpsi, initialguess, \
            firsteta, fixval = test, fixiter = j, epochs=200))

    for i in range(1, 101):
        print(i)
        test = 1/i
        for j in range(1, 20):
            gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = \
            ray.get(results[i][j-1])

            if gdsigolosses[-1] < bestloss and \
            np.log10(gdsigmafound[-1][0]) < 20:
                print(i)
                print(gdsigolosses[-1])
                print(j)
                bestconverge = gdsiglosses.shape[0]
                bestschedule = test
                bestdenom = i
                bestiter = j
                bestloss = gdsigolosses[-1]

    print(bestschedule)
    print(bestdenom)
    print(bestiter)
    # return test

def main():
    eta = 1/(4*200001)
    n = 10000
    b = 100
    m = 100
    s = 100
    initialguess = np.ones(shape=(xi[1].shape[0]+1,))

    print("SVRG")
    test_start_iter = timeit.default_timer()
    sigfound, siglosses, sigfuncs, sigolosses = svrg(initialguess, \
    s, m, eta, n, b, tau=1e-4)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()

    print("GD")
    test_start_iter = timeit.default_timer()
    gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = \
    steepestdescent(loss, dfloss, initialguess, 1e-4, 1, 1e-5, \
    .5, fixval = 1, fixiter = 3, epochs=s)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()

    print("SGD")
    test_start_iter = timeit.default_timer()
    sigsgdfound, sigsgdlosses, sigsgdfuncs, sigsgdolosses = \
    sgdescent(psi, dfpsi, initialguess, firsteta, fixval=1/72, \
    fixiter=19, epochs=s)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()

    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (18, 18),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(1, 3)

        for j in range(3):
            if j == 0:
                # plt.setp(axs[j], title = 'SGD')
                plt.setp(axs[j], ylabel='L(w) Values')
            elif j == 1:
                # plt.setp(axs[0, j], title = 'Naive SVRG')
                plt.setp(axs[j], ylabel=r'||$\nabla$L(w)|| Values')
            else:
                # plt.setp(axs[0], title = 'Random SVRG')
                plt.setp(axs[j], ylabel='||Δw|| Values')
            plt.setp(axs[:], xlabel=r'Major Epochs (# of $\nabla$Ψ(w)/m)')

        axs[0].plot(range(gdsiglosses.shape[0]), gdsigfuncs, '-', \
        alpha=.6, label=f'Gradient Descent')
        axs[0].set_yscale('log')
        #
        axs[1].plot(range(gdsiglosses.shape[0]), gdsigolosses, '-',
        alpha=.6, label=f'Gradient Descent')
        axs[1].set_yscale('log')
        #
        axs[2].plot(range(gdsiglosses.shape[0]), gdsiglosses, '-',
        alpha=.6, label=f'Gradient Descent')
        axs[2].set_yscale('log')


        print(siglosses.shape[0])
        print(gdsiglosses.shape[0])
        print(sigsgdlosses.shape[0])
        axs[0].plot(range(0, siglosses.shape[0]*2, 2), \
        sigfuncs, '-.', markeredgecolor="none", alpha=.6, label=f'SVRG')

        axs[1].plot(range(0, siglosses.shape[0]*2, 2), \
        sigolosses, '-.', markeredgecolor="none", alpha=.6, label=f'SVRG')

        axs[2].plot(range(0, siglosses.shape[0]*2, 2), \
        siglosses, '-.', markeredgecolor="none", alpha=.6, label=f'SVRG')

        axs[0].plot(range(0, sigsgdlosses.shape[0], 1), \
        sigsgdfuncs, '-.', alpha=.8, label=f'SGD')

        axs[1].plot(range(0, sigsgdlosses.shape[0], 1), \
        sigsgdolosses, '-.', alpha=.8, label=f'SGD')

        axs[2].plot(range(0, sigsgdlosses.shape[0], 1), \
        sigsgdlosses, '-.', alpha=.8, label=f'SGD')

        plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("toy_idealnewparam.png", bbox_inches='tight')


if __name__ == "__main__":
    # optimize_schedule()
    main()
