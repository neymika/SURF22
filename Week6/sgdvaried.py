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

        if k >fixiter:
            alfk = fixval/(k-fixiter)
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
        if k == epochs:
            print("Failed to converge")
            break

    return xhist, losses, ahist, olosses

def firsteta(it):
    return 1/it

def main():
    s = 100

    print("Performing SGD at multiple different stepsizes")
    stepsizes = [1, .01, .001, .0001, .00001]
    decayschedule = [5, 10, 20, 50, 100]
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))

    histsigsgdfound = {}
    histsigsgdlosses = {}
    histsigsgdfuncs = {}
    histsigsgdolosses = {}
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 30),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(5, 5)

        for i in range(len(stepsizes)):
            for j in range(len(stepsizes)):
                print(f'SGD with step size = {stepsizes[i]} and decay epoch = {decayschedule[j]}')
                test_start_iter = timeit.default_timer()
                sigsgdfound, sigsgdlosses, sigsgdfuncs, sigsgdolosses = \
                sgdescent(psi, dfpsi, initialguess, firsteta, fixval=stepsizes[i], \
                fixiter=decayschedule[j], epochs=s)
                test_end_iter = timeit.default_timer()
                print(test_end_iter - test_start_iter)
                print(sigsgdolosses[-1])
                print(sigsgdfuncs[-1])
                print()

                dictkey = (i+1)*10 + j
                histsigsgdfound[dictkey] = sigsgdfound
                histsigsgdlosses[dictkey] = sigsgdlosses
                histsigsgdfuncs[dictkey] = sigsgdfuncs
                histsigsgdolosses[dictkey] = sigsgdolosses

                # axs[i, j].plot(range(sigsgdfuncs.shape[0]), sigsgdolosses, '-')
                # axs[i, j].plot(range(sigsgdfuncs.shape[0]), sigsgdfuncs, '-')
                # axs[i, j].set_yscale('log')
                # axs[i, j].set_title(\
                # rf'$\eta$: {stepsizes[i]}, $\eta = 1/s$ Starting Epoch: {decayschedule[j]}')


        # plt.setp(axs[-1, :], xlabel='Epochs')
        # # plt.setp(axs[:, 0], ylabel=r'||$\nabla$L(w)|| Values')
        # plt.setp(axs[:, 0], ylabel=r'L(w) Values')
        #
        #
        # # plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Tradeoff Analysis for SGD")
        # plt.savefig("toy_sgdtradeofffunc.png", bbox_inches='tight')

        dffuncs = dict([ (k,pd.Series(v)) for k,v in histsigsgdfuncs.items() ])
        dffuncs = pd.DataFrame(data=dffuncs)
        dflosses = dict([ (k,pd.Series(v)) for k,v in histsigsgdlosses.items() ])
        dflosses = pd.DataFrame(data=dflosses)
        dfolosses = dict([ (k,pd.Series(v)) for k,v in histsigsgdolosses.items() ])
        dfolosses = pd.DataFrame(data=dfolosses)

        dffuncs.to_csv("sgdfunc.csv", index=False)
        dflosses.to_csv("sgdlosses.csv", index=False)
        dfolosses.to_csv("sgdolosses.csv", index=False)

if __name__ == "__main__":
    main()
