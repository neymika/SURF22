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
    derivs /= xi.shape[0]
    derivs += lam * sig.T

    return derivs

def psi(sig, i, lam=1e-4):
    sumi = lam*(sig.T @ sig) + ((sig.T @ np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dfpsi(sig, j, lam=1e-4):
    largei = np.append(xi[j], 1)
    derivs = (sig.T @ largei - yi[j])*largei
    derivs += lam * sig.T

    return derivs

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
    while np.linalg.norm(dfloss(xk), ord=2) > tau:
        change = np.copy(xk)

        if k != 0:
            dk = df(xk)
            pk = -dk
        # alfk = backtracing(f, df, xk, pk, mu1, alfk, rho)
        # print(alfk)
        # alfk = fixval
        if k > fixiter:
            alfk = fixval/(k-fixiter)
        else:
            alfk = fixval
        xk = xk + alfk*pk

        xhist = np.append(xhist, [xk], axis=0)
        ahist = np.append(ahist, [f(xk)])
        olosses = np.append(olosses, [np.linalg.norm(df(xk), ord=2)])
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])

        k += 1
        if k == epochs:
            print("Failed to converge")
            break
    return xhist, losses, ahist, olosses

def firsteta(it):
    return 1/it

def main():
    s = 100

    print("Performing GD at multiple different stepsizes")
    stepsizes = [1, .1, .01, .001, .0001, .00001]
    decayschedule = [1, 5, 10, 20, 50, 100]
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    npbetas = np.linalg.lstsq(xi, yi)
    histsigfound = {}
    histsiglosses = {}
    histsigfuncs = {}
    histsigolosses = {}
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 30),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(5, 5)

        for i in range(len(stepsizes)):
            for j in range(len(decayschedule)):
                print(f'GD with step size = {stepsizes[i]} and decay epoch = {decayschedule[j]}')
                test_start_iter = timeit.default_timer()
                gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = \
                steepestdescent(loss, dfloss, initialguess, 1e-4, 1, 1e-5, \
                .5, fixval = stepsizes[i], fixiter = decayschedule[j], epochs=s)
                test_end_iter = timeit.default_timer()
                print(test_end_iter - test_start_iter)
                print(f'Norm grad losses: {gdsigolosses[-1]}')
                print(f'losses: {gdsigfuncs[-1]}')
                print(f'W found: {gdsigmafound[-1]}')
                print(f'LSTSQ found w {npbetas[0]}')
                print()

                dictkey = (i+1)*10 + j
                histsigfound[dictkey] = gdsigmafound
                histsiglosses[dictkey] = gdsiglosses
                histsigfuncs[dictkey] = gdsigfuncs
                histsigolosses[dictkey] = gdsigolosses

                # axs[i, j].plot(range(gdsigfuncs.shape[0]), gdsigolosses, '-')
                # axs[i, j].set_yscale('log')
                # axs[i, j].set_title(\
                # rf'$\eta$: {stepsizes[i]}, $\eta = 1/s$ Starting Epoch: {decayschedule[j]}')


        # plt.setp(axs[-1, :], xlabel='Iterations')
        # plt.setp(axs[:, 0], ylabel=r'||$\nabla$L(w)|| Values')
        #
        #
        # # plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Tradeoff Analysis for SGD")
        # plt.savefig("toy_gdtradeoff.png", bbox_inches='tight')

        dffuncs = dict([ (k,pd.Series(v)) for k,v in histsigfuncs.items() ])
        dffuncs = pd.DataFrame(data=dffuncs)
        dflosses = dict([ (k,pd.Series(v)) for k,v in histsiglosses.items() ])
        dflosses = pd.DataFrame(data=dflosses)
        dfolosses = dict([ (k,pd.Series(v)) for k,v in histsigolosses.items() ])
        dfolosses = pd.DataFrame(data=dfolosses)

        dffuncs.to_csv("gdfunc.csv", index=False)
        dflosses.to_csv("gdlosses.csv", index=False)
        dfolosses.to_csv("gdolosses.csv", index=False)


if __name__ == "__main__":
    main()
