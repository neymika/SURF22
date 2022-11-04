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

def calc_rho(high_eval, low_eval, cos=False):
    denom = np.linalg.norm(high_eval, ord=2) * np.linalg.norm(low_eval, ord=2)
    if cos:
        return np.dot(high_eval, low_eval)/denom
    else:
        return np.linalg.norm(high_eval - low_eval, ord=2)/denom

# num_epochs = S, batch_size = N = m1, epoch_size = m, minibatch_size = B = m2
# @ray.remote
def svrg(xk, num_epochs, epoch_size, step_size, batch_size, \
minibatch_size, tau=1e-4, fixiter=100, fixval=1/10, p = 250000, c_one = 10000, \
c_two = 100):
    xs = np.copy(xk)
    change = xk+1
    mu = 0
    olosses = np.array([np.linalg.norm(dfpsi(xs, np.random.choice(int(epoch_size))), ord=2)])
    losses = np.array([np.linalg.norm(change, ord=2)])
    ahist = np.array([dfloss(xs)])
    psihist = np.array([np.zeros(shape=ahist[-1].shape)])
    xhist = np.array([xs])
    costs = np.array([c_one, c_two])
    alfhist = np.array([dfloss(xs)])

    for s in range(num_epochs):
        change = np.copy(xs)
        epoch_choice = np.random.choice(xi.shape[0], batch_size, replace=False)

        if s > fixiter:
            alfk = step_size
        else:
            alfk = fixval

        mu *= 0
        for i in range(batch_size):
            mu += dfpsi(xs, epoch_choice[i])
        mu /= epoch_choice.shape[0]
        r_one = 1
        num_two = c_one*(calc_rho(alfhist[-1], mu)**2)
        denom_two = c_two*(1-calc_rho(alfhist[-1], mu)**2)
        r_two = np.sqrt(num_two/denom_two)
        m_one = p/(costs.T @ np.array([r_one, r_two]))
        m_two = m_one * r_two
        print(f"num {num_two}")
        print(f"denom {denom_two}")
        print(f"r_one: {r_one} r_two: {r_two}")
        print(f"m_one: {m_one} m_two: {m_two}")
        print()
        if np.isnan(r_two) or m_two < 100:
            minibatch_size_check = minibatch_size
        else:
            minibatch_size_check = int(np.ceil(m_two))

        saved = np.array([xs - alfk*mu])

        for iter in range(epoch_size):
            select = np.random.choice(xi.shape[0], minibatch_size_check, replace=False)
            bmu = 0
            for i in range(1, minibatch_size_check):
                bmu += dfpsi(saved[iter-1], select[i]) - dfpsi(xs, select[i])
            bmu /= minibatch_size_check
            alfmf = calc_rho(alfhist[-1], mu)*np.std(ahist[-1])/np.std(mu)
            diff = mu + bmu

            saved = np.append(saved, [saved[iter-1]-alfk*diff], axis=0)

        olosses = np.append(olosses, [np.linalg.norm(mu, ord=2)])

        xs = saved[np.random.choice(int(epoch_size))]

        xhist = np.append(xhist, [xs], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xs, ord=2)])
        alfhist = np.append(alfhist, [dfloss(xs)], axis=0)
        ahist = np.append(ahist, [loss(xs)])

        if olosses[-1] < tau:
            break

        if np.log10(ahist[-1]) > 30 or np.log10(ahist[-1]) < -30:
            print("Failed to converge")
            break

    return xhist, losses, ahist, olosses

def firsteta(it):
    return 1/it

def main():
    s = 100

    print("Performing SVRG at multiple different stepsizes")
    stepsizes = [1, .1, .01, .001, .0001, .00001, 1/(4*1800001)]
    decayschedule = [1, 5, 10, 20, 50, 100]
    eta = 1/(4*200001)
    n = 10000
    b = 100
    m = 100
    s = 200
    initialguess = np.ones(shape=(xi[1].shape[0]+1,))

    histsigsvrgfound = {}
    histsigsvrglosses = {}
    histsigsvrgfuncs = {}
    histsigsvrgolosses = {}
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (35, 35),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(1, len(stepsizes))

        for i in range(len(stepsizes)):
            print(f'SVRG with step size = {stepsizes[i]}')
            test_start_iter = timeit.default_timer()
            sigfound, siglosses, sigfuncs, sigolosses = \
            svrg(initialguess, s, m, eta, n, b, tau=1e-4, fixval=stepsizes[i])
            test_end_iter = timeit.default_timer()
            print(test_end_iter - test_start_iter)
            print(sigolosses[-1])
            print(sigfuncs[-1])
            print()

            dictkey = (i+1)*10
            histsigsvrgfound[dictkey] = sigfound
            histsigsvrglosses[dictkey] = siglosses
            histsigsvrgfuncs[dictkey] = sigfuncs
            histsigsvrgolosses[dictkey] = sigolosses

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

        dffuncs = dict([ (k,pd.Series(v)) for k,v in histsigsvrgfuncs.items() ])
        dffuncs = pd.DataFrame(data=dffuncs)
        dflosses = dict([ (k,pd.Series(v)) for k,v in histsigsvrglosses.items() ])
        dflosses = pd.DataFrame(data=dflosses)
        dfolosses = dict([ (k,pd.Series(v)) for k,v in histsigsvrgolosses.items() ])
        dfolosses = pd.DataFrame(data=dfolosses)

        dffuncs.to_csv("svrgeucfunc.csv", index=False)
        dflosses.to_csv("svrgeuclosses.csv", index=False)
        dfolosses.to_csv("svrgeucolosses.csv", index=False)

if __name__ == "__main__":
    main()
