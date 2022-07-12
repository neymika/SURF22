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


xi = np.random.normal(0.0, 1.0, size=(1000, 20))
truebetas = .5*np.ones(shape=(20,))
trueyis = np.matmul(truebetas, np.transpose(xi)) + 2
yi = trueyis + np.random.normal(0.0, .01, size=(1000,))

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

def multipsi(sig, inds, lam=1e-4):
    sumi = lam*(sig.T @ sig)
    chosenxi = np.take(xi, inds)
    chosenyi = np.take(yi, inds)
    xi_til = np.hstack((chosenxi, np.ones((chosenxi.shape[0],1))))
    netsum = np.mean((xi_til @ sig - chosenyi)**2)
    sumi += netsum

    return sumi/2

def dfmultipsi(sig, inds, lam=1e-4):
    chosenxi = np.take(xi, inds)
    chosenyi = np.take(yi, inds)
    xi_til = np.hstack((chosenxi, np.ones((chosenxi.shape[0],1))))
    derivs  = (xi_til @ sig - chosenyi) @ xi_til

    derivs += lam * sig.T

    return derivs/chosenxi.shape[0]

def phi(f, xk, pk, a):
    return f(xk+a*pk)

def phipr(df, xk, pk, a):
    return (df(xk + a*pk).T @ pk)

def multiphi(f, xk, pk, a, inds):
    return f(xk+a*pk, inds)

def multiphipr(df, xk, pk, a, inds):
    return (df(xk + a*pk).T @ pk, inds)

def backtracing(f, df, xk, pk, mu1, alf0, rho, multilin=False):
    alf = alf0
    ahist = alf0

    if multilin:
        for i in range(100):
            if multiphi(f, xk, pk, alf) <= multiphi(f, xk, pk, 0) + \
            mu1*alf*multiphipr(df, xk, pk, 0):
                break
            alf = rho*alf
            ahist = np.append(ahist, alf)
            if i == 99:
                print('Backtracking exited without satsifying Armijo condition.')
                return alf
    else:
        for i in range(100):
            if phi(f, xk, pk, alf) <= phi(f, xk, pk, 0) + mu1*alf*phipr(df, xk, pk, 0):
                break
            alf = rho*alf
            ahist = np.append(ahist, alf)
            if i == 99:
                print('Backtracking exited without satsifying Armijo condition.')
                return alf

    return alf

def sgdescent(f, df, x0, etait, epochs=1000, miter=50, tau=1e-4, linesearch=True):
    k = 0
    maxes = xi.shape[0]
    xk = np.copy(x0)
    xhist = np.array([xk])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    ahist = np.array([loss(xk)])
    olosses = np.array([np.linalg.norm(dfloss(xk), ord=2)/np.linalg.norm(xk, ord=2)])

    if linesearch:
        alfk = 1
        mu1 = 1e-5
        rho = .5

    while losses[-1] > tau:
        change = np.copy(xk)

        chosen = np.random.choice(xi.shape[0], maxes, replace=False)
        for m in range(maxes):
            if m % miter == 0:
                if linesearch:
                    if miter > 1:
                        pk = df(xk, chosen[m:m+miter-1])
                        alfk = backtracing(multipsi, dfmultipsi, xk, pk, mu1, \
                        alfk, rho, multilin=True)
                    else:
                        pk = df(xk, chosen[m])
                        alfk = backtracing(f, df, xk, pk, mu1, alfk, \
                        rho, multilin=True)

                else:
                    if k >= 100:
                        alfk = etait(k)
                    else:
                        alfk = 1/100

                minibatch = alfk*df(xk, chosen[m])
            else:
                minibatch += alfk*df(xk, chosen[m])

            if m % miter == miter-1:
                xk -= minibatch/miter

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

def secondeta(it):
    return 1/np.sqrt(it)

def svrgdescent(f, df, x0, etait, tau=1e-4, epochs=1000, miter=100, naive=True, xi=None, yi=None, func=False):
    if xi is None:
        xi, yi = data_table()

    k = 0
    xs = np.copy(x0)
    xhist = np.array([xs])
    reset = False
    change = x0+1
    mu = np.zeros(x0.shape[0])
    losses = np.array([np.linalg.norm(change, ord=2)])
    np.random.seed(2022)
    ahist = np.array([loss(xs)])
    chosen = np.random.choice(xi.shape[0])
    olosses = np.array([np.linalg.norm(df(xs, chosen), ord=2)])
    etas = [1/1000, 1/500, 1/80, 1/30, 1/20]
    miters = [1, 10, 50, 100, 1000]

    ind = miters.index(miter)
    maxes = xi.shape[0]

    for s in range(epochs):
        xk = np.copy(xs)
        change = np.copy(xs)

        mu *= 0

        if func:
            if k < maxes:
                eta = etas[ind]
            else:
                eta = etait(k)
        else:
            eta = etait

        for i in range(xi.shape[1]):
            mu += df(xs, i)

        mu /= (xi.shape[1]+1)
        saved = np.array([xk])
        chosen = np.random.choice(xi.shape[0], maxes, replace=False)

        for m in range(maxes):
            if m % miter == 0:
                minibatch = eta*(df(xk, chosen[m]) - df(xs, chosen[m]) + mu)
            else:
                minibatch += eta*(df(xk, chosen[m]) - df(xs, chosen[m]) + mu)

            if m % miter == miter-1:
                xk -= minibatch/miter
                saved = np.append(saved, [xk], axis=0)

        olosses = np.append(olosses, [np.linalg.norm(mu, ord=2)/np.linalg.norm(xs, ord=2)])
        if naive:
            xs = xk
        else:
            chosen = np.random.choice(int(maxes/miter))
            xs = saved[chosen]

        xhist = np.append(xhist, [xs], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xs, ord=2)])
        ahist = np.append(ahist, [loss(xk)])

        k += 1
        if k == epochs:
            print("Failed to converge")
            break
        if losses[-1] < tau:
            break
#         print(np.linalg.norm(mu, ord=2))

    return xhist, losses, ahist, olosses

def steepestdescent(f, df, x0, tau, alf0, mu1, rho, epochs =1100):
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
        alfk = backtracing(f, df, xk, pk, mu1, alfk, rho)
        if k > 100:
            alfk = 1/k
        else:
            alfk = 1/100
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

def main():
    print("Performing SVRG at multiple different minibatch sizes")
    minibatches = [1, 10, 50, 100, 1000]
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    histsignfound = {}
    histsignlosses = {}
    histsignfuncs = {}
    histsignolosses = {}
    for i in range(len(minibatches)):
        print(f'SVRG with minibatch size = {minibatches[i]}')
        test_start_iter = timeit.default_timer()
        sigmafound, siglosses, sigfuncs, sigolosses = svrgdescent(psi, \
        dfpsi, initialguess, firsteta, tau=1e-4, \
        epochs=200, miter=minibatches[i], \
        naive=True, xi=xi, yi=yi, func=True)
        test_end_iter = timeit.default_timer()
        print(test_end_iter - test_start_iter )
        print(sigolosses.shape)
        print()

        histsignfound[minibatches[i]] = sigmafound
        histsignlosses[minibatches[i]] = siglosses
        histsignfuncs[minibatches[i]] = sigfuncs
        histsignolosses[minibatches[i]] = sigolosses

    histsigfound = {}
    histsiglosses = {}
    histsigfuncs = {}
    histsigolosses = {}
    for i in range(len(minibatches)):
        print(f'SVRG with minibatch size = {minibatches[i]}')
        test_start_iter = timeit.default_timer()
        sigmafound, siglosses, sigfuncs, sigolosses = svrgdescent(psi, \
        dfpsi, initialguess, firsteta, tau=1e-4, \
        epochs=200, miter=minibatches[i], \
        naive=False, xi=xi, yi=yi, func=True)
        test_end_iter = timeit.default_timer()
        print(test_end_iter - test_start_iter )
        print(sigolosses.shape)
        print()

        histsigfound[minibatches[i]] = sigmafound
        histsiglosses[minibatches[i]] = siglosses
        histsigfuncs[minibatches[i]] = sigfuncs
        histsigolosses[minibatches[i]] = sigolosses

    histsigsgdfound = {}
    histsigsgdlosses = {}
    histsigsgdfuncs = {}
    histsigsgdolosses = {}
    for i in range(len(minibatches)):
        print(f'SGD with minibatch size = {minibatches[i]}')
        test_start_iter = timeit.default_timer()
        sigmafound, siglosses, sigfuncs, sigolosses = sgdescent(psi, dfpsi, initialguess, \
        firsteta, epochs=200, miter=minibatches[i], tau=1e-4)
        test_end_iter = timeit.default_timer()
        print(test_end_iter - test_start_iter )
        print(sigolosses[-1])
        print()

        histsigsgdfound[minibatches[i]] = sigmafound
        histsigsgdlosses[minibatches[i]] = siglosses
        histsigsgdfuncs[minibatches[i]] = sigfuncs
        histsigsgdolosses[minibatches[i]] = sigolosses

    print("Performing Steepest Descent")
    test_start_iter = timeit.default_timer()
    gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = steepestdescent(loss, dfloss, initialguess, 1e-4, 1, 1e-5, .5, epochs=200)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()

    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (18, 9),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(3, 3)

        for j in range(3):
            if j == 0:
                plt.setp(axs[0, j], title = 'SGD')
                plt.setp(axs[j, 0], ylabel='L(w) Values')
            elif j == 1:
                plt.setp(axs[0, j], title = 'Naive SVRG')
                plt.setp(axs[j, 0], ylabel=r'||$\nabla$L(w)|| Values')
            else:
                plt.setp(axs[0, j], title = 'Random SVRG')
                plt.setp(axs[j, 0], ylabel='||Δw|| Values')
            plt.setp(axs[:, j], xlabel=r'Major Epochs (# of $\nabla$Ψ(w)/m)')

            axs[0,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, '-.', markeredgecolor="none", \
            label=f'Gradient Descent')
            axs[0,j].set_yscale('log')

            axs[1,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsigolosses, '-.', markeredgecolor="none", \
            label=f'Gradient Descent')
            axs[1,j].set_yscale('log')

            axs[2,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsiglosses, '-.', markeredgecolor="none", \
            label=f'Gradient Descent')
            axs[2,j].set_yscale('log')


        for mini in range(len(minibatches)):
            # gradn = int(1000/minibatches[mini])
            sizes = histsigsgdfuncs[minibatches[mini]].shape[0]
            axs[0, 0].plot(range(0, sizes, 1), \
            histsigsgdfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
            label=f'minibatch size = {minibatches[mini]}')

            axs[1, 0].plot(range(0, sizes, 1), \
            histsigsgdolosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

            axs[2, 0].plot(range(0, sizes, 1), \
            histsigsgdlosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

            # gradn = int(np.ceil(1000/(minibatches[mini]+1000)))
            sizes = histsignfuncs[minibatches[mini]].shape[0]
            axs[0, 1].plot(range(0, 2*sizes, 2), \
            histsignfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
            label=f'minibatch size = {minibatches[mini]}')

            axs[1, 1].plot(range(0, 2*sizes, 2), \
            histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

            axs[2, 1].plot(range(0, 2*sizes, 2), \
            histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

            sizes = histsigfuncs[minibatches[mini]].shape[0]
            axs[0, 2].plot(range(0, 2*sizes, 2), \
            histsigfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
            label=f'minibatch size = {minibatches[mini]}')

            axs[1, 2].plot(range(0, 2*sizes, 2), \
            histsigolosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

            axs[2, 2].plot(range(0, 2*sizes, 2), \
            histsiglosses[minibatches[mini]], '-.', markeredgecolor="none", \
            label=f'minibatch size = {minibatches[mini]}')

        fig.set_size_inches(18, 18)

        plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("toy_net.png", bbox_inches='tight')

    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="L(w) Values", title="L(w) vs. Major Epochs", \
                          y_axis_type="log")
        colors = itertools.cycle(palette(('#000003', '#410967', '#932567', '#DC5039', '#FBA40A', '#79D151'), len(minibatches) + 1))
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigfuncs[mini].shape[0]
            p.line(np.linspace(start=0, stop=(50), num=sizes), \
            histsigfuncs[mini], color=next(colors), \
            legend_label=f'SVRG with minibatch size = {mini}')
        p.line(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, color = next(colors), \
        legend_label=f'Gradient Descent')

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        p.legend.location = "top_right"
        p.legend.title = "Optimization Methods"
        p.legend.title_text_font_size = '10px'
        p.legend.label_text_font_size = '10px'
        p.legend.background_fill_alpha = 0.2
        export_png(style(p), filename="toyn_func.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="||∇L(w)|| Values", title="||∇L(w)|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigolosses[mini].shape[0]
            p.line(np.linspace(start=0, stop=(50), num=sizes), \
            histsigolosses[mini], line_color=next(colors), \
            legend_label=f'SVRG with minibatch size = {mini}')
        p.line(range(0, gdsigolosses.shape[0], 1), gdsigolosses, line_color = next(colors), \
        legend_label=f'Gradient Descent')

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        p.legend.location = "top_right"
        p.legend.title = "Optimization Methods"
        p.legend.title_text_font_size = '10px'
        p.legend.label_text_font_size = '10px'
        p.legend.background_fill_alpha = 0.2
        export_png(style(p), filename="toyn_oloss.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="||Δw|| Values", title="||Δw|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsiglosses[mini].shape[0]
            p.line(np.linspace(start=0, stop=(50), num=sizes), \
            histsiglosses[mini], line_color=next(colors), \
            legend_label=f'SVRG with minibatch size = {mini}')
        p.line(range(0, gdsiglosses.shape[0], 1), gdsiglosses, line_color = next(colors), \
        legend_label=f'Gradient Descent')

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        p.legend.location = "top_right"
        p.legend.title = "Optimization Methods"
        p.legend.title_text_font_size = '10px'
        p.legend.label_text_font_size = '10px'
        p.legend.background_fill_alpha = 0.2
        export_png(style(p), filename="toyn_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigfuncs[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(50), num=sizes), \
            histsigfuncs[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title("L(w) vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"L(w) Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("toy_func.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsiglosses[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(50), num=sizes), \
            histsiglosses[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsiglosses.shape[0], 1), gdsiglosses, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title("||Δw|| vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"||Δw|| Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("toy_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigolosses[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(50), num=sizes), \
            histsigolosses[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsigolosses.shape[0], 1), gdsigolosses, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title(r"||$\nabla$L(w)|| vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"||$\nabla$L(w)|| Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("toy_oloss.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
