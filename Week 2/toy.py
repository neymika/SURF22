import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.io import output_notebook, export_png
from bokeh.palettes import linear_palette as palette
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

def jacob(sig, lam=1e-4):
    sumi = (lam/2)*(sig.T @ sig)
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    netsum = np.mean((xi_til @ sig - yi)**2)/2
    sumi += netsum

    return sumi

def dfjacob(sig, lam=1e-4):
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    derivs  = (xi_til @ sig - yi) @ xi_til

    derivs += lam * sig.T

    return derivs/xi.shape[0]

def fi(sig, i, lam=1e-4):
    sumi = lam*(sig.T @ sig) + ((sig.T @ np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dffi(sig, j, lam=1e-4):
    largei = np.append(xi[j], 1)
    derivs = (sig.T @ largei - yi[j])*largei
    derivs += lam * sig.T

    return derivs

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

def sgdescent(f, df, x0, etait, epochs=1000, miter=50, tau=1e-4):
    k = 0
    xk = x0
    xhist = np.array([xk])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    ahist = np.array([jacob(xk)])
    olosses = np.array([np.linalg.norm(dfjacob(xk), ord=2)/np.linalg.norm(xk, ord=2)])


    while losses[-1] > tau:
        change = np.copy(xk)

        if k >= 3:
            alfk = etait(k)
        else:
            alfk = 1/(k+3)

        chosen = np.random.choice(xi.shape[0], miter, replace=False)
        for m in range(miter):
            xk -= alfk*df(xk, chosen[m])/miter

        xhist = np.append(xhist, [xk], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])
        olosses = np.append(olosses, [np.linalg.norm(dfjacob(xk), ord=2)/np.linalg.norm(xk, ord=2)])
        ahist = np.append(ahist, [jacob(xk)])

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

def steepestdescent(f, df, x0, tau, alf0, mu1, rho):
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


    while losses[-1] > tau:
        change = np.copy(xk)

        if k != 0:
            dk = df(xk)
            pk = -dk
        alfk = backtracing(f, df, xk, pk, mu1, alfk, rho)
        xk = xk + alfk*pk

        xhist = np.append(xhist, [xk], axis=0)
        ahist = np.append(ahist, [f(xk)])
        olosses = np.append(olosses, [np.linalg.norm(df(xk), ord=2)/np.linalg.norm(xk, ord=2)])
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])

        k += 1
        if k == 1100:
            print("Failed to converge")
            break

    return xhist, losses, ahist, olosses

def main():
    print("Performing SGD at multiple different minibatch sizes")
    minibatches = [1, 10, 50, 100, 1000]
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    histsigfound = {}
    histsiglosses = {}
    histsigfuncs = {}
    histsigolosses = {}
    for i in range(len(minibatches)):
        print(f'SGD with minibatch size = {minibatches[i]}')
        test_start_iter = timeit.default_timer()
        sigmafound, siglosses, sigfuncs, sigolosses = sgdescent(fi, dffi, initialguess, firsteta, epochs=5000, miter=minibatches[i])
        test_end_iter = timeit.default_timer()
        print(test_end_iter - test_start_iter )
        print()

        histsigfound[minibatches[i]] = sigmafound
        histsiglosses[minibatches[i]] = siglosses
        histsigfuncs[minibatches[i]] = sigfuncs
        histsigolosses[minibatches[i]] = sigolosses

    print("Performing Steepest Descent")
    test_start_iter = timeit.default_timer()
    gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = steepestdescent(jacob, dfjacob, initialguess, .0001, 1, 1e-5, .5)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()


    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(θ)/m)", \
                          y_axis_label="J(θ) Values", title="J(θ) vs. Major Epochs", \
                          y_axis_type="log")
        colors = itertools.cycle(palette(('#000003', '#410967', '#932567', '#DC5039', '#FBA40A', '#79D151'), len(minibatches) + 1))
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigfuncs[mini].shape[0]
            p.line(np.linspace(start=1, stop=1+(sizes/gradn), num=sizes), \
            histsigfuncs[mini], color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
        p.line(range(1, 1+gdsigfuncs.shape[0], 1), gdsigfuncs, color = next(colors), \
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
        export_png(style(p), filename="toy_func.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(θ)/m)", \
                          y_axis_label="||∇J(θ)|| Values", title="||∇J(θ)|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsigolosses[mini].shape[0]
            p.line(np.linspace(start=1, stop=1+(sizes/gradn), num=sizes), \
            histsigolosses[mini], line_color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
        p.line(range(1, 1+gdsigolosses.shape[0], 1), gdsigolosses, line_color = next(colors), \
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
        export_png(style(p), filename="toy_olosses.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(θ)/m)", \
                          y_axis_label="||Δθ|| Values", title="||Δθ|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(1000/mini)
            sizes = histsiglosses[mini].shape[0]
            p.line(np.linspace(start=1, stop=1+(sizes/gradn), num=sizes), \
            histsiglosses[mini], line_color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
        p.line(range(1, 1+gdsiglosses.shape[0], 1), gdsiglosses, line_color = next(colors), \
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
        export_png(style(p), filename="toy_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||∇J(θ)|| vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("||∇J(θ)|| Values")
        plt.savefig("toysgd_oloss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||Δθ|| vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("||Δθ|| Values")
        plt.savefig("toysgd_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigfuncs.shape[0], 1), sigfuncs, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("J(θ) vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("J(θ) Values")
        plt.savefig("toysgd_func.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
