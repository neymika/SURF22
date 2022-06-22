import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.io import output_notebook, export_png
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

def sgdescent(f, df, x0, etait, epochs=1000, miter=50):
    k = 0
    xog = x0
    xk = x0
    xhist = np.array([xk])
    reset = False
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    chosen = np.random.choice(xi.shape[0])
    ahist = np.array([jacob(xk)])
    olosses = np.array([np.linalg.norm(dfjacob(xk), ord=2)/np.linalg.norm(xk, ord=2)])


    while np.linalg.norm(dfjacob(xk), ord=2)/np.linalg.norm(xk, ord=2) > .01:
        change = np.copy(xk)
        xog = np.copy(xk)

        if k >= 3:
            alfk = etait(k)
        else:
            alfk = 1/(k+3)

        for m in range(miter):
            chosen = np.random.choice(xi.shape[0])
            xk -= alfk*df(xk, chosen)/miter

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

def main():
    test_start_iter = timeit.default_timer()
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    sigmafound, siglosses, sigfuncs, sigolosses = sgdescent(fi, dffi, initialguess, firsteta, epochs=1000, miter=20)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Stochastic Gradient Descent Values")
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

    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Number of iterations", \
                          y_axis_label="J(θ) Values", title="J(θ) vs. Number Iterations for SGD", \
                          y_axis_type="log")
        p.line(range(0, sigfuncs.shape[0], 1), sigfuncs, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="toysgd_func.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Number of iterations", \
                          y_axis_label="||∇J(θ)|| Values", title="||∇J(θ)|| vs. Number Iterations for SGD", \
                          y_axis_type="log")
        p.line(range(0, sigolosses.shape[0], 1), sigolosses, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="toysgd_olosses.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Number of iterations", \
                          y_axis_label="||Δθ|| Values", title="||Δθ|| vs. Number Iterations for SGD", \
                          y_axis_type="log")
        p.line(range(0, siglosses.shape[0], 1), siglosses, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="toysgd_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||∇J(θ)|| vs. Number Iterations for SGD")
        plt.xlabel("Number of iterations")
        plt.ylabel("||∇J(θ)|| Values")
        plt.savefig("toysgd_oloss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||Δθ|| vs. Number Iterations for SGD")
        plt.xlabel("Number of iterations")
        plt.ylabel("||Δθ|| Values")
        plt.savefig("toysgd_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigfuncs.shape[0], 1), sigfuncs, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("J(θ) vs. Number Iterations for SGD")
        plt.xlabel("Number of iterations")
        plt.ylabel("J(θ) Values")
        plt.savefig("toysgd_func.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
