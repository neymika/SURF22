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

def sgdescent(f, df, x0, etait, epochs=1000, miter=50):
    k = 0
    xk = x0
    xhist = np.array([xk])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    ahist = np.array([loss(xk)])
    olosses = np.array([np.linalg.norm(dfloss(xk), ord=2)/np.linalg.norm(xk, ord=2)])


    while losses[-1] > .01:
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

def main():
    test_start_iter = timeit.default_timer()
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    sigmafound, siglosses, sigfuncs, sigolosses = sgdescent(psi, dfpsi, initialguess, firsteta, epochs=1000, miter=20)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Stochastic Gradient Descent Values")
    print(sigmafound[-1])
    print(sigfuncs[-1])
    print(siglosses[-1])

    xib = np.concatenate((xi, np.ones((xi.shape[0], 1))), 1)
    d = (xib.T @ xib)+ 0.0001*np.identity(xib.shape[1])
    theta_star = np.linalg.lstsq(d, xib.T @ yi, rcond=None)
    true_objective = loss(theta_star[0])

    print("A\\b Values")
    print(theta_star[0])
    print(true_objective)

    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs", \
                          y_axis_label="L(w) Values", title="L(w) vs. Major Epochs for SGD", \
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

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs", \
                          y_axis_label="||∇L(w)|| Values", title="||∇L(w)|| vs. Major Epochs for SGD", \
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

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs", \
                          y_axis_label="||Δw|| Values", title="||Δw|| vs. Major Epochs for SGD", \
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
        plt.title("||∇L(w)|| vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("||∇L(w)|| Values")
        plt.savefig("toysgd_oloss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||Δw|| vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("||Δw|| Values")
        plt.savefig("toysgd_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigfuncs.shape[0], 1), sigfuncs, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("L(w) vs. Major Epochs for SGD")
        plt.xlabel("Major Epochs")
        plt.ylabel("L(w) Values")
        plt.savefig("toysgd_func.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
