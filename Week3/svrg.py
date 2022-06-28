import numpy as np
import timeit
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.io import output_notebook, export_png
import sys
from Week2.makedata import data_table
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

def jacob(sig, xi, yi, lam=1e-4):
    sumi = (lam/2)*(sig.T @ sig)
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    netsum = np.mean((xi_til @ sig - yi)**2)/2
    sumi += netsum

    return sumi

def dfjacob(sig, xi, yi, lam=1e-4):
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    derivs  = (xi_til @ sig - yi) @ xi_til

    derivs += lam * sig.T

    return derivs/xi.shape[0]

def fi(sig, i, xi, yi, lam=1e-4):
    sumi = lam*(sig.T @ sig) + ((sig.T @ np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dffi(sig, j, xi, yi, lam=1e-4):
    largei = np.append(xi[j], 1)
    derivs = (sig.T @ largei - yi[j])*largei
    derivs += lam * sig.T

    return derivs


def svrgdescent(f, df, x0, etait, epochs=1000, miter=100, naive=True, xi=None, yi=None):
    if xi is None:
        xi, yi = data_table()

    k = 0
    xog = x0
    xs = x0
    xhist = np.array([xs])
    reset = False
    change = x0
    mu = np.zeros(x0.shape[0])
    losses = np.array([np.linalg.norm(change, ord=2)])
    np.random.seed(2022)
    chosen = np.random.choice(xi.shape[0])
    olosses = np.array([np.linalg.norm(df(xs, chosen, xi, yi), ord=2)])


    for s in range(epochs):
        xk = np.copy(xs)
        xog = np.copy(xs)

        for i in range(49):
            mu += df(xog, i, xi, yi)

        mu /= 49
        saved = np.array([xk])

        for m in range(miter):
            chosen = np.random.choice(xi.shape[0])
            xk -= etait*(df(xk, chosen, xi, yi) - df(xog, chosen, xi, yi) + mu)/miter
            saved = np.append(saved, [xk], axis=0)

        if naive:
            xs = xk
        else:
            chosen = np.random.choice(miter)
            xs = saved[chosen]

        xhist = np.append(xhist, [xs], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xs, ord=2)])
        olosses = np.append(olosses, [np.linalg.norm(mu, ord=2)])

        k += 1
        if k == epochs:
            print("Failed to converge")
            break
        if np.linalg.norm(mu, ord=2) <= .01:
            break
#         print(np.linalg.norm(mu, ord=2))

    return xhist, losses, olosses

def firsteta(it):
    return 1/it

def secondeta(it):
    return 1/np.sqrt(it)

def main():
    xi, yi = data_table()
    test_start_iter = timeit.default_timer()
    xsvrgd, lsvrgd, olsvrgd = svrgdescent(fi, dffi, np.ones(shape=(xi[1].shape[0]+1,)), .5, \
    epochs=1000, miter=20, naive=True, xi=xi, yi=yi)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Stochastic Gradient Descent Values")
    print(xsvrgd[-1])
    print(lsvrgd[-1])
    print(olsvrgd[-1])

    xib = np.concatenate((xi, np.ones((xi.shape[0], 1))), 1)
    d = (xib.T @ xib)+ 0.0001*np.identity(xib.shape[1])
    theta_star = np.linalg.lstsq(d, xib.T @ yi, rcond=None)
    true_objective = jacob(theta_star[0], xi, yi)

    print("A\\b Values")
    print(theta_star[0])
    print(true_objective)

if __name__ == "__main__":
    main()
