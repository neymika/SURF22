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

def loss(sig, xi, yi, lam=1e-4):
    sumi = (lam/2)*(sig.T @ sig)
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    netsum = np.mean((xi_til @ sig - yi)**2)/2
    sumi += netsum

    return sumi

def dfloss(sig, xi, yi, lam=1e-4):
    xi_til = np.hstack((xi, np.ones((xi.shape[0],1))))
    derivs  = (xi_til @ sig - yi) @ xi_til

    derivs += lam * sig.T

    return derivs/xi.shape[0]

def psi(sig, i, xi, yi, lam=1e-4):
    sumi = lam*(sig.T @ sig) + ((sig.T @ np.append(xi[i], 1)) - yi[i])**2

    return sumi/2

def dfpsi(sig, j, xi, yi, lam=1e-4):
    largei = np.append(xi[j], 1)
    derivs = (sig.T @ largei - yi[j])*largei
    derivs += lam * sig.T

    return derivs

@ray.remote
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
    ahist = np.array([loss(xs, xi, yi)])
    chosen = np.random.choice(xi.shape[0])
    olosses = np.array([np.linalg.norm(df(xs, chosen, xi, yi), ord=2)])


    for s in range(epochs):
        xk = np.copy(xs)
        change = np.copy(xs)

        mu *= 0

        if func:
            if k < 3:
                eta = 1/(k+3)
            else:
                eta = etait(k)
        else:
            eta = etait

        for i in range(xi.shape[1]):
            mu += df(xs, i, xi, yi)

        mu /= (xi.shape[1]+1)
        saved = np.array([xk])
        chosen = np.random.choice(xi.shape[0], miter, replace=False)

        for m in range(miter):
            xk -= eta*(df(xk, chosen[m], xi, yi) - df(xs, chosen[m], xi, yi) + mu)/miter
            saved = np.append(saved, [xk], axis=0)

        olosses = np.append(olosses, [np.linalg.norm(mu, ord=2)/np.linalg.norm(xs, ord=2)])
        if naive:
            xs = xk
        else:
            chosen = np.random.choice(miter)
            xs = saved[chosen]

        xhist = np.append(xhist, [xs], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xs, ord=2)])
        ahist = np.append(ahist, [loss(xk, xi, yi)])

        k += 1
        if k == epochs:
            print("Failed to converge")
            break
        if losses[-1] < tau:
            break
#         print(np.linalg.norm(mu, ord=2))

    return xhist, losses, ahist, olosses

def firsteta(it):
    return 1/it

def secondeta(it):
    return 1/np.sqrt(it)

def main():
    xi, yi = data_table()
    initialguess = np.ones(shape=(xi[1].shape[0]+1,))
    test_start_iter = timeit.default_timer()
    sigmafound, siglosses, sigfuncs, sigolosses = ray.get(svrgdescent.remote(psi, \
    dfpsi, initialguess, firsteta, tau=1e-4, epochs=10000, miter=200, \
    naive=False, xi=xi, yi=yi, func=True))
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Stochastic Gradient Descent Values")
    print(sigmafound[-1])
    print(siglosses[-1])
    print(sigolosses[-1])
    print(sigolosses.shape)

    xib = np.concatenate((xi, np.ones((xi.shape[0], 1))), 1)
    d = (xib.T @ xib)+ 0.0001*np.identity(xib.shape[1])
    theta_star = np.linalg.lstsq(d, xib.T @ yi, rcond=None)
    true_objective = loss(theta_star[0], xi, yi)

    print("A\\b Values")
    print(theta_star[0])
    print(true_objective)

    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Number of iterations", \
                          y_axis_label="L(w) Values", title="L(w) vs. Number Iterations for SVRG", \
                          y_axis_type="log")
        p.line(range(0, sigfuncs.shape[0], 1), sigfuncs, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="svrg_func.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs", \
                          y_axis_label="||∇L(w)|| Values", title="||∇L(w)|| vs. Major Epochs for SVRG", \
                          y_axis_type="log")
        p.line(range(0, sigolosses.shape[0], 1), sigolosses, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="svrg_olosses.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs", \
                          y_axis_label="||Δw|| Values", title="||Δw|| vs. Major Epochs for SVRG", \
                          y_axis_type="log")
        p.line(range(0, siglosses.shape[0], 1), siglosses, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="svrg_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||∇L(w)|| vs. Major Epochs for SVRG")
        plt.xlabel("Major Epochs")
        plt.ylabel("||∇L(w)|| Values")
        plt.savefig("svrg_oloss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||Δw|| vs. Major Epochs for SVRG")
        plt.xlabel("Major Epochs")
        plt.ylabel("||Δw|| Values")
        plt.savefig("svrg_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigfuncs.shape[0], 1), sigfuncs, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("L(w) vs. Major Epochs for SVRG")
        plt.xlabel("Major Epochs")
        plt.ylabel("L(w) Values")
        plt.savefig("svrg_func.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
