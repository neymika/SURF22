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
    test_start_iter = timeit.default_timer()
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    sigmafound, siglosses, sigfuncs, sigolosses = steepestdescent(loss, dfloss, initialguess, 1e-7, 1, 1e-5, .5, epochs=10000)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )

    print("Steepest Descent Values")
    print(sigmafound[-1])
    print(sigfuncs[-1])
    print(siglosses[-1])
    print(sigolosses[-1])

    xib = np.concatenate((xi, np.ones((xi.shape[0], 1))), 1)
    d = (xib.T @ xib)+ 0.0001*np.identity(xib.shape[1])
    theta_star = np.linalg.lstsq(d, xib.T @ yi, rcond=None)
    true_objective = np.linalg.norm(dfloss(theta_star[0]), ord=2)

    print("A\\b Values")
    print(theta_star[0])
    print(true_objective)

    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Number of iterations", \
                          y_axis_label="||∇L(w)|| Values", title="||∇L(w)|| vs. Number Iterations for GD", \
                          y_axis_type="log")
        p.line(range(0, sigolosses.shape[0], 1), sigolosses, line_color="navy")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.title.align = "center"
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename="toygradient_descent_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(range(0, sigolosses.shape[0], 1), sigolosses, '-o', markeredgecolor="none")
        ax.set_yscale('log')
        plt.title("||∇L(w)|| vs. Number Iterations for GD")
        plt.xlabel("Number of iterations")
        plt.ylabel("||∇L(w)|| Values")
        plt.savefig("toygradient_descent_loss.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
