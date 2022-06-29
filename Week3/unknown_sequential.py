import numpy as np
import timeit
import matplotlib.pyplot as plt
import bokeh.plotting
from bokeh.io import output_notebook, export_png
import sys
from Week2.steepestdescent import *
from Week2.sgd import psi, dfpsi, firsteta
from Week2.makedata import data_table
import ray
from bokeh.palettes import linear_palette as palette
import itertools

ray.init()

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

def sgdescent(f, df, x0, etait, xi=None, yi=None, epochs=1000, miter=50, tau=1e-6):
    if xi is None:
        xi, yi = data_table()

    k = 0
    xk = x0
    xhist = np.array([xk])
    change = x0+1
    losses = np.array([np.linalg.norm(change-1, ord=2)])
    np.random.seed(2022)
    ahist = np.array([loss(xk, xi, yi)])
    num = dfloss(xk, xi, yi)
    olosses = np.array([np.linalg.norm(num, ord=2)/np.linalg.norm(xk, ord=2)])

    while losses[-1] > tau:
        change = np.copy(xk)

        if k >= 3:
            alfk = etait(k)
        else:
            alfk = 1/(k+3)

        chosen = np.random.choice(xi.shape[0], miter, replace=False)
        for m in range(miter):
            deriv = df(xk, chosen[m], xi, yi)
            xk = xk - alfk*deriv/miter

        xhist = np.append(xhist, [xk], axis=0)
        losses = np.append(losses, [np.linalg.norm(change - xk, ord=2) / np.linalg.norm(xk, ord=2)])
        num = dfloss(xk, xi, yi)
        olosses = np.append(olosses, [np.linalg.norm(num, ord=2)/np.linalg.norm(xk, ord=2)])
        ahist = np.append(ahist, [loss(xk, xi, yi)])

        change -= xk

        k += 1
        if k == epochs:
            print("Failed to converge")
            break

    return xhist, losses, ahist, olosses

def main():
    xi, yi = data_table()
    print("Performing SGD at multiple different minibatch sizes")
    minibatches = [1, 10, 50, 100, 200]
    initialguess = 2.25*np.ones(shape=(xi[1].shape[0]+1,))
    histsigfound = {}
    histsiglosses = {}
    histsigfuncs = {}
    histsigolosses = {}

    for i in range(len(minibatches)):
        print(f'SGD with minibatch size = {minibatches[i]}')
        test_start_iter = timeit.default_timer()
        sigmafound, siglosses, sigfuncs, sigolosses = sgdescent(psi, dfpsi, initialguess, \
        firsteta, xi, yi, epochs=2000*200/minibatches[i], miter=minibatches[i], tau=1e-6)
        test_end_iter = timeit.default_timer()
        print(test_end_iter - test_start_iter )
        print()
        
        histsigfound[minibatches[i]] = sigmafound
        histsiglosses[minibatches[i]] = siglosses
        histsigfuncs[minibatches[i]] = sigfuncs
        histsigolosses[minibatches[i]] = sigolosses

    print("Performing Steepest Descent")
    test_start_iter = timeit.default_timer()
    gdsigmafound, gdsiglosses, gdsigfuncs, gdsigolosses = steepestdescent(loss, dfloss, initialguess, 1e-6, 1, 1e-5, .5, xi, yi, epochs=2000)
    test_end_iter = timeit.default_timer()
    print(test_end_iter - test_start_iter )
    print()


    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="L(w) Values", title="L(w) vs. Major Epochs", \
                          y_axis_type="log")
        colors = itertools.cycle(palette(('#000003', '#410967', '#932567', '#DC5039', '#FBA40A', '#79D151'), len(minibatches) + 1))
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsigfuncs[mini].shape[0]
            p.line(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsigfuncs[mini], color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
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
        export_png(style(p), filename="unknown_func.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="||∇L(w)|| Values", title="||∇L(w)|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsigolosses[mini].shape[0]
            p.line(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsigolosses[mini], line_color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
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
        export_png(style(p), filename="unknown_oloss.png")

        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Major Epochs (# of ∇Ψ(w)/m)", \
                          y_axis_label="||Δw|| Values", title="||Δw|| vs. Major Epochs", \
                          y_axis_type="log")
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsiglosses[mini].shape[0]
            p.line(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsiglosses[mini], line_color=next(colors), \
            legend_label=f'SGD with minibatch size = {mini}')
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
        export_png(style(p), filename="unknown_loss.png")
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        print(e)
        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsigfuncs[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsigfuncs[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title("L(w) vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"L(w) Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("unknown_func.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsiglosses[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsiglosses[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsiglosses.shape[0], 1), gdsiglosses, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title("||Δw|| vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"||Δw|| Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("unknown_loss.png", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.gca()
        for mini in minibatches:
            gradn = int(200/mini)
            sizes = histsigolosses[mini].shape[0]
            ax.plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
            histsigolosses[mini], '-.', markeredgecolor="none",  \
            label=f'SGD with minibatch size = {mini}')
        ax.plot(range(0, gdsigolosses.shape[0], 1), gdsigolosses, '-.', markeredgecolor="none", \
        label=f'Gradient Descent')
        ax.set_yscale('log')
        plt.title(r"||$\nabla$L(w)|| vs. Major Epochs")
        plt.xlabel(r"Major Epochs (# of $\nabla$Ψ(w)/m)")
        plt.ylabel(r"||$\nabla$L(w)|| Values")
        plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1, title="Optimization Methods")
        plt.savefig("unknown_oloss.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
