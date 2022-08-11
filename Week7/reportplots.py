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
plt.style.use('tableau-colorblind10')
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

def main():
    # eta = 1 w/ decay schedule (3 subplots)
    # eta = .01 vs. eta = 1 w/ decay schedule (3 subplots)
    # best schedule for GD, SGD, and SVRG w/ same decay schedule 1/(s || t)
        # include a convergence line at tau = 10e-4
    print("Performing SVRG at multiple different stepsizes")
    stepsizes = [1, .1, .01, .001, .0001, .00001, 1/(4*1800001)]
    plotsize = [0, 1, 4]
    decayschedule = [1, 5, 10, 20, 50, 100]

    histsiglosses = pd.read_csv("../Week6/svrglosses.csv", header=0)
    histsiglosses = histsiglosses.apply(lambda x: pd.Series(x.dropna().values))
    histsigfuncs = pd.read_csv("../Week6/svrgfunc.csv", header=0)
    histsigfuncs = histsigfuncs.apply(lambda x: pd.Series(x.dropna().values))
    histsigolosses = pd.read_csv("../Week6/svrgolosses.csv", header=0)
    histsigolosses = histsigolosses.apply(lambda x: pd.Series(x.dropna().values))

    print(histsiglosses.keys())

    params = {'legend.fontsize': 8,
         'figure.titlesize':10,
         'figure.figsize': (12, 4),
         'axes.labelsize': 9,
         'axes.titlesize':10,
         'xtick.labelsize':8,
         'ytick.labelsize':8,
         'lines.linewidth': 0.7,
         'lines.markersize': 2.5,
         "figure.facecolor": "#FFFFFF",
         "axes.facecolor": "#FFFFFF",
         "savefig.facecolor":"#FFFFFF",
         "axes.labelcolor":"#000000",
         }
    lbl = "#000000"
    tk = "#808080"
    colors = ["#3A637B", "#C4A46B", "#FF6917", "#D44141" ]
    with mpl.rc_context(params):
        fig, axs = plt.subplots(1, 3, sharey='row')
        # fig.suptitle(fr'||$\nabla$L(w)|| vs Epochs for SVRG')

        for i in range(3):
            dictkey = str(20 + plotsize[i])
            lastindex = histsigfuncs[dictkey].notna()[::-1].idxmax()
            xrange = range(0, lastindex+1)

            axs[i].plot(xrange, histsigolosses[dictkey][0:lastindex+1], '-', color=colors[0])
            axs[i].plot(xrange, 1e-4*np.ones(len(xrange)), color=colors[3])
            axs[i].set_yscale('log')
            axs[i].set_title(\
            rf'$\eta$: {stepsizes[0]}, $\eta = 1/t$ Starting Epoch: {decayschedule[plotsize[i]]}',\
            pad=15,fontsize=10,color=lbl)

            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["left"].set_color(tk)
            axs[i].spines["left"].set_linewidth(0.3)
            axs[i].spines["bottom"].set_color(tk)
            axs[i].spines["bottom"].set_linewidth(0.3)
            axs[i].tick_params(colors=tk,width=0.3)
            axs[i].tick_params(which="minor",bottom=False) # remove minor tick labels

        plt.setp(axs[:], xlabel='Epochs')
        plt.setp(axs[:], ylabel=r'||$\nabla$L(w)|| Values')
        plt.savefig("svrg_best_oloss.png", bbox_inches='tight')

        fig, axs = plt.subplots(1, 3, sharey='row')
        # fig.suptitle(fr'L(w)vs Epochs for SVRG')

        for i in range(3):
            dictkey = str(20 + plotsize[i])
            odictkey = str(30+plotsize[i])
            lastindex = histsigfuncs[dictkey].notna()[::-1].idxmax()
            olastindex = histsigfuncs[odictkey].notna()[::-1].idxmax()
            xrange = range(0, olastindex+1)
            print(xrange)

            axs[i].plot(range(0, lastindex+1), \
            histsigolosses[dictkey][0:lastindex+1], '-', alpha=.65, \
            label=rf'$\eta$: {stepsizes[1]}', color=colors[0])
            axs[i].plot(xrange, histsigolosses[odictkey][0:olastindex+1], '-.', \
            alpha=.8, label=rf'$\eta$: {stepsizes[2]}', color=colors[1])
            axs[i].plot(xrange, 1e-4*np.ones(len(xrange)), \
            '--', markeredgecolor="none", alpha=.6, \
            label=r'Convergence $\tau = 10^{-4}$', color=colors[3])
            axs[i].set_yscale('log')
            axs[i].set_title(\
            rf'$\eta = 1/t$ Starting Epoch: {decayschedule[plotsize[i]]}', \
            pad=15,fontsize=10,color=lbl)

            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["left"].set_color(tk)
            axs[i].spines["left"].set_linewidth(0.3)
            axs[i].spines["bottom"].set_color(tk)
            axs[i].spines["bottom"].set_linewidth(0.3)
            axs[i].tick_params(colors=tk,width=0.3)
            axs[i].tick_params(which="minor",bottom=False) # remove minor tick labels


        plt.setp(axs[:], xlabel='Epochs')
        plt.setp(axs[:], ylabel=r'||$\nabla$L(w)|| Values')
        plt.legend(loc = "upper center",bbox_to_anchor=(-0.6,1.05), ncol=3, title="Optimization Methods")
        plt.savefig("svrg_best_comp.png", bbox_inches='tight')

    params = {'legend.fontsize': 8,
         'figure.titlesize':10,
         'figure.figsize': (6, 6),
         'axes.labelsize': 9,
         'axes.titlesize':10,
         'xtick.labelsize':8,
         'ytick.labelsize':8,
         'lines.linewidth': 0.7,
         'lines.markersize': 2.5,
         "figure.facecolor": "#FFFFFF",
         "axes.facecolor": "#FFFFFF",
         "savefig.facecolor":"#FFFFFF",
         "axes.labelcolor":"#000000",
         }
    with mpl.rc_context(params):
        fig, axs = plt.subplots(1, 1, sharey='row')
        # fig.suptitle(fr'L(w)vs Epochs for SVRG')

        svrgdictkey = str(21)
        sgddictkey = str(32)
        gddictkey = str(11)
        lastindex = histsigfuncs[svrgdictkey].notna()[::-1].idxmax()
        xrange = range(0, 2*(lastindex+1), 2)
        print(xrange)

        axs.plot(xrange, \
        histsigolosses[svrgdictkey][0:lastindex+1], '-', alpha=.65, \
        label=rf'SVRG $\eta$: {stepsizes[1]}, Decay Epoch: 5', color=colors[0])
        histsigolosses = pd.read_csv("../Week6/gdolosses.csv", header=0)
        histsigolosses = histsigolosses.apply(lambda x: pd.Series(x.dropna().values))
        lastindex = histsigolosses[gddictkey].notna()[::-1].idxmax()
        xrange = range(0, lastindex+1)
        axs.plot(xrange, histsigolosses[gddictkey][0:lastindex+1], '-.', \
        alpha=.8, label=rf'GD $\eta$: {stepsizes[0]}', color=colors[1])
        histsigolosses = pd.read_csv("../Week6/sgdolosses.csv", header=0)
        histsigolosses = histsigolosses.apply(lambda x: pd.Series(x.dropna().values))
        lastindex = histsigolosses[sgddictkey].notna()[::-1].idxmax()
        xrange = range(0, lastindex+1)
        axs.plot(xrange, histsigolosses[sgddictkey][0:lastindex+1], '--', \
        alpha=.8, label=rf'SGD $\eta$: {stepsizes[2]}, Decay Epoch: 10', color=colors[2])
        axs.plot(xrange, 1e-4*np.ones(len(xrange)), \
        '-.', markeredgecolor="none", alpha=.6, label=r'Convergence $\tau = 10^{-4}$', color=colors[3])
        axs.set_yscale('log')
        axs.set_title(\
        rf'Comparison of Optimization Methods with Their Best Tradeoffs', \
        pad=15,fontsize=10,color=lbl)


        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.spines["left"].set_color(tk)
        axs.spines["left"].set_linewidth(0.3)
        axs.spines["bottom"].set_color(tk)
        axs.spines["bottom"].set_linewidth(0.3)
        axs.tick_params(colors=tk,width=0.3)
        axs.tick_params(which="minor",bottom=False) # remove minor tick labels

        plt.setp(axs, xlabel=r'Major Epochs (# of $\nabla$Ψ(w)/m)')
        plt.setp(axs, ylabel=r'||$\nabla$L(w)|| Values')
        plt.legend(loc = "lower center",bbox_to_anchor=(0.5,0.9), ncol=4, title="Optimization Methods")
        plt.savefig("all_best_comp.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
