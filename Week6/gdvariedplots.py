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

def main():
    s = 100

    print("Performing SGD at multiple different stepsizes")
    stepsizes = [1, .1, .01, .001, .0001, .00001]
    decayschedule = [5, 10, 20, 50, 100]

    histsiglosses = pd.read_csv("gdlosses.csv")
    histsiglosses = histsiglosses.apply(lambda x: pd.Series(x.dropna().values))
    histsigfuncs = pd.read_csv("gdfunc.csv")
    histsigfuncs = histsigfuncs.apply(lambda x: pd.Series(x.dropna().values))
    histsigolosses = pd.read_csv("gdolosses.csv")
    histsigolosses = histsigolosses.apply(lambda x: pd.Series(x.dropna().values))

    params = {'legend.fontsize': 'x-large',
         'figure.titlesize':'x-large',
         'figure.figsize': (35, 30),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    with mpl.rc_context(params):
        fig, axs = plt.subplots(6, 5, sharey='row')
        fig.suptitle(fr'||$\nabla$L(w)|| vs Iterations for GD')

        for i in range(len(stepsizes)):
            for j in range(len(decayschedule)):
                print(f'GD with step size = {stepsizes[i]} and decay Iteration = {decayschedule[j]}')
                dictkey = str((i+1)*10 + j)

                axs[i, j].plot(range(histsigfuncs[dictkey].shape[0]), histsigolosses[dictkey], '-')
                axs[i, j].set_yscale('log')
                axs[i, j].set_title(\
                rf'$\eta$: {stepsizes[i]}, $\eta = 1/s$ Starting Iteration: {decayschedule[j]}')


        plt.setp(axs[-1, :], xlabel='Iterations')
        plt.setp(axs[:, 0], ylabel=r'||$\nabla$L(w)|| Values')
        plt.savefig("toy_gdtradeoff_oloss.png", bbox_inches='tight')

        fig, axs = plt.subplots(6, 5, sharey='row')
        fig.suptitle(fr'L(w) vs Iterations for GD')

        for i in range(len(stepsizes)):
            for j in range(len(decayschedule)):
                print(f'GD with step size = {stepsizes[i]} and decay Iteration = {decayschedule[j]}')
                dictkey = str((i+1)*10 + j)

                print(histsigfuncs[dictkey])
                axs[i, j].plot(range(histsigfuncs[dictkey].shape[0]), histsigfuncs[dictkey], '-')
                axs[i, j].set_yscale('log')
                axs[i, j].set_title(\
                rf'$\eta$: {stepsizes[i]}, $\eta = 1/t$ Starting Iteration: {decayschedule[j]}')


        plt.setp(axs[-1, :], xlabel='Iterations')
        plt.setp(axs[:, 0], ylabel=r'L(w) Values')
        plt.savefig("toy_gdtradeoff_func.png", bbox_inches='tight')



if __name__ == "__main__":
    main()
