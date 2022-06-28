import numpy as np
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

# Functions and Variance Estimates based on
def highfid(z, a, b):
    return np.sin(z[0]) + a*np.power(np.sin(z[1]), 2) + b*np.power(z[2], 4)*np.sin(z[0])

def lowfid(z, a, b):
    return np.sin(z[0]) + .6*a*np.power(np.sin(z[1]), 2) + 9*b*np.power(z[2], 2)*np.sin(z[0])

def varest(fz, e):
    return np.sum((fz - e)**2)/(fz.shape[0] - 1)

def calculate_exp(a = 5, b = .1, lowsample = 9070, highsample = 8, budget = 200,
    alpha = .9455):
    exphs = np.zeros(shape=(budget,))
    expls = np.zeros(shape=(budget,))
    expmfs = np.zeros(shape=(budget,))

    for i in range(budget):
        z = np.random.uniform(-np.pi, np.pi, size=(3,lowsample))
        exphs[i] = np.mean(highfid(z[:, :highsample], a, b))
        expls[i] = np.mean(lowfid(z, a, b))
        expmfs[i] = exphs[i] + alpha*(expls[i] - np.mean(lowfid(z[:, :highsample], a, b)))

    return exphs, expls, expmfs

def main():
    print("IN HERE")
    try:
        filename = sys.argv[1]
        try:
            args = float(sys.argv[2:])
        except Exception as e:
            print("To control parameters please add the arguments in this order:")
            print("toymultifidelity.py filename.png a b lowsample highsample budget alpha")
            args = None
    except Exception as e:
        print("Must at least pass desired plot name")
        return

    if args == None:
        print("CORRECT")
        exps = calculate_exp()
    elif len(args) == 6:
        exps = calculate_exp(a = args[0], b = args[1], lowsample = args[2],
        highsample = args[3], budget = args[4], alpha = args[5])
    else:
        print("Incorrect number of arguments")
        return

    hhist, hedges = np.histogram(exps[0], density=True, bins=50)
    lhist, ledges = np.histogram(exps[1], density=True, bins=50)
    mfhist, mfedges = np.histogram(exps[2], density=True, bins=50)
    try:
        p = bokeh.plotting.figure(height=300, width=800, x_axis_label="Expected Values", \
                              y_axis_label="Frequency")
        p.quad(top=hhist, bottom=0, left=hedges[:-1], right=hedges[1:],
                   fill_color="navy", line_color="navy", alpha=0.5)
        p.quad(top=lhist, bottom=0, left=ledges[:-1], right=ledges[1:],
                   fill_color="seagreen", line_color="seagreen", alpha=0.5)
        p.quad(top=mfhist, bottom=0, left=mfedges[:-1], right=mfedges[1:],
                   fill_color="maroon", line_color="maroon", alpha=0.5)
        p.title.align = "center"
        legend = bokeh.models.Legend(items=[("Expected Values of High-Fidelity Model", [p.line(line_color="navy")]),
                                            ("Expected Values of Low-Fidelity Model", [p.line(line_color="seagreen")]),
                                            ("Expected Values of Multi-Fidelity Model", [p.line(line_color="maroon")])
                                           ], location = "center")
        p.add_layout(legend, "right")
        # bokeh.io.show(style(p))
        p.background_fill_color = None
        p.border_fill_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(style(p), filename=filename)
    except Exception as e:
        print("Unable to use bokeh. Using matplotlib instead")
        plt.stairs(hhist, hedges, fill=True, label='Expected Values of High-Fidelity Model')
        plt.stairs(lhist, ledges, fill=True, label='Expected Values of Low-Fidelity Model')
        plt.stairs(mfhist, mfedges, fill=True, label='Expected Values of Multi-Fidelity Model')
        plt.title("Histogram of Different Fidelity Expected Values")
        plt.xlabel("Expected Values")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    main()
