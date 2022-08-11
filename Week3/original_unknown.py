params = {'legend.fontsize': 'x-large',
        'figure.figsize': (18, 9),
       'axes.labelsize': 'x-large',
       'axes.titlesize':'x-large',
       'xtick.labelsize':'x-large',
       'ytick.labelsize':'x-large'}
  with mpl.rc_context(params):
      fig, axs = plt.subplots(3, 3)

      for j in range(3):
          if j == 0:
              plt.setp(axs[0, j], title = 'SGD')
              plt.setp(axs[j, 0], ylabel='L(w) Values')
          elif j == 1:
              plt.setp(axs[0, j], title = 'Naive SVRG')
              plt.setp(axs[j, 0], ylabel=r'||$\nabla$L(w)|| Values')
          else:
              plt.setp(axs[0, j], title = 'Random SVRG')
              plt.setp(axs[j, 0], ylabel='||Δw|| Values')
          plt.setp(axs[:, j], xlabel=r'Major Epochs (# of $\nabla$Ψ(w)/m)')

          for i in range(3):
              axs[i,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, '-.', markeredgecolor="none", \
              label=f'Gradient Descent')
              axs[i,j].set_yscale('log')

      for mini in range(len(minibatches)):
          gradn = int(200/minibatches[mini])
          sizes = histsigsgdfuncs[minibatches[mini]].shape[0]
          axs[0, 0].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsigsgdfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
          label=f'minibatch size = {minibatches[mini]}')

          axs[1, 0].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsigsgdolosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

          axs[2, 0].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsigsgdlosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

          gradn = int(np.ceil(200/(minibatches[mini]+200)))
          sizes = histsignfuncs[minibatches[mini]].shape[0]
          axs[0, 1].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsignfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
          label=f'minibatch size = {minibatches[mini]}')

          axs[1, 1].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

          axs[2, 1].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

          sizes = histsigfuncs[minibatches[mini]].shape[0]
          axs[0, 2].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsigfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
          label=f'minibatch size = {minibatches[mini]}')

          axs[1, 2].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsigolosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

          axs[2, 2].plot(np.linspace(start=0, stop=(sizes/gradn), num=sizes), \
          histsiglosses[minibatches[mini]], '-.', markeredgecolor="none", \
          label=f'minibatch size = {minibatches[mini]}')

      fig.set_size_inches(18, 18)

      plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Optimization Methods")
      plt.savefig("unknown_net.png", bbox_inches='tight')
