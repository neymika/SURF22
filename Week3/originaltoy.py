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

    axs[0,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsigfuncs, '-.', markeredgecolor="none", \
    label=f'Gradient Descent')
    axs[0,j].set_yscale('log')

    axs[1,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsigolosses, '-.', markeredgecolor="none", \
    label=f'Gradient Descent')
    axs[1,j].set_yscale('log')

    axs[2,j].plot(range(0, gdsigfuncs.shape[0], 1), gdsiglosses, '-.', markeredgecolor="none", \
    label=f'Gradient Descent')
    axs[2,j].set_yscale('log')


for mini in range(len(minibatches)):
    # gradn = int(1000/minibatches[mini])
    sizes = histsigsgdfuncs[minibatches[mini]].shape[0]
    axs[0, 0].plot(range(0, sizes, 1), \
    histsigsgdfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
    label=f'minibatch size = {minibatches[mini]}')

    axs[1, 0].plot(range(0, sizes, 1), \
    histsigsgdolosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

    axs[2, 0].plot(range(0, sizes, 1), \
    histsigsgdlosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

    # gradn = int(np.ceil(1000/(minibatches[mini]+1000)))
    sizes = histsignfuncs[minibatches[mini]].shape[0]
    axs[0, 1].plot(range(0, sizes, 1), \
    histsignfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
    label=f'minibatch size = {minibatches[mini]}')

    axs[1, 1].plot(range(0, sizes, 1), \
    histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

    axs[2, 1].plot(range(0, sizes, 1), \
    histsignolosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

    sizes = histsigfuncs[minibatches[mini]].shape[0]
    axs[0, 2].plot(range(0, sizes, 1), \
    histsigfuncs[minibatches[mini]], '-.', markeredgecolor="none",  \
    label=f'minibatch size = {minibatches[mini]}')

    axs[1, 2].plot(range(0, sizes, 1), \
    histsigolosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

    axs[2, 2].plot(range(0, sizes, 1), \
    histsiglosses[minibatches[mini]], '-.', markeredgecolor="none", \
    label=f'minibatch size = {minibatches[mini]}')

fig.set_size_inches(18, 18)

plt.legend(bbox_to_anchor=(1,0.25), loc='upper left', ncol=1, title="Optimization Methods")
plt.savefig("toy_net.png", bbox_inches='tight')
