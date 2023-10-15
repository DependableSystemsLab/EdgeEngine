from cProfile import label
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
font_size = 14
lw=1

def plot(ax, deadline, autoscale,  neuos, edgeengine, model):
    ax.axhline(y=deadline, color='grey', linestyle='--', label="deadline", linewidth=lw)
    ax.bar(1, neuos, color='black', linestyle='-', width=0.4)
    ax.bar(2, autoscale, color='#B7B7B7', linestyle='-', width=0.4)
    ax.bar(3, edgeengine, color='#666666', linestyle='-', width=0.4)
    ax.set_xlabel(model, fontsize=font_size)
    ax.set_ylabel("Runtime (ms)", fontsize=font_size)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticks([1, 2, 3], ["NeuOS", "Autoscale", "EdgeEngine"],fontsize=font_size)
    ax.set_ylim(autoscale-4)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)


plot(ax[0][0], 13, 12.83, 13.13, 12.88, "Mobilenetv2")
plot(ax[0][1], 20, 15.29, 20.81, 17.99, "Alexnet")
plot(ax[1][0], 20, 15.53, 20.69, 19.33, "Resnet")
plot(ax[1][1], 75, 70.63, 76.14, 74.52, "Inception3")

handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=font_size)
plt.tight_layout()
plt.savefig('autoscale_neuos_runtime.pdf')
