import numpy as np
import matplotlib.pyplot as plt
from configs import TOTAL_EPSIODES

def plot_runtime(eval_data, deadline):
    fig, axs = plt.subplots(2, 4, figsize=(24, 6))
    for i in range(2):
        for j in range(4):
            axs[i][j].plot(eval_data['runtime'], color='dimgrey')
            violated_data = [[inf_num, runtime] for inf_num, runtime in enumerate(eval_data['runtime']) if runtime > deadline]
            print(violated_data)
            axs[i][j].axhline(y=deadline, color='k', linestyle='--', label="deadline")
            axs[i][j].plot(*zip(*violated_data), 'x', color='k', label="violation")
            if i % 2 == 1:
                axs[i][j].set_xlabel("Frame number")
            if j == 0:
                axs[i][j].set_ylabel("Runtime (ms)")
            axs[i][j].set_ylim(0, deadline+1)
            axs[i][j].legend()
    plt.rcParams.update({'font.size': 18})
    fig.tight_layout()
    plt.savefig("adaptation.pdf")

def plot_energy(eval_data):
    plt.plot(eval_data['energy'], color='dimgrey')
    # violated_data = [[inf_num, runtime] for inf_num, runtime in enumerate(eval_data['runtime']) if runtime > deadline]
    # plt.axhline(y=deadline, color='k', linestyle='--', label="deadline")
    # plt.plot(*zip(*violated_data), 'x', color='k', label="violation")
    plt.xlabel("Frame number")
    plt.ylabel("Energy (mJ)")
    # plt.ylim(0, deadline+1)
    plt.legend()
    plt.show()

def average_of_consecutive_m(arr, m):
    arr = np.array(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    n = len(arr)
    result = []
    
    for i in range(0, n, m):
        start = i
        end = min(i + m, n)
        average = sum(arr[start:end]) / (end - start)
        result.append(average)
    
    return result

def plot_reward_history(rewards):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for i in range(4):
        m = 50
        axs[i//2][i%2].plot(average_of_consecutive_m(rewards[i]["data"], m), color="dimgrey")
        axs[i//2][i%2].set_ylim(0.5, 1.05)
        xticks = np.arange(0, TOTAL_EPSIODES/m+1, TOTAL_EPSIODES/(m*5))
        axs[i//2][i%2].set_xticks(xticks)
        axs[i//2][i%2].set_xlabel(rewards[i]["xlabel"])
        temps = ["25ºC", "35ºC", "45ºC", "35ºC", "25ºC"]
        if i%2 == 0:
            axs[i//2][i % 2].set_ylabel("Normalized reward")
        
        for j in range(5):
            axs[i//2][i%2].axvline(x=xticks[j], linestyle=':', color='k')
            x1, y1 = xticks[j]+3, 0.6
            x2, y2 = xticks[j+1]-3, 0.6
            axs[i//2][i%2].plot([x1, x2], [y1, y2], linestyle='--', color='k')
            axs[i//2][i%2].annotate(temps[j], xy=(x1 + (x2 - x1)/2, y1 - 0.05), ha='center')

    plt.rcParams.update({'font.size': 18})
    fig.tight_layout()
    plt.savefig("rewards.pdf")
