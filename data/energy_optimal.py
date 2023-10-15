import json
from statistics import median, mean
from turtle import color
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import matplotlib.lines as mlines

ENERGY_BUDGET = [120, 100, 130, 160]

DATA_DIR = f"./"
MODELS = ["squeezenet", "mobilenetv2", "alexnet", "googlenet"]
TEMPS = [25, 70]

fig, ax = plt.subplots(figsize=(12, 6))
font_size=14
plt.rcParams.update({'font.size': font_size})

for index, model in enumerate(MODELS):
    print(model)
    energy = {}
    collected_data = {}
    for temp in TEMPS:
        dir = f"{DATA_DIR}/{model}/{temp}"
        files = next(walk(dir), (None, None, []))[2]
        collected_data[temp] = []
        for file in files:
            if file == ".DS_Store":
                continue
            with open(f"{dir}/{file}", 'r') as f:
                file_jsons = json.load(f)
            for file_json in file_jsons:
                data = {}
                data["runtime"] = float("{:.2f}".format(median(file_json["runtime_ms"])))
                data["power"] = float("{:.2f}".format(mean(file_json["power_readings"]["p_all"][4:-1])))
                data["energy"] = float("{:.2f}".format(data["runtime"] * data["power"]))
                data["dvfs_config"] = file_json["dvfs_config"]
                collected_data[temp].append(data)
        satisfy_energy = []
        for data in collected_data[temp]:
            if data["energy"] < ENERGY_BUDGET[index]:
                satisfy_energy.append(data)
        energy[temp] = sorted(satisfy_energy, key=lambda d: d['runtime'])

    # print(energy[25][0])
    # print(energy[70][0])
    opt25in70, opt70in25 = 0, 0
    for i in range(len(collected_data[70])):
       if collected_data[70][i]["dvfs_config"] == energy[25][0]["dvfs_config"]:
           opt25in70 = collected_data[70][i]
        #    print(collected_data[70][i]["energy"] / ENERGY_BUDGET[index])
    for i in range(len(collected_data[25])):
       if collected_data[25][i]["dvfs_config"] == energy[70][0]["dvfs_config"]:
            opt70in25 = collected_data[25][i]
            # print(collected_data[25][i])
    
    ax = plt.subplot(2, 2, index+1)
    # # Hide the right and top spines
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)

    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    
    
    # if index == 0 or index == 2:
    plt.ylabel("Energy (mJ)", fontsize=font_size)
    # if index == 2 or index == 3:
    plt.xlabel("Runtime (ms)", fontsize=font_size)
    
    plt.plot(energy[25][0]["runtime"], energy[25][0]["energy"], marker="o", markerfacecolor='none', markersize=8, color="grey")
    plt.plot(opt70in25["runtime"], opt70in25["energy"], marker="^", markersize=8, color="grey", markerfacecolor='none')
    
    plt.plot(energy[70][0]["runtime"], energy[70][0]["energy"], marker="^", markersize=8, color="black", markerfacecolor='none')
    plt.plot(opt25in70["runtime"], opt25in70["energy"], marker="o", markersize=8, color="black", markerfacecolor='none')
    
    plt.axhline(y=ENERGY_BUDGET[index], color='grey', linestyle='--')
    
    plt.title(MODELS[index].title(), y=-0.005)
    
    #for showing legends
    l1 = mlines.Line2D([], [], color='grey', marker='o', markerfacecolor='none', linestyle='None',
                          markersize=8, label=r'$CGM_{optimal}^{25ºC}$ run at 25ºC')
    l2 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=8, label=r'$CGM_{optimal}^{25ºC}$ run at 70ºC', markerfacecolor='none')
    l3 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                            markersize=8, label=r'$CGM_{optimal}^{70ºC}$ run at 70ºC', markerfacecolor='none')
    l4 = mlines.Line2D([], [], color='grey', marker='^', linestyle='None',
                            markersize=8, label=r'$CGM_{optimal}^{70ºC}$ run at 25ºC', markerfacecolor='none')
    l5 = mlines.Line2D([], [], color='dimgrey', marker='_', linestyle='None',
                            markersize=8, label='energy budget')


plt.legend(handles=[l1, l4, l2, l3, l5], fontsize=font_size-2, loc='upper center')
plt.tight_layout()
plt.savefig("opt_cgm_thermal_ignorance.pdf")

