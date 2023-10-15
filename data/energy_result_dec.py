import json
from statistics import median, mean, mode
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import pandas as pd

DATA_DIR = "./"
MODELS = ["mobilenetv2", "alexnet", "resnet", "inception3"]
ENERGY_BUDGET_TIGHT = [98, 125, 150, 500]
ENERGY_BUDGET_LOOSE = [110, 150, 200, 600]
TEMPS = [25, 70]

fig, ax = plt.subplots(figsize=(12, 6))
plt.rcParams.update({'font.size': 18})

def find_data(ebudgets):
    energy = {}
    opt25in70, opt70in25 = {}, {}
    collected_data = {}

    for model, budget in zip(MODELS, ebudgets):
        energy[model] = {}
        collected_data[model] = {}
        for temp in TEMPS:
            dir = f"{DATA_DIR}/{model}/{temp}"
            files = next(walk(dir), (None, None, []))[2]
            collected_data[model][temp] = []
            for file in files:
                if file == ".DS_Store":
                    continue
                with open(f"{dir}/{file}", 'r') as f:
                    file_jsons = json.load(f)
                for file_json in file_jsons:
                    data = {}
                    data["runtime"] = median(file_json["runtime_ms"])
                    data["power"] = mean(file_json["power_readings"]["p_all"][4:-1])
                    data["energy"] = data['power'] * data['runtime']
                    data["dvfs_config"] = file_json["dvfs_config"]
                    collected_data[model][temp].append(data)

            satisfy_fps = []
            for data in collected_data[model][temp]:
                if data["energy"] < budget:
                    satisfy_fps.append(data)

            energy[model][temp] = sorted(satisfy_fps, key=lambda d: d['runtime'])

        for i in range(len(collected_data[model][70])):
            if collected_data[model][70][i]["dvfs_config"] == energy[model][25][0]["dvfs_config"]:
                opt25in70[model] = collected_data[model][70][i]
        for i in range(len(collected_data[model][25])):
            if collected_data[model][25][i]["dvfs_config"] == energy[model][70][0]["dvfs_config"]:
                opt70in25[model] = collected_data[model][25][i]
    return energy, opt70in25

energy_loose, opt70in25_loose = find_data(ENERGY_BUDGET_LOOSE)
energy_tight, opt70in25_tight = find_data(ENERGY_BUDGET_TIGHT)

TAF, TIF = [], []
for model in MODELS:
    TAF.append(energy_loose[model][25][0]["runtime"] / energy_loose[model][25][0]["runtime"])
    TIF.append(opt70in25_loose[model]["runtime"] / energy_loose[model][25][0]["runtime"])

    TAF.append(energy_tight[model][25][0]["runtime"] / energy_tight[model][25][0]["runtime"])
    TIF.append(opt70in25_tight[model]["runtime"] / energy_tight[model][25][0]["runtime"])

print(mean(TIF), max(TIF))

# cell_text = []
# cell_header = [['', *[f'{x}' for x in MODELS]]]
# cell_text.append(["EdgeEngine", *[f'{x:1.2f}' for x in TAF]])
# cell_text.append(["Temp-Oblivious", *[f'{x:1.2f}' for x in TIF]])
# df = pd.DataFrame(cell_text, columns = cell_header)
# print(df)

x_axis = np.arange(0, len(MODELS) * 4, 2)
plt.bar(x_axis-0.3, TAF, width=0.6, label = 'EdgeEngine', color='black')
plt.bar(x_axis+0.3, TIF, width=0.6, label = 'TOF', color='grey')

plt.ylabel("Normalized runtime", fontsize=18)
xticks_tuples = [(f"{m.title()}\n-loose", f"{m.title()}\n-tight") for m in MODELS]
xticks = [m for t in xticks_tuples for m in t]
plt.xticks(x_axis, xticks, fontsize=14)
plt.ylim(0.7, 1.3)
plt.legend()
plt.tight_layout()
plt.savefig("eval_energy_temp_decrease.pdf")