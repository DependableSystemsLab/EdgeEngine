import json
from statistics import median, mean
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

DATA_DIR = "./"
MODELS = ["squeezenet", "mobilenetv2", "alexnet", "googlenet", "inception3", "resnet", "shufflenetv2"]
DEADLINES = [13, 14, 15, 18, 60, 15, 16]
TEMPS = [25, 70]

fps = {}
optimized_25_in_70 = {}
fig, ax = plt.subplots(figsize=(12, 6))
plt.rcParams.update({'font.size': 15})


for model, deadline in zip(MODELS, DEADLINES):
    fps[model] = {}
    for temp in TEMPS:
        dir = f"{DATA_DIR}/{model}/{temp}"
        files = next(walk(dir), (None, None, []))[2]
        collected_data = []
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
                collected_data.append(data)

        satisfy_fps = []
        for data in collected_data:
            if data["runtime"] * 0.95 < deadline:
                satisfy_fps.append(data)

        fps[model][temp] = sorted(satisfy_fps, key=lambda d: d['power'])

    print(fps[model][25][0])
    print(fps[model][70][0])

    # if not fps[model][25][0]["dvfs_config"] == fps[model][70][0]["dvfs_config"]:
    optimized_25 = fps[model][25][0]
    optimized_25_in_70[model] = {}
    for i in range(len(collected_data)):
        if collected_data[i]["dvfs_config"] == optimized_25["dvfs_config"]:
            optimized_25_in_70[model] = collected_data[i]
            print(optimized_25_in_70[model])
    print()

TAF, TIF = [], []
for model, deadline in zip(MODELS, DEADLINES):
    TAF.append(fps[model][70][0]["runtime"] / deadline)
    TIF.append(optimized_25_in_70[model]["runtime"] / deadline)

cell_text = []
cell_header = [['', *[f'{x}' for x in MODELS]]]
cell_text.append(["Deadline", *[f'{x:1.2f}' for x in DEADLINES]])
cell_text.append(["TAF-E", *[f'{x:1.2f}' for x in TAF]])
cell_text.append(["NeuOS", *[f'{x:1.2f}' for x in TIF]])
df = pd.DataFrame(cell_text, columns = cell_header)
print(df)

x_axis = np.arange(0, len(MODELS) * 2, 2)
# plt.bar(x_axis-0.2, TAF, width=0.4, label = 'TAF-E', color='#2C7BB6')
plt.bar(x_axis+0.2, TIF, width=0.4, color='dimgrey')

plt.ylabel("Normalized runtime", fontsize=18)

xticks = [f"{m.title()}" for m in MODELS]
plt.minorticks_on()
                                           
plt.xticks(x_axis, xticks, fontsize=14)
plt.axhline(y=1, color='grey', linestyle='--', label="deadline")
plt.ylim(0, 1.2)
plt.tight_layout()
plt.legend()
plt.savefig("deadline_violation.pdf")