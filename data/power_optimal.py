from audioop import avg
import json
from statistics import median, mean, mode
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import pandas as pd

DATA_DIR = "./"
MODELS = ["mobilenetv2", "alexnet", "resnet", "inception3"]
DEADLINE_TIGHT = [13, 14, 14, 66]
DEADLINE_LOOSE = [18, 19, 20, 85]
TEMPS = [25, 70]

def find_data(deadlines):
    energy = {}
    opt25in70, opt70in25 = {}, {}
    collected_data = {}

    for model, deadline in zip(MODELS, deadlines):
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
                if data["runtime"] < deadline:
                    satisfy_fps.append(data)

            energy[model][temp] = sorted(satisfy_fps, key=lambda d: d['power'])

        for i in range(len(collected_data[model][70])):
            if collected_data[model][70][i]["dvfs_config"] == energy[model][25][0]["dvfs_config"]:
                opt25in70[model] = collected_data[model][70][i]
        for i in range(len(collected_data[model][25])):
            if collected_data[model][25][i]["dvfs_config"] == energy[model][70][0]["dvfs_config"]:
                opt70in25[model] = collected_data[model][25][i]
    return energy, opt70in25

energy_loose, opt25in70_loose = find_data(DEADLINE_LOOSE)
energy_tight, opt25in70_tight = find_data(DEADLINE_TIGHT)

TAF_e, TIF_e = [], []
TAF_r, TIF_r = [], []
for model, lb, tb in zip(MODELS, DEADLINE_LOOSE, DEADLINE_TIGHT):
    TAF_r.append(energy_loose[model][25][0]["runtime"])
    TAF_r.append(energy_tight[model][25][0]["runtime"])

    TIF_r.append(opt25in70_loose[model]["runtime"])
    TIF_r.append(opt25in70_tight[model]["runtime"])

    TAF_e.append(energy_loose[model][25][0]["energy"])
    TAF_e.append(energy_tight[model][25][0]["energy"])

    TIF_e.append(opt25in70_loose[model]["energy"])
    TIF_e.append(opt25in70_tight[model]["energy"])



# print(TIF_r)

# print(TAF_r)
print(TAF_e)
print(TIF_e)
