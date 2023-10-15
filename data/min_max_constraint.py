import json
from os import walk
from statistics import median, mean

DATA_DIR = "./"
MODELS = ["mobilenetv2", "alexnet", "resnet", "inception3", "googlenet", "squeezenet", "shufflenetv2"]
TEMPS = [25, 70]

collected_data = {}

for model in MODELS:
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

for model in MODELS:
    for const in ["energy"]:
        for temp in TEMPS:
            collected_data[model][temp] = sorted(collected_data[model][temp], key=lambda x: x[const])
            print(model, temp, const)
            print(collected_data[model][temp][0][const], 
            collected_data[model][temp][len(collected_data[model][temp])//2][const],
            collected_data[model][temp][-1][const])
