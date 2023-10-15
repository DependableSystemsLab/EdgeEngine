import json
import numpy as np
import configs
import utils as utils
from scipy.stats import genextreme

from abc import ABC, abstractmethod
from os import walk
from statistics import median, mean

class IBoard(ABC):

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def CPU_FREQS(self):
        pass

    @property
    @abstractmethod
    def GPU_FREQS(self):
        pass

    @property
    @abstractmethod
    def MEM_FREQS(self):
        pass
    
    @abstractmethod
    def get_temperature(self):
        pass

    @abstractmethod
    def set_dvfs_settings(self, new_dvfs):
        pass
    
    @abstractmethod
    def run_inference(self):
        pass


class JetsonTX2(IBoard):
    CPU_FREQS = [806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200]
    GPU_FREQS = [420750000, 522750000, 624750000, 726750000, 854250000, 930750000, 1032750000, 1122000000, 1236750000, 1300500000]
    MEM_FREQS = [800000000, 1062400000, 1331200000, 1600000000, 1866000000]

    def __init__(self) -> None:
        super().__init__()
        self.client = LearnOffPolicy()
        self.temperature = None
    
    def set_dvfs_settings(self, new_dvfs):
        self.current_dvfs = str(new_dvfs)

    def get_temperature(self):
        return self.temperature
        
    def run_inference(self):
        return self.client.run_inference(self.current_dvfs, self.temperature)

class LearnOffPolicy:
    def __init__(self):
        self.DATA_DIR = f"../data/{configs.MODEL_NAME}"
        self.TEMPS = [25, 35, 45, 55, 65, 70]
        self.noise_c_val = -0.08
        self.noise_scale_val = 0.03
        
        self.collected_data = {}
        for temp in self.TEMPS:
            dir = f"{self.DATA_DIR}/{temp}"
            files = next(walk(dir), (None, None, []))[2]
            if len(files) == 0:
                continue
            self.collected_data[temp] = {}
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(f"{dir}/{file}", 'r') as f:
                    file_jsons = json.load(f)
                for file_json in file_jsons:
                    data = {}
                    data["runtime"] = float("{:.2f}".format(median(file_json["runtime_ms"])))
                    data["power"] = float("{:.2f}".format(mean(file_json["power_readings"]["p_all"][4:-1])))
                    data["p_cpu"] = float("{:.2f}".format(mean(file_json["power_readings"]["p_cpu"][4:-1])))
                    data["p_gpu"] = float("{:.2f}".format(mean(file_json["power_readings"]["p_gpu"][4:-1])))
                    data["p_ddr"] = float("{:.2f}".format(mean(file_json["power_readings"]["p_ddr"][4:-1])))
                    data["energy"] = float("{:.2f}".format(data["runtime"] * data["power"]))
                    dvfs_config = np.array([file_json["dvfs_config"]["cpu_freq"],
                                            file_json["dvfs_config"]["gpu_freq"], 
                                            file_json["dvfs_config"]["memory_freq"]])
                    self.collected_data[temp][str(dvfs_config)] = data
    
    def check_data(self):
        dvfs_config = str(np.array([1881600, 854250000, 1600000000]))
    
    def get_runtime_after_noise(self, pure_runtime):
        return genextreme.rvs(self.noise_c_val, pure_runtime, self.noise_scale_val)
    
    def run_inference(self, dvfs_config, temp, noise=True):
        data = {}
        if temp in self.collected_data.keys():
            data = self.collected_data[temp][dvfs_config]
        else:
            low_temp, high_temp = list(self.collected_data.keys())[0], list(self.collected_data.keys())[-1]
            slope = (temp - low_temp) / (high_temp - low_temp)
            data["runtime"] = self.collected_data[low_temp][dvfs_config]["runtime"] + slope * abs(self.collected_data[high_temp][dvfs_config]["runtime"] - self.collected_data[low_temp][dvfs_config]["runtime"])
            data["power"] = self.collected_data[low_temp][dvfs_config]["power"] + slope * (self.collected_data[high_temp][dvfs_config]["power"] - self.collected_data[low_temp][dvfs_config]["power"])
            data["energy"] = self.collected_data[low_temp][dvfs_config]["energy"] + slope * (self.collected_data[high_temp][dvfs_config]["energy"] - self.collected_data[low_temp][dvfs_config]["energy"])

        metrics = {}
        if noise:
            metrics["runtime"] = self.get_runtime_after_noise(data["runtime"])
        else:
            metrics["runtime"] = data["runtime"]
        metrics["power"] = data["power"]
        metrics["energy"] = data["power"] * metrics["runtime"]
        return metrics
