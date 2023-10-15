import utils as utils
import multiprocessing as mp
from collections import defaultdict
import time
import logging

class ThermalMonitor:
    @staticmethod
    def get_temp_all():
        res = dict()
        res['cpu'] = ThermalMonitor.get_temp_cpu()
        res['gpu'] = ThermalMonitor.get_temp_gpu()

        return res

    @staticmethod
    def get_temp_cpu():
        return float(utils.read_file("/sys/devices/virtual/thermal/thermal_zone0/temp")) / 1000

    @staticmethod
    def get_temp_gpu():
        return float(utils.read_file("/sys/devices/virtual/thermal/thermal_zone1/temp")) / 1000


class FanController:
    @staticmethod
    def get_speed():
        return int(utils.read_file("/sys/devices/pwm-fan/target_pwm"))

    @staticmethod
    def set_speed(speed):
        if 255 >= int(speed) >= 0:
            utils.write_file("/sys/devices/pwm-fan/target_pwm", str(speed))

    @staticmethod
    def set_speed_to_max():
        FanController.set_speed(255)

def collect_pwr_samples(pwr_meter, exit_flag: mp.Event, saved_results: mp.Event,
                        keep_sampling: mp.Event, tmp_results_ready: mp.Event,
                        num_observations: mp.Value,
                        sampling_rate, tmp_file, pausing_tmp_file, board_power_monitor):
    res = defaultdict(list)
    stopwatch = utils.StopWatch()
    stopwatch.start()

    if pwr_meter == 'internal':
        while True:
            if exit_flag.is_set():
                break

            if not keep_sampling.is_set():
                utils.FileUtils.silent_remove(pausing_tmp_file)
                utils.FileUtils.serialize(res['p_all'], pausing_tmp_file, save_as='list')
                tmp_results_ready.set()
                keep_sampling.wait()

            p = board_power_monitor.get_pmu_reading_all_watts()

            res['p_gpu'].append(p["gpu"])
            res['p_cpu'].append(p["cpu"])
            res['p_soc'].append(p["soc"])
            res['p_ddr'].append(p["ddr"])
            res['p_all'].append(p["all"])
            num_observations.value = len(res['p_all'])
            time.sleep(1 / sampling_rate)

    res['duration_sec'] = stopwatch.elapsed_s()
    utils.FileUtils.serialize(res, tmp_file)
    saved_results.set()

class PowerMonitor(mp.Process):
    tmp_file = "/tmp/tmp_data.dat"
    pausing_tmp_file = "/tmp/tmp_pausing_data.json"

    def __init__(self, power_meter: str = 'internal', logging_level=logging.INFO,
                 sampling_rate=1, board_power_monitor=None):
        self.exit_flag = mp.Event()
        self.saved_results = mp.Event()
        self.keep_sampling = mp.Event()
        self.tmp_results_ready = mp.Event()
        self.keep_sampling.set()
        self.num_observations = mp.Value("i", 0)
        self.power_meter = power_meter

        utils.FileUtils.silent_remove(self.tmp_file)
        utils.FileUtils.silent_remove(self.pausing_tmp_file)

        self.readings = None
        super().__init__(target=collect_pwr_samples,
                         args=(power_meter, self.exit_flag, self.saved_results, self.keep_sampling,
                               self.tmp_results_ready, self.num_observations, sampling_rate,
                               self.tmp_file, self.pausing_tmp_file, board_power_monitor))

    def resume_sampling(self):
        self.keep_sampling.set()

    def pause_sampling(self):
        self.tmp_results_ready.clear()
        self.keep_sampling.clear()

    def is_paused(self):
        return not self.keep_sampling.is_set()

    def get_records(self):
        if self.is_paused():
            self.tmp_results_ready.wait()
            records = utils.FileUtils.deserialize(self.pausing_tmp_file)
            return records
        return None

    def get_num_observations(self):
        return self.num_observations.value

    def stop(self):
        if self.is_paused():
            self.resume_sampling()
        self.exit_flag.set()
        self.join(timeout=10)
        counter = 0
        while not self.saved_results.is_set() and counter < 600:
            time.sleep(0.1)
            counter += 1
        if counter == 600 and not self.saved_results.is_set():
            raise Exception("Timed out while waiting on saving results to disk.")
        else:
            self.readings = utils.FileUtils.deserialize(self.tmp_file)
            utils.FileUtils.silent_remove(self.tmp_file)
            utils.FileUtils.silent_remove(self.pausing_tmp_file)
