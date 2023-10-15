import logging, sys, csv
from pickle import TRUE
from time import time
from qtable import Action, QTable
import numpy as np
import matplotlib.pyplot as plt

from board import IBoard, JetsonTX2
import configs
import utils
from configs import EPISLON, MIN_EPSILON, RUNTIME_GUARD, LEARN_TEMPS, EVALUATE_TEMPS,\
                    LOGGING_LEVEL, TEMPS, MODELS, ALPHAS, CONSTRAINTS, LOW_TEMP, HIGH_TEMP
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S %p', stream=sys.stdout, level=LOGGING_LEVEL)

ALPHA, ENERGY_BUDGET, TARGET_DEADLINE = 0, 0, 0

class Agent():
    def __init__(self, board: IBoard):
        self.round_num = 0
        self.epsilon = [EPISLON for _ in range(len(TEMPS))]
        self.client = board
        temp_list = [i for i in range(len(TEMPS))]
        self.states = np.array(np.meshgrid(temp_list, board.CPU_FREQS, board.GPU_FREQS, board.MEM_FREQS, indexing='ij')).T.reshape(-1,4)
        self.Q = QTable(self.client, self.states, Action)
        self.cpu_index = len(board.CPU_FREQS) - 3
        self.gpu_index = len(board.GPU_FREQS) - 3
        self.mem_index = len(board.MEM_FREQS) - 3
        self.current_state = np.array([])
        self.last_freqs = {}
        self.best_freqs = {}
        self.oracle_data = np.array([{} for i in range(len(TEMPS))])
        self.measured_data = {}
        self.client.temperature = LOW_TEMP
        self.num_of_violations_training = np.array([0 for i in range(len(TEMPS))])
        self.num_of_violations_evaluations = 0
        self.evaluation = False
        self.toggled = False
        self.eval_data = {'runtime': [], 'energy': []}
        self.reward_history = []
    
    def clean(self):
        self.round_num = 0
        self.measured_data = {}
        
    def get_current_state(self):
        self.temp_index = self.get_current_temp_index()
        current_state = np.array([self.temp_index, *self.get_dvfs_setting()])
        return current_state
    
    def get_current_state_index(self):
        temp_index = self.get_current_temp_index()
        return [temp_index, self.cpu_index, self.gpu_index, self.mem_index]
    
    def get_dvfs_setting(self):
        cpu_freq = self.client.CPU_FREQS[self.cpu_index]
        gpu_freq = self.client.GPU_FREQS[self.gpu_index]
        mem_freq = self.client.MEM_FREQS[self.mem_index]
        return np.array([cpu_freq, gpu_freq, mem_freq])
    
    def get_temp_index(self, temp):
        temp_index = 0
        for i, (low_temp, high_temp) in enumerate(TEMPS):
            if low_temp <= temp and temp < high_temp:
                temp_index = i
        return temp_index
    
    def get_current_temp_index(self):
        temp = self.client.get_temperature()
        return self.get_temp_index(temp)
    
    def get_best_oracle_data(self, temp):
        temp_index = self.get_temp_index(temp)
        data = []
        for dvfs in self.oracle_data[temp_index].keys():
            measured = self.oracle_data[temp_index][dvfs]
            measured['dvfs'] = dvfs
            
            if ALPHA == 0 and measured['runtime'] < TARGET_DEADLINE:
                data.append(measured)
            elif ALPHA == 1 and measured['energy'] < ENERGY_BUDGET:
                data.append(measured)
        data = sorted(data, key=lambda d: d['runtime'] if ALPHA == 1 else d['energy']) 
        return data[0]
    
    def special_dvfs_adjustment(self):
        if ALPHA == 0:
            self.cpu_index = len(self.client.CPU_FREQS)-1
            self.gpu_index = len(self.client.GPU_FREQS)-1
            self.mem_index = len(self.client.MEM_FREQS)-1
        else:
            self.cpu_index = 2
            self.gpu_index = 2
            self.mem_index = 2

    def set_client_settings(self, action):
        cpu_freq_size = len(self.client.CPU_FREQS)
        gpu_freq_size = len(self.client.GPU_FREQS)
        mem_freq_size = len(self.client.MEM_FREQS)

        if action  == Action.INC_C:
            self.cpu_index = min(self.cpu_index + 1, cpu_freq_size - 1)
        elif action  == Action.DEC_C:
            self.cpu_index = max(0, self.cpu_index - 1)
        elif action  == Action.INC_G:
            self.gpu_index = min(self.gpu_index + 1, gpu_freq_size - 1)
        elif action  == Action.DEC_G:
            self.gpu_index = max(0, self.gpu_index - 1)
        elif action  == Action.INC_M:
            self.mem_index = min(self.mem_index + 1, mem_freq_size - 1)
        elif action  == Action.DEC_M:
            self.mem_index = max(0, self.mem_index - 1)
        elif action == Action.DO_NOTHING:
            pass
        elif action == None:
            self.special_dvfs_adjustment()
        self.client.set_dvfs_settings(self.get_dvfs_setting())
    
    def run_inference_and_collect_data(self):
        new_state = str(self.get_current_state())
        if new_state in self.measured_data.keys() and not self.evaluation:
            return self.measured_data[new_state]
        measured = self.client.run_inference()
        if self.calculate_reward(measured) < 0:
            self.num_of_violations_training[self.get_current_temp_index()] += 1
        if not new_state in self.oracle_data[self.get_current_temp_index()].keys() \
            or self.oracle_data[self.get_current_temp_index()][str(self.get_dvfs_setting())]['runtime'] < measured['runtime']:
            self.oracle_data[self.get_current_temp_index()][str(self.get_dvfs_setting())] = measured
        self.measured_data[new_state] = measured
        return measured
    
    def calculate_reward(self, measured_metrics):
        guard = 0 if self.evaluation else RUNTIME_GUARD
        energy, runtime = measured_metrics['energy'], measured_metrics['runtime']
        if ALPHA == 1 and energy * (1 + guard) > (ENERGY_BUDGET):
            reward = -1
        elif ALPHA == 0 and runtime * (1 + guard) > TARGET_DEADLINE:
            reward = -1
        else:
            reward = ALPHA * (TARGET_DEADLINE / runtime) + (1 - ALPHA) * (ENERGY_BUDGET / energy)
        self.last_freqs[self.client.temperature] = [str(self.get_dvfs_setting()), [self.cpu_index, self.gpu_index, self.mem_index]]
        if not self.client.temperature in self.best_freqs or reward > self.best_freqs[self.client.temperature][0]:
            self.best_freqs[self.client.temperature] = [reward, [self.cpu_index, self.gpu_index, self.mem_index]]
        return reward
            
    def set_temperature(self):
        if not self.evaluation:
            for index, temp in enumerate(LEARN_TEMPS[1:]):
                if self.round_num > (configs.TOTAL_EPSIODES * (index+1)) // len(LEARN_TEMPS):
                    self.client.temperature = temp
            if self.round_num < (configs.TOTAL_EPSIODES) // len(LEARN_TEMPS):
                self.client.temperature = LEARN_TEMPS[0]
        if not str(self.get_current_state) in self.measured_data.keys():
            self.set_client_settings(Action.DO_NOTHING)
            self.run_inference_and_collect_data()
    
    def quick_action(self, reward):
        if reward > 0 or self.evaluation:
            return
        if ALPHA == 0:
            logging.debug("quick action, increase freqs")
            self.set_client_settings(Action.INC_C)
            self.set_client_settings(Action.INC_C)
            self.set_client_settings(Action.INC_M)
            self.set_client_settings(Action.INC_M)
            self.set_client_settings(Action.INC_G)
            self.set_client_settings(Action.INC_G)
        else:
            logging.debug("quick action, decrease freqs")
            self.set_client_settings(Action.DEC_C)
            self.set_client_settings(Action.DEC_C)
            self.set_client_settings(Action.DEC_G)
            self.set_client_settings(Action.DEC_G)
            self.set_client_settings(Action.DEC_M)
            self.set_client_settings(Action.DEC_M)
        self.current_state = np.array([self.current_state[0], *self.get_dvfs_setting()])

    def capture_data(self, reward, measured_metrics, choosen_action):
        self.reward_history.append(reward)
        if reward < 0:
            for temp_index in range(self.get_current_temp_index(), len(TEMPS)):
                self.Q.prohibited_states[temp_index][str(self.get_dvfs_setting())] = True
        if self.evaluation == True:
            self.eval_data['runtime'].append(measured_metrics['runtime'])
            self.eval_data['energy'].append(measured_metrics['energy'])
            if reward < 0:
                self.toggled = True
                self.num_of_violations_evaluations += 1
            if self.toggled == True and reward > 0:
                self.toggled = False
    
    def train(self):
        self.start_time = time()
        logging.info("agent started learning optimization")

        while self.round_num < configs.TOTAL_EPSIODES:
            logging.debug(f"round number: {self.round_num}, epsilon: {self.epsilon}")
            self.round_num += 1
            
            self.current_state = self.get_current_state()
            self.set_temperature()
            logging.debug(f"current state: {self.current_state}")
            
            choosen_action = self.Q.get_action(self.current_state, self.epsilon[self.temp_index], self.get_current_state_index())
            logging.debug(f"new choosen action: {str(choosen_action)}")
            
            self.set_client_settings(choosen_action)
            logging.debug(f"new dvfs setting: {self.get_dvfs_setting()}")
            if choosen_action == None:
                continue
            measured_metrics = self.run_inference_and_collect_data()
            logging.debug(f"runtime: {measured_metrics['runtime']} - energy: {measured_metrics['energy']}")
            
            reward = self.calculate_reward(measured_metrics)
            logging.debug(f"reward: {reward}")
            
            new_state = np.array([self.current_state[0], *self.get_dvfs_setting()])
            new_state_qvalue = self.Q.update_table(self.current_state, choosen_action, new_state, reward)
            logging.debug(f"new state q value: {new_state_qvalue}")
            self.current_state = new_state
            
            self.capture_data(reward, measured_metrics, choosen_action)

            self.epsilon[self.temp_index] = max(self.epsilon[self.temp_index] * EPISLON, MIN_EPSILON)
            
        self.stop_time = time()
        logging.info(f"agent finished learning optimization after {self.round_num} steps, {self.stop_time - self.start_time}s")


def main():
    global ALPHA, ENERGY_BUDGET, TARGET_DEADLINE
    
    result = []
    oracle_res = []
    rewards = []
    for model in MODELS:
        for const in CONSTRAINTS:
            for alpha in ALPHAS:
                configs.MODEL_NAME = model["name"]
                ENERGY_BUDGET = model[const]["eb"]
                TARGET_DEADLINE = model[const]["td"]
                ALPHA = alpha
                
                print(model["name"], ALPHA, const, ENERGY_BUDGET, TARGET_DEADLINE)
                
                board_client = JetsonTX2()
                agent = Agent(board_client)
                agent.train()
                # print(agent.num_of_violations_training)
                # print(len(agent.measured_data))
                
                qos = "energy" if ALPHA==1 else "deadline"
                rewards.append({"data": agent.reward_history, "xlabel": f"{configs.MODEL_NAME.title()} QoS: {const}-{qos}"})    
                
                for eval_temp in EVALUATE_TEMPS:
                    agent.round_num = 0
                    agent.evaluation = True
                    agent.epsilon = [MIN_EPSILON for _ in range(len(TEMPS))]
                    agent.num_of_violations_evaluations = 0
                    agent.client.temperature = eval_temp
                    configs.TOTAL_EPSIODES = 500
                    agent.cpu_index, agent.gpu_index, agent.mem_index = agent.best_freqs[eval_temp][1]
                    agent.train()
                utils.plot_runtime(agent.eval_data, TARGET_DEADLINE)
                # utils.plot_energy(agent.eval_data)


                # print(agent.client.client.run_inference(str(agent.last_freqs[temp][0]), temp, False))
                # row_25 = agent.client.client.run_inference(str(agent.last_freqs[temp][0]), temp, False)
                # result.append({"model": model["name"], "temp": LOW_TEMP, "energy_budget": ENERGY_BUDGET, "deadline": TARGET_DEADLINE, "alpha": ALPHA,
                #     "freq": str(agent.last_freqs[LOW_TEMP][0]), "runtime": row_25["runtime"], "energy": row_25["energy"], "violations": agent.num_of_violations_evaluations})
                # oracle_data = agent.get_best_oracle_data(LOW_TEMP)
                # oracle_res.append({"model": model["name"], "temp": LOW_TEMP, "energy_budget": ENERGY_BUDGET, "deadline": TARGET_DEADLINE, "alpha": ALPHA,
                #     "freq": oracle_data['dvfs'], "runtime": oracle_data["runtime"], "energy": oracle_data["energy"]})
                                    

                # row_70 = agent.client.client.run_inference(str(agent.last_freqs[temp][0]), temp, False)
                # result.append({"model": model["name"], "temp": HIGH_TEMP, "energy_budget": ENERGY_BUDGET, "deadline": TARGET_DEADLINE, "alpha": ALPHA,
                #     "freq": str(agent.last_freqs[HIGH_TEMP][0]), "runtime": row_70["runtime"], "energy": row_70["energy"], "violations": agent.num_of_violations_evaluations})
                # oracle_data = agent.get_best_oracle_data(HIGH_TEMP)
                # oracle_res.append({"model": model["name"], "temp": HIGH_TEMP, "energy_budget": ENERGY_BUDGET, "deadline": TARGET_DEADLINE, "alpha": ALPHA,
                #     "freq": oracle_data['dvfs'], "runtime": oracle_data["runtime"], "energy": oracle_data["energy"]})

                # with open('result.csv', 'a', newline='') as output_file:
                #     dict_writer = csv.DictWriter(output_file, result[0].keys())
                #     dict_writer.writeheader()
                #     dict_writer.writerows(result)
                
                # with open('oracle.csv', 'a', newline='') as output_file:
                #     dict_writer = csv.DictWriter(output_file, oracle_res[0].keys())
                #     dict_writer.writeheader()
                #     dict_writer.writerows(oracle_res)
                
    # plot_reward_history(rewards)
    
    # plt.xlabel("inference num")
    # plt.ylabel("energy consumption (mJ)")
    # plt.plot(agent.eval_data['energy'], color='grey', marker='.', linestyle='None')
    # plt.show()
    


if __name__ == "__main__":
    main()