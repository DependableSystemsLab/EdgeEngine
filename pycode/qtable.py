import numpy as np

from enum import Enum
from configs import MU, GAMMA, TEMPS
from board import IBoard

class Action(Enum):
    INC_C, INC_G, INC_M, DEC_C, DEC_G, DEC_M, DO_NOTHING = range(7)

class QTable():
    def __init__(self, client: IBoard, states, actions):
        self.client = client
        self.mu = MU
        self.gamma = GAMMA
        self.states = states
        self.actions = actions
        self.init_table()
        self.init_prohibited_states()
    
    def init_table(self):
        self.table = {}
        for state in self.states:
            self.table[str(state)] = np.zeros(len(self.actions))
    
    def init_prohibited_states(self):
        self.prohibited_states = []
        for _ in range(len(TEMPS)):
            self.prohibited_states.append({})
    
    def get_next_state(self, current_state_index, action):
        cpu_index, gpu_index, mem_index = current_state_index[1:]
        cpu_freq_size = len(self.client.CPU_FREQS)
        gpu_freq_size = len(self.client.GPU_FREQS)
        mem_freq_size = len(self.client.MEM_FREQS)

        if action  == Action.INC_C:
            cpu_index = min(cpu_index + 1, cpu_freq_size - 1)
        elif action  == Action.DEC_C:
            cpu_index = max(0, cpu_index - 1)
        elif action  == Action.INC_G:
            gpu_index = min(gpu_index + 1, gpu_freq_size - 1)
        elif action  == Action.DEC_G:
            gpu_index = max(0, gpu_index - 1)
        elif action  == Action.INC_M:
            mem_index = min(mem_index + 1, mem_freq_size - 1)
        elif action  == Action.DEC_M:
            mem_index = max(0, mem_index - 1)
        elif action == Action.DO_NOTHING:
            pass
        next_state_index = [current_state_index[0], cpu_index, gpu_index, mem_index]
        err = 1 if action != Action.DO_NOTHING and next_state_index == current_state_index else 0
        return next_state_index, err
    
    def available_actions(self, current_state_index: np.array):
        available_actions = []
        for action_index, action in enumerate(Action):
            next_state_index, err = self.get_next_state(current_state_index, action)
            next_state = np.array([self.client.CPU_FREQS[next_state_index[1]],
                                   self.client.GPU_FREQS[next_state_index[2]],
                                   self.client.MEM_FREQS[next_state_index[3]],
                                   ])
            if err == 1 or str(next_state) in self.prohibited_states[next_state_index[0]]:
                continue
            available_actions.append(action_index)
    
        return available_actions
            
    def get_action(self, current_state: np.array, epsilon: float, current_state_index):
        rand_num = np.random.rand()
        if rand_num < epsilon:
            available_actions = self.available_actions(current_state_index)
            if len(available_actions) == 0:
                return None
            action = np.random.choice(available_actions)
        else:
            action = self.get_largest_q_action(str(current_state))
        return Action(action)
    
    def get_largest_q_action(self, state):
        return np.argmax(self.table[state])
           
    def update_table(self, state, action, new_state, reward):
        action = action.value
        state, new_state = str(state), str(new_state)
        
        new_state_action = self.get_largest_q_action(new_state)
        qsa = self.table[state][action]

        qnewsa = self.table[new_state][new_state_action]
        self.table[state][action] = qsa + self.gamma * (reward + self.mu * (qnewsa) - qsa)
        return self.table[state][action]
