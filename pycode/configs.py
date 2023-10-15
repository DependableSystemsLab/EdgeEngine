
import logging

RUNTIME_GUARD = 0/100

EPISLON = 0.9997
MIN_EPSILON = 0
TOTAL_EPSIODES = 5

LOW_TEMP = 25
HIGH_TEMP = 70

MU = 0.1
GAMMA = 0.9

LOGGING_LEVEL = logging.INFO

TEMPS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]

LEARN_TEMPS = [25, 35, 45, 55, 65]

EVALUATE_TEMPS = [25, 35, 45, 55, 65]

MODELS = [\
              {"name": "mobilenetv2", "loose": {"eb": 105, "td": 19}, "tight": {"eb": 100, "td": 14}},
              {"name": "alexnet", "loose": {"eb": 145, "td": 19}, "tight": {"eb": 130, "td": 15}},
              {"name": "resnet", "loose": {"eb": 170, "td": 20}, "tight": {"eb": 150, "td": 15}},
              {"name": "inception3", "loose": {"eb": 590, "td": 76}, "tight": {"eb": 505, "td": 60}},
              {"name": "googlenet", "loose": {"eb": 200, "td": 33}, "tight": {"eb": 160, "td": 18}},
              {"name": "squeezenet", "loose": {"eb": 140, "td": 21}, "tight": {"eb": 120, "td": 13}},
              {"name": "shufflenetv2", "loose": {"eb": 100, "td": 24}, "tight": {"eb": 80, "td": 16}}
            ]
CONSTRAINTS = ["tight"]
ALPHAS = [0]