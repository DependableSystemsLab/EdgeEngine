# EdgeEngine

This repository contains the code and data used in the research paper titled "EdgeEngine: A Thermal-Aware Optimization Framework for Edge Inference." This study aims to introduce a thermal-aware optimization framework for heterogenous edge devices to find the near-optimal frequency configurations (CPU, GPU, Memory controller) at different ambient temperatures. EdgeEngine uses reinforcement learning for the purpose of the adaptation.

![alt text](https://imageupload.io/ib/6N9fw2HS70lB2kX_1697415457.png)

## Code/Data
**agenet.pyy**: the main codes for the EdgeEngine RL agent.

**qtable.py**: codes related to the Q-Table and its updates.

**board.py**: codes and interfaces for gathering data from the embedded edge platform and frequency configuration adjustment.

**data**: Data regarding the runtime and power consumption of the seven different ML inference applications at 25 and 70ºC on NVIDIA Jetson TX2. The codes for analyzing the data are also placed in the directory.

## Citation

Please cite EdgeEngine's research paper if its code/data helps your research:

    inproceedings{edgeengine23sec,
    title={EdgeEngine: A Thermal-Aware Optimization Framework for Edge Inference},
    author={Ahmadi, Amirhossein and Abdelhafez, Hazem A and Pattabiraman, Karthik and Ripeanu, Matei},
    booktitle={2023 IEEE/ACM 8th Symposium on Edge Computing (SEC)},
    year={2023},
    organization={IEEE}
    }
