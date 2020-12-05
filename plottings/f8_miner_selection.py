
# figure 8 (args must be the same as f6 and f7)

import matplotlib.pyplot as plt
from os import listdir
import sys
import os.path
from os import path
import numpy as np

# 20 devices
log_folder_PoS_0_1 = sys.argv[1]
log_folder_PoS_0_2 = sys.argv[2]
log_folder_PoS_0_3 = sys.argv[3]
log_folder_PoW_1_1 = sys.argv[4]
log_folder_PoW_2_1 = sys.argv[5]

draw_comm_rounds = 100

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

plt.figure(dpi=250, figsize=(6,2))

vars_names = ["log_folder_PoW_2_1","log_folder_PoW_1_1","log_folder_PoS_0_1","log_folder_PoS_0_2","log_folder_PoS_0_3"]

PoS_vars_names = ["log_folder_PoS_0_1","log_folder_PoS_0_2","log_folder_PoS_0_3"]

for var in vars_names:
	vars()[f'{var}_malicious'] = []


# get PoS miner maliciousness
for log_folder in PoS_vars_names:
	for round_iter in all_rounds:
		stake_file = f'{vars()[log_folder]}/{round_iter}/stake_{round_iter}.txt'
		file = open(stake_file,"r")
		lines_list = file.read().split("\n")
		for line in lines_list:
			if 'PoS_block_mined_by' in line:
				maliciousness = line.split(' ')[-1]
				if maliciousness == 'M':
					vars()[f'{log_folder}_malicious'].append(int(round_iter.split('_')[-1]))

# get PoW 2 miner maliciousness
log_folder_PoW_2_1_malicious = [2,5,6,16,18,26,37,42,43,45]

# get PoW 1 miner maliciousness
for round_iter in all_rounds:
	accuracy_file = f'{log_folder_PoW_1_1}/{round_iter}/accuracy_{round_iter}.txt'
	file = open(accuracy_file,"r")
	lines_list = file.read().split("\n")
	for line in lines_list:
		if 'block_mined_by' in line:
			maliciousness = line.split(' ')[-1]
			if maliciousness == 'M':
				log_folder_PoW_1_1_malicious.append(int(round_iter.split('_')[-1]))

draw_vars_malicious = [f"{var}_malicious" for var in vars_names]

# from bottom to top
y_axis_labels = ["POW_D2", "POW_D1", "POS_r1", "POS_r2", "POS_r3"][::-1]
plt.yticks(range(len(y_axis_labels)), y_axis_labels)
colors = ['orange', 'green', 'blue', 'magenta', 'red']

# draw forking and no valid block
for draw_var_iter in range(len(draw_vars_malicious)):
	draw_var = draw_vars_malicious[draw_var_iter]
	for draw_point in vars()[draw_var]:
		plt.plot(draw_point, len(draw_vars_malicious) - draw_var_iter - 1, 'o', color=colors[draw_var_iter])
# no malicious for 3 PoS run
plt.scatter(0, 0, color='white')
plt.scatter(0, 1, color='white')
plt.scatter(0, 2, color='white')

x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,y1,y2))

# plt.legend(loc='b', fontsize='small')
plt.xlabel('Communication Round')
plt.ylabel('Setup')
plt.title('Events of Legitimate Block Mined by Malicious Device')

plt.show()
