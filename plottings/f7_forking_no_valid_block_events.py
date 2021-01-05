
# figure 7 (args must be the same as f6)

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

for var in vars_names:
	vars()[f'{var}_forking'] = []
	vars()[f'{var}_no_valid_block'] = []

# get forking and no valid block events indication
for log_folder in vars_names:
	file = open(f'{vars()[log_folder]}/forking_and_no_valid_block_log.txt',"r")
	lines_list = file.read().split("\n")
	for line in lines_list:
		if 'Forking' in line:
			vars()[f'{log_folder}_forking'].append(int(line.split(' ')[-1]))
		if 'No valid' in line:
			vars()[f'{log_folder}_no_valid_block'].append(int(line.split(' ')[-1]))

draw_vars_forking = [f"{var}_forking" for var in vars_names]
draw_vars_no_valid_block = [f"{var}_no_valid_block" for var in vars_names]

# from bottom to top
y_axis_labels = ["POW_D2", "POW_D1", "POS_r1", "POS_r2", "POS_r3"][::-1]
plt.yticks(range(len(y_axis_labels)), y_axis_labels)
# labels = ['VBFL_PoW_3/20_vh0.08_D_2', 'VBFL_PoW_3/20_vh0.08_D_1', 'VBFL_PoS_3/20_vh0.08_run1', 'VBFL_PoS_3/20_vh0.08_run2', 'VBFL_PoS_3/20_vh0.08_run3'] # labels have order in legend
colors = ['orange', 'green', 'blue', 'magenta', 'red']

# draw forking and no valid block
for draw_var_iter in range(len(draw_vars_forking)):
	draw_var = draw_vars_forking[draw_var_iter]
	for draw_point in vars()[draw_var]:
		plt.plot(draw_point, len(draw_vars_forking) - draw_var_iter - 1, 'o', color=colors[draw_var_iter], mfc='none')

for draw_var_iter in range(len(draw_vars_no_valid_block)):
	draw_var = draw_vars_no_valid_block[draw_var_iter]
	for draw_point in vars()[draw_var]:
		plt.plot(draw_point, len(draw_vars_forking) - draw_var_iter - 1, 'x', color=colors[draw_var_iter])

x1,x2,y1,y2 = plt.axis()
plt.axis((0,100,y1,y2))

# plt.legend(loc='b', fontsize='small')
plt.xlabel('Communication Round')
plt.ylabel('Setup')
plt.title('Forking and No_Valid_Block Event Indicator')

plt.show()
