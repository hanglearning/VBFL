# shaded graph https://riptutorial.com/matplotlib/example/11221/shaded-plots

# figure 6

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
	vars()[f'{var}_SRC'] = []
	vars()[f'{var}_BG'] = []

# get time
for log_folder in vars_names:
	for round_iter in all_rounds:
		accuracy_file = f'{vars()[log_folder]}/{round_iter}/accuracy_{round_iter}.txt'
		file = open(accuracy_file,"r")
		lines_list = file.read().split("\n")
		for line in lines_list:
			if 'slowest' in line:
				try:
					this_time = round(float(line.split(' ')[-1]), 2)
				except:
					this_time = 0
				# accumulated SRC
				if round_iter == 'comm_1':
					vars()[f'{log_folder}_SRC'].append(this_time)
				else:
					vars()[f'{log_folder}_SRC'].append(vars()[f'{log_folder}_SRC'][-1] + this_time)
				# individual SRC
				# vars()[f'{log_folder}_SRC'].append(this_time)
			if 'block_gen' in line:
				try:
					vars()[f'{log_folder}_BG'].append(round(float(line.split(':')[-1]), 2))
				except:
					vars()[f'{log_folder}_BG'].append(-1)

draw_vars_SRC = [f"{var}_SRC" for var in vars_names]
draw_vars_BG = [f"{var}_BG" for var in vars_names]


labels = ['VBFL_PoW_3/20_vh0.08_D_2', 'VBFL_PoW_3/20_vh0.08_D_1', 'VBFL_PoS_3/20_vh0.08_run1', 'VBFL_PoS_3/20_vh0.08_run2', 'VBFL_PoS_3/20_vh0.08_run3'] # labels have order in legend
colors = ['orange', 'green', 'blue', 'magenta', 'red']

# draw SRC
for draw_var_iter in range(len(draw_vars_SRC)):
	draw_var = draw_vars_SRC[draw_var_iter]
	plt.plot(range(1,draw_comm_rounds+1), vars()[draw_var], color=colors[draw_var_iter], label=labels[draw_var_iter])

# draw gen_time
# skip for now

x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,10000))

plt.legend(loc='b', fontsize='small')
plt.xlabel('Communication Round')
plt.ylabel('Seconds')
plt.title('Comparison of the SRC-Time between PoW and VBFL-PoS')

plt.show()
