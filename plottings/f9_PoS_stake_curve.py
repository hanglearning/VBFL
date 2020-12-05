
# figure 9 (PoS args must be the same as f6, f7 and f8)

import matplotlib.pyplot as plt
from os import listdir
import sys
import os.path
from os import path
import numpy as np
import matplotlib.lines as mlines

# 20 devices
log_folder_PoS_0_1 = sys.argv[1]
log_folder_PoS_0_2 = sys.argv[2]
log_folder_PoS_0_3 = sys.argv[3]
log_folder_PoW_1_1 = sys.argv[4]
log_folder_PoW_2_1 = sys.argv[5]

draw_comm_rounds = 100

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

vars_names = ["log_folder_PoW_2_1","log_folder_PoW_1_1","log_folder_PoS_0_1","log_folder_PoS_0_2","log_folder_PoS_0_3"]

for var in vars_names:
	vars()[f'{var}_stakes'] = {}


# get all devices stakes
for log_folder in vars_names:
	for round_iter in all_rounds:
		stake_file = f"{vars()[log_folder]}/{round_iter}/stake_{round_iter}.txt"
		file = open(stake_file,"r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_'):
				device_idx = line.split(":")[0].split(" ")[0]
				stake = int(line.split(":")[-1])
				b_or_m = line.split(":")[0].split(" ")[-1]
				whole_device_idx = f'{device_idx} {b_or_m}'
				if not whole_device_idx in vars()[f'{log_folder}_stakes'].keys():
					vars()[f'{log_folder}_stakes'][whole_device_idx] = [stake]
				else:
					vars()[f'{log_folder}_stakes'][whole_device_idx].append(stake)


log_vars = ["POW_D2", "POW_D1", "POS_r1", "POS_r2", "POS_r3"]
# axs_iters_y = [0,1,2,3,4]
fig, axs = plt.subplots(1, 5, sharey=True)
plt.setp(axs, ylim=(0, 1000))
axs[0].set_ylabel('Total Stakes')
import matplotlib.ticker as mticker    
for log_var_iter in range(len(vars_names)):
	log_var = vars_names[log_var_iter]
	# set label
	axs[log_var_iter].set_xlabel('Comm Round')
	axs[log_var_iter].set_title(f'Stake Accum {log_vars[log_var_iter]}')
	axs[log_var_iter].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fk'))
	# axs[0, axs_iters_y[log_var_iter]].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fk'))
	for device_idx, stakes in vars()[f'{log_var}_stakes'].items():
		if device_idx.split(' ')[-1] == 'M':
			axs[log_var_iter].plot(range(draw_comm_rounds), [_/1000 for _ in stakes], color='red')
		else:
			axs[log_var_iter].plot(range(draw_comm_rounds), [_/1000 for _ in stakes], color='green')

	green_line = mlines.Line2D([], [], color='green', label="legitimate")
	red_line = mlines.Line2D([], [], color='red', label="malicious")

	axs[log_var_iter].legend(handles=[green_line,red_line], loc='best', prop={'size': 10})

plt.figure(dpi=500, figsize=(12,1))

plt.show()

