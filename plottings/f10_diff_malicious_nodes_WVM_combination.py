# shaded graph https://riptutorial.com/matplotlib/example/11221/shaded-plots

# figure 10 

import matplotlib.pyplot as plt
from os import listdir
import sys
import os.path
from os import path
import numpy as np

# 20 devices
wvm_12_5_3 = sys.argv[1]
wvm_9_8_3 = sys.argv[2]

draw_comm_rounds = 50

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

nm_folders = [f'nm{_}' for _ in [3,4,6,8,10,12,14,16]]

wvm_log_vars = ["wvm_12_5_3", "wvm_9_8_3"]
for wvm_log_var in wvm_log_vars:
	vars()[f'{wvm_log_var}_accuracies'] = {}

for log_folder in wvm_log_vars:
	for nm in nm_folders:
		if nm not in vars()[f'{log_folder}_accuracies'].keys():
			vars()[f'{log_folder}_accuracies'][nm] = []
		for round_iter in all_rounds:
			try:
				accuracy_file = f'{vars()[log_folder]}/{nm}/{round_iter}/accuracy_{round_iter}.txt'
				file = open(accuracy_file,"r")
			except:
				vars()[f'{log_folder}_accuracies'][nm].append(vars()[f'{log_folder}_accuracies'][nm][-1])
				continue
			lines_list = file.read().split("\n")
			all_accuracies = []
			for line in lines_list:
				if line.startswith('device'):
					all_accuracies.append(float(line.split(' ')[-1]))
			vars()[f'{log_folder}_accuracies'][nm].append(max(all_accuracies))

labels = ['mal. 3/20', 'mal. 4/20', 'mal. 6/20', 'mal. 8/20', 'mal. 10/20', 'mal. 12/20', 'mal. 14/20', 'mal. 16/20'] # labels have order in legend

rain_bow_colors = ['#ff0000','#ffa500','#d0d57c','#008000','#0000ff','cyan','#ee82ee','black']
titles = [r'Learning curve of 12 $\mathcal{W}$, 5 $\mathcal{V}$, 3 $\mathcal{M}$', r'Learning curve of 8 $\mathcal{W}$, 9 $\mathcal{V}$, 3 $\mathcal{M}$']

for log_folder_iter in range(len(wvm_log_vars)):
	plt.figure(dpi=250, figsize=(6,3))
	log_folder = wvm_log_vars[log_folder_iter]
	for nm_iter in range(len(nm_folders)):
		nm = nm_folders[nm_iter]
		plt.plot(range(1,draw_comm_rounds+1), vars()[f'{log_folder}_accuracies'][nm], color=rain_bow_colors[nm_iter], label=labels[nm_iter])
	plt.legend(loc='b', fontsize='small')
	plt.xlabel('Communication Round')
	plt.ylabel('Accuracy')
	plt.title(titles[log_folder_iter])
	plt.show()
	plt.clf()
