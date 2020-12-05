# shaded graph https://riptutorial.com/matplotlib/example/11221/shaded-plots

# figure 5

import matplotlib.pyplot as plt
from os import listdir
import sys
import os.path
from os import path
import numpy as np

# 20 devices
log_folder_VFL_0 = sys.argv[1]
log_folder_PoS_0_vh_1 = sys.argv[2]
log_folder_PoS_3_vh_008 = sys.argv[3]
log_folder_PoS_3_vh_008_mv = sys.argv[4]
log_folder_VFL_3 = sys.argv[5]

draw_comm_rounds = 100
total_runs = 3

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

plt.figure(dpi=250)
	
VFL_0_accuracies = {}
VFL_3_accuracies = {}
PoS_0_vh_1_accuracies = {}
PoS_3_vh_008_accuracies = {}
PoS_3_vh_008_mv_accuracies = {}

VFL_log_folders = ["log_folder_VFL_0", "log_folder_VFL_3"]
PoS_log_folders = ["log_folder_PoS_0_vh_1", "log_folder_PoS_3_vh_008", "log_folder_PoS_3_vh_008_mv"]
runs_folders = [f'run{i}' for i in range(1, total_runs+1)]

for log_folder in VFL_log_folders:
	for run in runs_folders:
		for round_iter in all_rounds:
			if run not in vars()[f'{log_folder[11:]}_accuracies'].keys():
				vars()[f'{log_folder[11:]}_accuracies'][run] = []
			accuracy_file = f'{vars()[log_folder]}/{run}/{round_iter}.txt'
			file = open(accuracy_file,"r")
			lines_list = file.read().split("\n")
			for line in lines_list:
				if line.startswith('client_1'):
					vars()[f'{log_folder[11:]}_accuracies'][run].append(float(line.split(' ')[-1]))
					break

for log_folder in PoS_log_folders:
	for run in runs_folders:
		for round_iter in all_rounds:
			if run not in vars()[f'{log_folder[11:]}_accuracies'].keys():
				vars()[f'{log_folder[11:]}_accuracies'][run] = []
			accuracy_file = f'{vars()[log_folder]}/{run}/{round_iter}/accuracy_{round_iter}.txt'
			if not path.exists(accuracy_file):
				vars()[f'{log_folder[11:]}_accuracies'][run].append(vars()[f'{log_folder[11:]}_accuracies'][run][-1])
				continue
			file = open(accuracy_file,"r")
			lines_list = file.read().split("\n")
			all_accuracies = []
			for line in lines_list:
				if line.startswith('device'):
					all_accuracies.append(float(line.split(' ')[-1]))
			vars()[f'{log_folder[11:]}_accuracies'][run].append(max(all_accuracies))

# manual zip
VFL_0_accuracies_zip = zip(VFL_0_accuracies['run1'], VFL_0_accuracies['run2'], VFL_0_accuracies['run3'])
VFL_3_accuracies_zip = zip(VFL_3_accuracies['run1'], VFL_3_accuracies['run2'], VFL_3_accuracies['run3'])
PoS_0_vh_1_accuracies_zip = zip(PoS_0_vh_1_accuracies['run1'], PoS_0_vh_1_accuracies['run2'], PoS_0_vh_1_accuracies['run3'])
PoS_3_vh_008_accuracies_zip = zip(PoS_3_vh_008_accuracies['run1'], PoS_3_vh_008_accuracies['run2'], PoS_3_vh_008_accuracies['run3'])
PoS_3_vh_008_mv_accuracies_zip = zip(PoS_3_vh_008_mv_accuracies['run1'], PoS_3_vh_008_mv_accuracies['run2'], PoS_3_vh_008_mv_accuracies['run3'])

# notice the order if want top to bottom label legend with specified color
draw_vars = ["VFL_0", "PoS_0_vh_1", "PoS_3_vh_008", "PoS_3_vh_008_mv", "VFL_3"]
labels = ['VFL_0/20', 'VBFL_PoS_0/20_vh1.00', 'VBFL_PoS_3/20_vh0.08', 'VBFL_PoS_3/20_vh0.08_mv', 'VFL_3/20'] # labels have order in legend
colors = ['orange', 'green', 'blue', 'magenta', 'red']
# draw VFL
for draw_var in draw_vars:
	vars()[f'{draw_var}_draw'] = {}
	vars()[f'{draw_var}_draw']['up'] = []
	vars()[f'{draw_var}_draw']['mean'] = []
	vars()[f'{draw_var}_draw']['down'] = []
	for round_values in list(vars()[f'{draw_var}_accuracies_zip']):
		mean = np.mean(round_values)
		std = np.std(round_values)
		vars()[f'{draw_var}_draw']['up'].append(mean+std)
		vars()[f'{draw_var}_draw']['mean'].append(mean)
		vars()[f'{draw_var}_draw']['down'].append(mean-std)


for draw_var_iter in range(len(draw_vars)):
	draw_var = draw_vars[draw_var_iter]
	plt.fill_between(range(1,draw_comm_rounds+1), vars()[f'{draw_var}_draw']['down'], vars()[f'{draw_var}_draw']['up'],
		facecolor=colors[draw_var_iter], # The fill color
		color=colors[draw_var_iter],       # The outline color
		alpha=0.2)          # Transparency of the fill
	plt.plot(range(1,draw_comm_rounds+1), vars()[f'{draw_var}_draw']['mean'], color=colors[draw_var_iter], label=labels[draw_var_iter])

plt.legend(loc='b')
plt.xlabel('Communication Round')
plt.ylabel('Accuracy')
plt.title('Global Model Accuracy Comparisons')

plt.show()
