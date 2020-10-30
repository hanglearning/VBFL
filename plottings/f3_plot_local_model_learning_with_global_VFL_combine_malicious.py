# figure 3

import matplotlib.pyplot as plt
import os
import sys
from adjustText import adjust_text


log_folder = sys.argv[1]

all_rounds_log_folders = sorted([f for f in os.listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_folders)

draw_comm_rounds = 20
draw_device_idx_b = 'client_1'
draw_device_idx_m = 'client_8'

local_accuracies_e1_b = []
local_accuracies_e5_b = []
draw_accuracies_b = []
local_accuracies_e1_m = []
local_accuracies_e5_m = []
draw_accuracies_m = []

global_accuracies = []


# get local b and global model's learning curve
for sub_log_folder_name in all_rounds_log_folders[:draw_comm_rounds]:
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	# record local accuracy
	file = open(f"{sub_log_folder_path}/{draw_device_idx_b}_local_{sub_log_folder_name}.txt","r")
	file_whole_text = file.read()
	local_accuracies_e1_b = round(float(file_whole_text.split("\n")[0].split(' ')[-1]), 3)
	local_accuracies_e5_b = round(float(file_whole_text.split("\n")[4].split(' ')[-1]), 3)
	# local_accuracies.append(local_accuracy)
	#draw_accuracies_b.append(local_accuracies_e1_b)
	draw_accuracies_b.append(local_accuracies_e5_b)
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	global_accuracies.append(global_accuracy)
	draw_accuracies_b.append(global_accuracy)

# get local m learning curve
for sub_log_folder_name in all_rounds_log_folders[:draw_comm_rounds]:
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	# record local accuracy
	file = open(f"{sub_log_folder_path}/{draw_device_idx_m}_local_{sub_log_folder_name}.txt","r")
	file_whole_text = file.read()
	local_accuracies_e1_m = round(float(file_whole_text.split("\n")[0].split(' ')[-1]), 3)
	local_accuracies_e5_m = round(float(file_whole_text.split("\n")[5].split(' ')[-1]), 3)
	# local_accuracies.append(local_accuracy)
	#draw_accuracies_m.append(local_accuracies_e1_m)
	draw_accuracies_m.append(local_accuracies_e5_m)
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	#global_accuracies.append(global_accuracy)
	draw_accuracies_m.append(global_accuracy)

plt.figure(dpi=250)

x_axis_labels = []
x_axis_ls = []
x_axis_gs = []
for i in range(draw_comm_rounds):
	# x_axis_labels.append(f'le1')
	x_axis_labels.append(f'le5')
	x_axis_labels.append(f'g{i+1}')
	# x_axis_ls.append(2*i)
	x_axis_gs.append(2*i+1)
# draw graphs over all available comm rounds
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
# draw the whole learning curve
plt.plot(range(len(x_axis_labels)), draw_accuracies_b, label=f'legitimate learning curve')
plt.plot(range(len(x_axis_labels)), draw_accuracies_m, label=f'malicious learning curve',color='red')


# connect local model learning curve
#plt.plot(x_axis_ls, local_accuracies, label=f'{draw_device_idx_b} local learning curve')
# connect global model learning curve
plt.plot(x_axis_gs, global_accuracies, label=f'global learning curve', linestyle="dashed",color='orange')


plt.legend(loc=2)
plt.title("Learning Curve Comparison")
plt.xlabel("Checkpoint")
plt.ylabel("Accuracies of Model At Checkpoint")

plt.show()
print()