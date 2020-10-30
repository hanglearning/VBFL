# figure 4 and 5

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
for sub_log_folder_name_iter in range(len(all_rounds_log_folders[:draw_comm_rounds])):

	sub_log_folder_name = all_rounds_log_folders[sub_log_folder_name_iter]
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	
	# record next round local accuracy
	sub_log_folder_name = all_rounds_log_folders[sub_log_folder_name_iter+1]
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	file = open(f"{sub_log_folder_path}/{draw_device_idx_b}_local_{sub_log_folder_name}.txt","r")
	file_whole_text = file.read()
	local_accuracies_e5_b = round(float(file_whole_text.split("\n")[4].split(' ')[-1]), 3)

	draw_accuracies_b.append(global_accuracy - local_accuracies_e5_b)

	

# get local m learning curve
for sub_log_folder_name_iter in range(len(all_rounds_log_folders[:draw_comm_rounds])):

	sub_log_folder_name = all_rounds_log_folders[sub_log_folder_name_iter]
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	
	# record next round local accuracy
	sub_log_folder_name = all_rounds_log_folders[sub_log_folder_name_iter+1]
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	file = open(f"{sub_log_folder_path}/{draw_device_idx_m}_local_{sub_log_folder_name}.txt","r")
	file_whole_text = file.read()
	local_accuracies_e5_m = round(float(file_whole_text.split("\n")[5].split(' ')[-1]), 3)

	draw_accuracies_m.append(global_accuracy - local_accuracies_e5_m)

plt.figure(dpi=250)

# x_axis_labels = []
# x_axis_ls = []
# x_axis_gs = []
# for i in range(draw_comm_rounds):
# 	# x_axis_labels.append(f'le1')
# 	x_axis_labels.append(f'le5')
# 	x_axis_labels.append(f'g{i+1}')
# 	# x_axis_ls.append(2*i)
# 	x_axis_gs.append(2*i+1)
# draw graphs over all available comm rounds
# plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
# draw the whole learning curve
# plt.plot(range(len(x_axis_labels)), draw_accuracies_b, label=f'b overall learning curve')
# plt.plot(range(len(x_axis_labels)), draw_accuracies_m, label=f'm overall learning curve')

acc_diff = []
for acc_iter in range(len(draw_accuracies_b)):
	acc_b = draw_accuracies_b[acc_iter]
	acc_m = draw_accuracies_m[acc_iter]
	acc_diff.append(acc_m-acc_b)

x_axis_labels = []
for i in range(draw_comm_rounds-1):
	x_axis_labels.append(f'{i+2}')
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
plt.plot(range(len(x_axis_labels)), draw_accuracies_m[:len(x_axis_labels)], label=r"$drop_j^{wm}$", color='red')
plt.plot(range(len(x_axis_labels)), draw_accuracies_b[:len(x_axis_labels)], label=r"$drop_j^{wl}$")
plt.plot(range(len(x_axis_labels)), acc_diff[:len(x_axis_labels)], label=r'$drop_j^{wm}$ - $drop_j^{wl}$', color='green')

# connect local model learning curve
#plt.plot(x_axis_ls, local_accuracies, label=f'{draw_device_idx_b} local learning curve')
# connect global model learning curve
#plt.plot(x_axis_gs, global_accuracies, label=f'global learning curve', linestyle="dashed",)


plt.legend(loc=2, prop={'size': 9})
plt.title("Comparison of Accuracy Drops")
plt.xlabel("Communication Rounds")
plt.ylabel("Extent of Accuracy Drops")

# plt.show()

plt.clf()

# figure 5

# draw shadow
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
plt.plot(range(len(x_axis_labels)), acc_diff[:len(x_axis_labels)], label=r'$drop_j^{wm}$ - $drop_j^{wl}$', color='green')

all_1 = []
for i in range(len(x_axis_labels)):
	all_1.append(1.0)
plt.axhline(y=0.08, linestyle='dashed', color='orange')
plt.text(18, 0.08, r'$vh_{j}^{v}$=0.08', fontsize=15, va='bottom', ha='right')
plt.text(9, 0.6, 'Malicious Workers Reside in \nThis Grey Area', fontsize=16, va='center', ha='center')
plt.fill_between(range(len(x_axis_labels)), all_1, acc_diff[:len(x_axis_labels)], color='#d3d3d3')

plt.legend(loc=2, prop={'size': 12})
plt.title("Malicious Workers Identity Area")
plt.xlabel("Communication Rounds")
plt.ylabel("Differences between two Drops")

plt.show()