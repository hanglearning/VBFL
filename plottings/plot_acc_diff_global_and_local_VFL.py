import matplotlib.pyplot as plt
import os
import sys
from adjustText import adjust_text


log_folder = sys.argv[1]

all_rounds_log_folders = sorted([f for f in os.listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_folders)

draw_comm_rounds = 20
draw_device_idx = 'client_1'

local_accuracies_e1 = []
local_accuracies_e5_lst = []
global_accuracies = []
draw_accuracies = []

# get local and global model's learning curve
for sub_log_folder_name in all_rounds_log_folders[:draw_comm_rounds]:
	sub_log_folder_path = f"{log_folder}/{sub_log_folder_name}"
	# record local accuracy
	file = open(f"{sub_log_folder_path}/{draw_device_idx}_local_{sub_log_folder_name}.txt","r")
	file_whole_text = file.read()
	local_accuracies_e1 = round(float(file_whole_text.split("\n")[0].split(' ')[-1]), 3)
	local_accuracies_e5 = round(float(file_whole_text.split("\n")[4].split(' ')[-1]), 3)
	# local_accuracies.append(local_accuracy)
	local_accuracies_e5_lst.append(local_accuracies_e5)
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	global_accuracies.append(global_accuracy)
	draw_accuracies.append(global_accuracy)

le5_minus_g = []
for acc_iter in range(len(local_accuracies_e5_lst)):
	try:
		acc_e5 = local_accuracies_e5_lst[acc_iter+1]
		le5_minus_g.append(round(abs(acc_e5-global_accuracies[acc_iter]),3))
	except:
		pass

x_axis_labels = []
for i in range(len(le5_minus_g)):
	x_axis_labels.append(f'R_{i+2}')

# draw graphs over all available comm rounds
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
# draw the whole learning curve
plt.plot(range(len(x_axis_labels)), le5_minus_g, label=f'{draw_device_idx} minus')
# dashed lines
# plt.vlines(range(len(x_axis_labels)), 0, le5_minus_g, linestyle="dashed", color='c')

# connect local model learning curve
#plt.plot(x_axis_ls, local_accuracies, label=f'{draw_device_idx} local learning curve')
# connect global model learning curve
#plt.plot(x_axis_gs, global_accuracies, label=f'{draw_device_idx} global learning curve')

# annotate
annotations = []
for x, y, acc in zip(range(len(x_axis_labels)), le5_minus_g, le5_minus_g):
	annotations.append(plt.text(x,y,acc))


adjust_text(annotations, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

# for draw_acc_iter in range(len(x_axis_labels)):
# 	draw_acc = le5_minus_g[draw_acc_iter]
# 	plt.annotate(draw_acc, xy=(draw_acc_iter, draw_acc), size=13)

plt.legend(loc='best')
plt.title("A Random Device's Continuous Learning Curve in Vanilla FL Using FedAvg")
plt.xlabel("Checkpoint")
plt.ylabel("Accuracies Evaluated by The Device's Model At Checkpoint")

plt.show()
print()