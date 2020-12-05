# figure 2

import matplotlib.pyplot as plt
import os
import sys
from adjustText import adjust_text


log_folder = sys.argv[1]

all_rounds_log_folders = sorted([f for f in os.listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_folders)

draw_comm_rounds = 10
draw_device_idx = 'client_1'

local_accuracies_e1 = []
local_accuracies_e5 = []
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
	#local_accuracies.append(local_accuracy)
	draw_accuracies.append(local_accuracies_e1)
	draw_accuracies.append(local_accuracies_e5)
	# record global accuracy
	file = open(f"{sub_log_folder_path}/global_{sub_log_folder_name}.txt","r")
	global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
	global_accuracies.append(global_accuracy)
	draw_accuracies.append(global_accuracy)

print()

plt.figure(dpi=250)

x_axis_labels = []
x_axis_ls = []
x_axis_gs = []
for i in range(draw_comm_rounds):
	x_axis_labels.append(f'{i+1}le1')
	x_axis_labels.append(f'{i+1}le5')
	x_axis_labels.append(f'g{i+1}')
	# x_axis_ls.append(2*i)
	x_axis_gs.append(3*i+2)
# draw graphs over all available comm rounds
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
# draw the whole learning curve
plt.plot(range(len(x_axis_labels)), draw_accuracies, label=f'overall learning curve')


# connect local model learning curve
#plt.plot(x_axis_ls, local_accuracies, label=f'{draw_device_idx} local learning curve')
# connect global model learning curve
plt.plot(x_axis_gs, global_accuracies, label=f'global learning curve')

# dashed lines
# plt.vlines(range(len(x_axis_labels)), 0, draw_accuracies, linestyle="dashed", color='c')
# g8
# plt.vlines(23, 0, draw_accuracies[23], linestyle="dashed", color='c')
# # g4~g5
# plt.vlines(12, 0, draw_accuracies[12], linestyle="dashed", color='c')
# plt.vlines(13, 0, draw_accuracies[13], linestyle="dashed", color='c')
# # last 2
# plt.vlines(21, 0, draw_accuracies[21], linestyle="dashed", color='c')
# plt.vlines(19, 0, draw_accuracies[19], linestyle="dashed", color='c')

# annotate
# plt.annotate(draw_accuracies[23], xy=(23, draw_accuracies[23]), size=10)
# plt.annotate(draw_accuracies[12], xy=(12, draw_accuracies[12]), size=10)
# plt.annotate(draw_accuracies[13], xy=(13, draw_accuracies[13]), size=10)
# plt.annotate(draw_accuracies[21], xy=(21, draw_accuracies[21]), size=10)
# plt.annotate(draw_accuracies[19], xy=(19, draw_accuracies[19]), size=10)
# annotations = []
# annotations.append(plt.text(23, draw_accuracies[23],draw_accuracies[23]))
# annotations.append(plt.text(12, draw_accuracies[12],draw_accuracies[12]))
# annotations.append(plt.text(13, draw_accuracies[13],draw_accuracies[13]))
# annotations.append(plt.text(21, draw_accuracies[21],draw_accuracies[21]))
# annotations.append(plt.text(19, draw_accuracies[19],draw_accuracies[19]))

# annotations = []
# for x, y, acc in zip(range(len(x_axis_labels)), draw_accuracies, draw_accuracies):
# 	annotations.append(plt.text(x,y,acc))


#adjust_text(annotations, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

# annotate whole
# for draw_acc_iter in range(len(draw_accuracies)):
# 	draw_acc = draw_accuracies[draw_acc_iter]
# 	if draw_acc_iter < 30 and x_axis_labels[draw_acc_iter].startswith('le5'):
# 		plt.annotate(draw_acc, xy=(draw_acc_iter, draw_acc-0.02), size=10)
# 	elif draw_acc_iter < 30 and x_axis_labels[draw_acc_iter].startswith('le1'):
# 		plt.annotate(draw_acc, xy=(draw_acc_iter, draw_acc+0.03), size=10)
# 	else:
# 		plt.annotate(draw_acc, xy=(draw_acc_iter, draw_acc), size=13)

plt.legend(loc=2)
plt.title("A Random Device's Learning Curve in Vanilla FL")
plt.xlabel("Checkpoint")
plt.ylabel("Accuracy of Model At Checkpoint")

plt.show()
print()