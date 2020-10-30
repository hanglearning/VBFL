# figure 2

import matplotlib.pyplot as plt
import os
import sys
from adjustText import adjust_text
from os import listdir

all_epochs_log_folder = sys.argv[1]

all_available_epochs = sorted([f for f in listdir(all_epochs_log_folder) if f.startswith('-E')], key=lambda x: int(x.split(' ')[-1]))

log_vars = [f"epoch_{epoch.split(' ')[-1]}" for epoch in all_available_epochs]

draw_comm_rounds = 10
all_devices_idxes = [f'client_{i}' for i in range(1, 21)]

all_rounds = [f'comm_{i}' for i in range(1, 51)]

# log_vars = ["epoch_6", "epoch_8", "epoch_10", "epoch_12", "epoch_14", "epoch_16", "epoch_18", "epoch_20"]

for log_var in log_vars:
	vars()[f'{log_var}_accuracies'] = {key: {} for key in all_devices_idxes}
	vars()[f'{log_var}_accuracies']['global_accuracies'] = []
	for device_idx, the_dict in vars()[f'{log_var}_accuracies'].items():
		if device_idx == 'global_accuracies':
			continue
		the_dict['local_accuracies_e_1'] = []
		the_dict['local_accuracies_e_end'] = []
		the_dict['draw_accuracies'] = []

for log_var_iter in range(len(log_vars)):
	log_var = log_vars[log_var_iter]
	# record malicious miner rounds
	for comm_round in all_rounds:
		# get global_accuracy
		epoch_log_var_path = f'{all_epochs_log_folder}/{all_available_epochs[log_var_iter]}/{comm_round}'
		file = open(f"{epoch_log_var_path}/global_{comm_round}.txt","r")
		global_accuracy = round(float(file.read().split("\n")[0].split(' ')[-1]), 3)
		vars()[f'{log_var}_accuracies']['global_accuracies'].append(global_accuracy)
		for device_idx in all_devices_idxes:
			file = open(f"{epoch_log_var_path}/{device_idx}_local_{comm_round}.txt","r")
			file_whole_text = file.read()
			local_accuracies_e_1 = round(float(file_whole_text.split("\n")[0].split(' ')[-1]), 3)
			local_accuracies_e_end = round(float(file_whole_text.split("\n")[-2].split(' ')[-1]), 3)
			vars()[f'{log_var}_accuracies'][device_idx]['local_accuracies_e_1'].append(local_accuracies_e_1)
			vars()[f'{log_var}_accuracies'][device_idx]['local_accuracies_e_end'].append(local_accuracies_e_end)
			vars()[f'{log_var}_accuracies'][device_idx]['draw_accuracies'].extend([local_accuracies_e_1, local_accuracies_e_end, global_accuracy])

x_axis_labels = []
for i in range(draw_comm_rounds):
	x_axis_labels.append(f'le1')
	x_axis_labels.append(f"le{log_var.split('_')[-1]}")
	x_axis_labels.append(f'g{i+1}')
	
# # draw subplots
# for log_var_iter in range(len(log_vars)):
# 	log_var = log_vars[log_var_iter]
# 	plt.clf()
# 	fig, axs = plt.subplots(5, 4, sharex=True, sharey=True)
# 	for device_idx_iter in range(len(all_devices_idxes)):
# 		device_idx = all_devices_idxes[device_idx_iter]
# 		row = device_idx_iter // 4 + 1 - 1
# 		column = device_idx_iter % 4 - 1
# 		axs[row, column].set_title(device_idx)
# 		axs[row, column].set_xticks(range(len(x_axis_labels)), x_axis_labels)
# 		axs[row, column].plot(range(len(x_axis_labels)), vars()[f'{log_var}_accuracies'][device_idx]['draw_accuracies'][:draw_comm_rounds*3])
# 	fig.suptitle(log_var)
# 	plt.xlabel('timestamp')
# 	plt.ylabel('accuracy')
# 	plt.show()

plt.clf()
x_axis_labels = []
for i in range(50):
	x_axis_labels.append(f'le1')
	x_axis_labels.append(f"le{log_var.split('_')[-1]}")
	x_axis_labels.append(f'g{i+1}')
plt.xticks(range(len(x_axis_labels)), x_axis_labels, rotation=90)
plt.plot(range(len(x_axis_labels)), vars()['epoch_20_accuracies']['client_14']['draw_accuracies'])
plt.title('epoch_20 client_14')
plt.show()