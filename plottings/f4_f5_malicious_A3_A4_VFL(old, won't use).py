# figure 4 and 5

import matplotlib.pyplot as plt
import os
import sys
from adjustText import adjust_text
import ast
import numpy as np


v1_folder_path = sys.argv[1]
v3_folder_path = sys.argv[2]
v5_folder_path = sys.argv[3]

all_v1_folders = sorted([f for f in os.listdir(v1_folder_path) if not f.startswith('.')])
all_v3_folders = sorted([f for f in os.listdir(v3_folder_path) if not f.startswith('.')])
all_v5_folders = sorted([f for f in os.listdir(v5_folder_path) if not f.startswith('.')])

draw_comm_rounds = 100

malicious_workers_logs = {}
all_workers_logs = {}
noise_variances = ['1', '3', '5']

all_workers = [str(idx) for idx in range(1, 21)]
for nv in noise_variances:
	malicious_workers_logs[f'noise_variance_{nv}'] = {}
	all_workers_logs[f'noise_variance_{nv}'] = {}
	for folder_name in vars()[f'all_v{nv}_folders']:
		malicious_workers_logs[f'noise_variance_{nv}'][folder_name] = {}
		all_workers_logs[f'noise_variance_{nv}'][folder_name] = {}
		# get malicious worker idx
		malicious_workers = [f.split('_')[1] for f in os.listdir(vars()[f'v{nv}_folder_path'] + f'/{folder_name}/comm_1') if 'M' in f]
		for malicious_worker_idx in malicious_workers:
			malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx] = {}
			malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['A1'] = []
			malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['A5'] = []
			malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['Noisy_A5'] = []
			malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['noise_variance'] = []
		
		for worker in all_workers:
			all_workers_logs[f'noise_variance_{nv}'][folder_name][worker] = {}
			all_workers_logs[f'noise_variance_{nv}'][folder_name][worker]['A1'] = []
			all_workers_logs[f'noise_variance_{nv}'][folder_name][worker]['A5'] = []
		# record values
		for comm_iter in range(draw_comm_rounds):
			log_folder = vars()[f'v{nv}_folder_path'] + f'/{folder_name}/comm_{comm_iter+1}'
			# record A1
			for malicious_worker_idx in malicious_workers:
				file = open(f"{log_folder}/client_{malicious_worker_idx}_local_comm_{comm_iter+1}.txt","r")
				lines_list = file.read().split("\n")
				for line in lines_list:
					if 'epoch_1' in line:
						malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['A1'].append(float(line.split(' ')[-1]))
					if 'epoch_5' in line:
						malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['A5'].append(float(line.split(' ')[-1]))
				file = open(f"{log_folder}/client_{malicious_worker_idx}_M_local_comm_{comm_iter+1}.txt","r")
				lines_list = file.read().split("\n")
				for line in lines_list:
					if 'noise_injected' in line:
						malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['Noisy_A5'].append(float(line.split(' ')[-1]))
					if 'noise_variance' in line:
						malicious_workers_logs[f'noise_variance_{nv}'][folder_name][malicious_worker_idx]['noise_variance'].append(ast.literal_eval(line.split(':')[-1][1:]))
			for worker in all_workers:
				file = open(f"{log_folder}/client_{worker}_local_comm_{comm_iter+1}.txt","r")
				lines_list = file.read().split("\n")
				for line in lines_list:
					if 'epoch_1' in line:
						all_workers_logs[f'noise_variance_{nv}'][folder_name][worker]['A1'].append(float(line.split(' ')[-1]))
					if 'epoch_5' in line:
						all_workers_logs[f'noise_variance_{nv}'][folder_name][worker]['A5'].append(float(line.split(' ')[-1]))


# draws_a1_a5 = {k:[] for k in all_workers}
# draws_a1_noisya5 = {}

# for worker in all_workers:
# 	for _ in range(draw_comm_rounds):
# 		a1_a5 = abs(all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A1'][_] - all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A5'][_])
# 		draws_a1_a5[worker].append(a1_a5)

# for worker, a1_a5 in draws_a1_a5.items():
# 	plt.plot(range(draw_comm_rounds), a1_a5, label='worker')
# plt.legend(loc='l', bbox_to_anchor=(0.32,0.7))
# plt.show()

# worker = '5'
# draws5 = []
# for _ in range(draw_comm_rounds):
# 	a1_a5 = abs(all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A1'][_] - all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A5'][_])
# 	draws5.append(a1_a5)
# plt.plot(range(draw_comm_rounds), draws5, label=worker)

# worker = '12'
# draws12 = []
# for _ in range(draw_comm_rounds):
# 	a1_a5 = abs(all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A1'][_] - all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A5'][_])
# 	draws12.append(a1_a5)
# plt.plot(range(draw_comm_rounds), draws12, label=worker)


# worker = '15'
# draws15 = []
# for _ in range(draw_comm_rounds):
# 	a1_a5 = abs(all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A1'][_] - all_workers_logs['noise_variance_1']['11092020_075048'][worker]['A5'][_])
# 	draws15.append(a1_a5)
# plt.plot(range(draw_comm_rounds), draws15, label=worker)
# plt.legend(loc='b')
# plt.show()

# 17, 13, 5
draws_a1 = []
draws_a2 = []
for _ in range(draw_comm_rounds):
	a1 = abs(malicious_workers_logs['noise_variance_1']['11092020_075048']['17']['A1'][_] - malicious_workers_logs['noise_variance_1']['11092020_075048']['17']['A5'][_])
	draws_a1.append(a1)
	a2 = abs(malicious_workers_logs['noise_variance_1']['11092020_075048']['5']['A1'][_] - malicious_workers_logs['noise_variance_1']['11092020_075048']['5']['Noisy_A5'][_])
	draws_a2.append(a2)
	# draws_a3.append(np.mean(malicious_workers_logs['noise_variance_1']['11092020_075048']['5']['noise_variance'][_]))
plt.figure(dpi=250)
plt.plot(range(draw_comm_rounds), draws_a1, label=r'$AC_1 = |A^{wl}(L_j^{wl}(1)) - A^{wl}(L_j^{wl}(n))|$, n=5', color='green')
plt.plot(range(draw_comm_rounds), draws_a2, label=r"$AC_2 = |A^{wl}(L_j^{wl}(1)) - A^{wm}(L_j^{wm}(n'))|$, n'=5", color='red')
# plt.plot(range(draw_comm_rounds), draws_a3, label='noise_variance_1', color='orange')

plt.legend(loc='b')
plt.xlabel('Communication Round')
plt.ylabel('Local Accuracy Change')
plt.title('Accuracy Changes Comparison')
plt.show()
plt.clf()


draws_a1 = []
draws_a2 = []
draws_a3 = []
for _ in range(draw_comm_rounds):
	a1 = abs(malicious_workers_logs['noise_variance_3']['11092020_115043']['5']['A1'][_] - malicious_workers_logs['noise_variance_3']['11092020_115043']['5']['A5'][_])
	draws_a1.append(a1)
	a2 = abs(malicious_workers_logs['noise_variance_3']['11092020_115043']['5']['A1'][_] - malicious_workers_logs['noise_variance_3']['11092020_115043']['5']['Noisy_A5'][_])
	draws_a2.append(a2)
	# draws_a3.append(np.mean(malicious_workers_logs['noise_variance_1']['11092020_075048']['5']['noise_variance'][_]))

plt.plot(range(draw_comm_rounds), draws_a1, label='a1-a5', color='green')
plt.plot(range(draw_comm_rounds), draws_a2, label='a1-noisy_a5', color='red')



draws_a1 = []
draws_a2 = []
draws_a3 = []
for _ in range(draw_comm_rounds):
	a1 = abs(malicious_workers_logs['noise_variance_5']['11122020_091200']['12']['A1'][_] - malicious_workers_logs['noise_variance_5']['11122020_091200']['12']['A5'][_])
	draws_a1.append(a1)
	a2 = abs(malicious_workers_logs['noise_variance_5']['11122020_091200']['12']['A1'][_] - malicious_workers_logs['noise_variance_5']['11122020_091200']['12']['Noisy_A5'][_])
	draws_a2.append(a2)
	# draws_a3.append(np.mean(malicious_workers_logs['noise_variance_5']['11122020_091200']['12']['noise_variance'][_]))


plt.plot(range(draw_comm_rounds), draws_a1, label='a1-a5', color='green')
plt.plot(range(draw_comm_rounds), draws_a2, label='a1-noisy_a5', color='red')
#plt.plot(range(draw_comm_rounds), draws_a3, label='noise_variance_5', color='orange')


#plt.plot(range(draw_comm_rounds), draws_a3, label='noise_variance_1', color='orange')

plt.show()

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