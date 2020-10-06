import matplotlib.pyplot as plt
from os import listdir
import sys

log_folder = sys.argv[1]


all_rounds_log_files_g = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_l = sorted([f for f in listdir(log_folder) if f.startswith('worker_local_accuracies_comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_files_g) if len(all_rounds_log_files_g) < len(all_rounds_log_files_l) else len(all_rounds_log_files_l)

draw_comm_rounds = 60

device_global_accuracies_across_rounds = []

# record global accuracies
for log_file in all_rounds_log_files_g:
	file = open(f"{log_folder}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device'):
			accuracy = round(float(line.split(":")[-1]), 3)
			device_global_accuracies_across_rounds.append(accuracy)
			break

# record local accuracies by workers
# get num of devices with their maliciousness
benign_devices_idx_list = []
malicious_devices_idx_list = []
comm_1_file_path = f"{log_folder}/comm_1.txt"
file = open(comm_1_file_path,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if line.startswith('device'):
		device_idx = line.split(":")[0].split(" ")[0]
		device_maliciousness = line.split(":")[0].split(" ")[-1]
		if device_maliciousness == 'M':
			malicious_devices_idx_list.append(f"{device_idx} {device_maliciousness}")
		else:
			benign_devices_idx_list.append(f"{device_idx} {device_maliciousness}")

devices_idx_list = sorted(malicious_devices_idx_list, key=lambda k: int(k.split(" ")[0].split('_')[-1])) + sorted(benign_devices_idx_list, key=lambda k: int(k.split(" ")[0].split('_')[-1]))
	
devices_accuracies_across_rounds = dict.fromkeys(devices_idx_list)
for client_idx, _ in devices_accuracies_across_rounds.items():
	devices_accuracies_across_rounds[client_idx] = []

for log_file in all_rounds_log_files_l:
	file = open(f"{log_folder}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	this_round_workers = set()
	for line in lines_list:
		if line.startswith('device'):
			device_idx = line.split(":")[0].split(" ")[0]
			device_maliciousness = line.split(":")[0].split(" ")[-1]
			device_id_mali = f"{device_idx} {device_maliciousness}"
			this_round_workers.add(device_id_mali)
			accuracy = round(float(line.split(":")[-1]), 3)
			devices_accuracies_across_rounds[device_id_mali].append(accuracy)
	not_chosen_as_workers_this_round = list(set(devices_accuracies_across_rounds.keys()).difference(this_round_workers))
	for not_worker_device in not_chosen_as_workers_this_round:
		try:
			devices_accuracies_across_rounds[not_worker_device].append(devices_accuracies_across_rounds[not_worker_device][-1])
		except:
			# when later workers were not chosen as workers in the comm_1
			devices_accuracies_across_rounds[not_worker_device].append(0)



plt.xticks(range(draw_comm_rounds), [i for i in range(1, draw_comm_rounds + 1)], rotation=90)
# draw global accuracies graph
plt.plot(range(draw_comm_rounds), device_global_accuracies_across_rounds[:draw_comm_rounds], label=f'global accuracy')
# draw local accuracies
# draw graphs over all available comm rounds
for device_idx, accuracy_list in devices_accuracies_across_rounds.items():
	plt.plot(range(draw_comm_rounds), accuracy_list, label=device_idx)
plt.legend(loc='best')
plt.xlabel('Comm Round')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('Global Accuracies of devices vs Local Worker Accuracies')
plt.show()
print()