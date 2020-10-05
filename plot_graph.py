import matplotlib.pyplot as plt
from os import listdir

latest_log_folder_name = sorted([f for f in listdir("logs") if not f.startswith('.')], reverse=True)[0]

log_files_folder_path = f"logs/{latest_log_folder_name}"

all_rounds_log_files = sorted([f for f in listdir(log_files_folder_path) if f.startswith('comm')])

# get num of devices with their maliciousness
benign_devices_idx_list = []
malicious_devices_idx_list = []
comm_1_file_path = f"{log_files_folder_path}/comm_1.txt"
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

for worker_idx, _ in devices_accuracies_across_rounds.items():
	devices_accuracies_across_rounds[worker_idx] = []
round_time_record = []
forking_record = []

for log_file in all_rounds_log_files:
	file = open(f"{log_files_folder_path}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device'):
			device_idx = line.split(":")[0].split(" ")[0]
			device_maliciousness = line.split(":")[0].split(" ")[-1]
			device_id_mali = f"{device_idx} {device_maliciousness}"
			accuracy = round(float(line.split(":")[-1]), 3)
			devices_accuracies_across_rounds[device_id_mali].append(accuracy)
		if line.startswith('comm_round_block_gen_time'):
			spent_time = round(float(line.split(":")[-1]), 2)
			round_time_record.append(spent_time)
		if line.startswith('forking'):
			forking_happened = line.split(":")[-1][1:2]
			forking_record.append(forking_happened)

round_time_record_with_forking_indicator = list(zip(round_time_record, forking_record))

# draw graphs over all available comm rounds
annotating_accuracy_list = []
for device_idx, accuracy_list in devices_accuracies_across_rounds.items():
	plt.xticks(range(len(round_time_record)), round_time_record_with_forking_indicator, rotation=90)
	plt.plot(range(len(round_time_record)), accuracy_list, label=device_idx)
	annotating_accuracy_list = accuracy_list
# annotate graph
annotating_points = 20
for accuracy_iter in range(len(annotating_accuracy_list)):
	if not accuracy_iter % (len(annotating_accuracy_list) // annotating_points):
		plt.annotate(annotating_accuracy_list[accuracy_iter], xy=(accuracy_iter, annotating_accuracy_list[accuracy_iter]), size=8)

plt.legend(loc='best')
plt.xlabel('Comm Round with Block Generation Time Point And Forking Indicator')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('Learning Curve through Block FedAvg Comm Rounds')
plt.show()
plt.savefig(f'{log_files_folder_path}/{latest_log_folder_name}.png')