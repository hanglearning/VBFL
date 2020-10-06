import matplotlib.pyplot as plt
from os import listdir
import sys

log_folder_b = sys.argv[1]
log_folder_m = sys.argv[2]
log_folder_add = None
try:
	log_folder_add = sys.argv[3]
except:
	pass

all_rounds_log_files_b = sorted([f for f in listdir(log_folder_b) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_m = sorted([f for f in listdir(log_folder_m) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_a = None
if log_folder_add:
	all_rounds_log_files_a = sorted([f for f in listdir(log_folder_add) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_files_b) if len(all_rounds_log_files_b) < len(all_rounds_log_files_m) else len(all_rounds_log_files_m)

draw_comm_rounds = 100

# get num of devices with their maliciousness
benign_devices_idx_list = []
malicious_devices_idx_list = []
comm_1_file_path = f"{log_folder_m}/comm_1.txt"
file = open(comm_1_file_path,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if line.startswith('client'):
		device_idx = line.split(":")[0].split(" ")[0]
		device_maliciousness = line.split(":")[0].split(" ")[-1]
		if device_maliciousness == 'M':
			malicious_devices_idx_list.append(device_idx)
		else:
			benign_devices_idx_list.append(device_idx)

total_malicious_devices = len(malicious_devices_idx_list)
total_devices = len(malicious_devices_idx_list + benign_devices_idx_list)
	
b_device_accuracies_across_rounds = []
m_device_accuracies_across_rounds = []
a_device_accuracies_across_rounds = []

for log_file in all_rounds_log_files_b:
	file = open(f"{log_folder_b}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			b_device_accuracies_across_rounds.append(accuracy)
			break

for log_file in all_rounds_log_files_m:
	file = open(f"{log_folder_m}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			m_device_accuracies_across_rounds.append(accuracy)
			break

if all_rounds_log_files_a:
	for log_file in all_rounds_log_files_a:
		file = open(f"{log_folder_add}/{log_file}","r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_1'):
				accuracy = round(float(line.split(":")[-1]), 3)
				a_device_accuracies_across_rounds.append(accuracy)
				break

# draw graphs over all available comm rounds
plt.xticks(range(draw_comm_rounds), [i for i in range(1, draw_comm_rounds + 1)], rotation=90)
plt.plot(range(draw_comm_rounds), m_device_accuracies_across_rounds[:draw_comm_rounds], label=f'{total_malicious_devices}/{total_devices} malicious curve')
plt.plot(range(draw_comm_rounds), b_device_accuracies_across_rounds[:draw_comm_rounds], label=f'all {total_devices} benigh curve')
plt.plot(range(draw_comm_rounds), a_device_accuracies_across_rounds[:draw_comm_rounds], label=f'all {total_devices} benigh PoS')

if a_device_accuracies_across_rounds:
	annotating_points = 20
	for accuracy_iter in range(len(a_device_accuracies_across_rounds)):
		if not accuracy_iter % (len(a_device_accuracies_across_rounds) // annotating_points):
			plt.annotate(a_device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, a_device_accuracies_across_rounds[accuracy_iter]), size=8)


plt.legend(loc='best')
plt.xlabel('Comm Round')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('Learning Curve Comparison Before and After Introducing Noices through Vanilla FedAvg Comm Rounds')
plt.show()
print()