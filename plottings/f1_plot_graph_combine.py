# figure 1

import matplotlib.pyplot as plt
from os import listdir
import sys

log_folder_b = sys.argv[1]
log_folder_m = sys.argv[2]
log_folder_PoS_3_vh_008_run1 = sys.argv[3]

all_rounds_log_files_b = sorted([f for f in listdir(log_folder_b) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_m = sorted([f for f in listdir(log_folder_m) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_3_vh_008_run1 = sorted([f for f in listdir(log_folder_PoS_3_vh_008_run1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_files_b) if len(all_rounds_log_files_b) < len(all_rounds_log_files_m) else len(all_rounds_log_files_m)

draw_comm_rounds = 100
plt.figure(dpi=250)

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
PoS_3_vh_008_run1_accuracies = []

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

# get global accuracy PoS_3_vh_008
runs = ['PoS_3_vh_008_run1']
for run in runs:
	vars()[f"all_rounds_{run}"]
	for log_file_folder in vars()[f"all_rounds_{run}"]:
		if len(log_file_folder) > 8:
			continue
		try:
			file = open(f"{vars()[f'log_folder_{run}']}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		except:
			log_file_folder = f"comm_{int(log_file_folder.split('_')[-1])-1}"
			file = open(f"{vars()[f'log_folder_{run}']}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_1'):
				accuracy = round(float(line.split(":")[-1]), 3)
				vars()[f'{run}_accuracies'].append(accuracy)
				break

m_device_accuracies_across_rounds = m_device_accuracies_across_rounds[:draw_comm_rounds]
b_device_accuracies_across_rounds = b_device_accuracies_across_rounds[:draw_comm_rounds]
PoS_3_vh_008_run1_accuracies = PoS_3_vh_008_run1_accuracies[:draw_comm_rounds]

# draw graphs over all available comm rounds
# plt.xticks(range(draw_comm_rounds), [i for i in range(1, draw_comm_rounds + 1)], rotation=90)
plt.plot(range(draw_comm_rounds), b_device_accuracies_across_rounds, label=f'Vanilla FL all {total_devices} legitimate devices', color='orange')
plt.plot(range(draw_comm_rounds), PoS_3_vh_008_run1_accuracies, label=r'VBFL 3 out of 20 malicious devices', color='green')
plt.plot(range(draw_comm_rounds), m_device_accuracies_across_rounds, label=f'Vanilla FL {total_malicious_devices} out of {total_devices} malicious devices', color='blue')
#plt.plot(range(draw_comm_rounds), a_device_accuracies_across_rounds, label=f'all {total_devices} benigh PoS')

if b_device_accuracies_across_rounds:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(b_device_accuracies_across_rounds)):
		if not accuracy_iter % (len(b_device_accuracies_across_rounds) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(b_device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, b_device_accuracies_across_rounds[accuracy_iter]), size=12)

if m_device_accuracies_across_rounds:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(m_device_accuracies_across_rounds)):
		if not accuracy_iter % (len(m_device_accuracies_across_rounds) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(m_device_accuracies_across_rounds[accuracy_iter], xy=(accuracy_iter, m_device_accuracies_across_rounds[accuracy_iter]), size=12)

if PoS_3_vh_008_run1_accuracies:
	annotating_points = 5
	skipped_1 = False
	for accuracy_iter in range(len(PoS_3_vh_008_run1_accuracies)):
		if not accuracy_iter % (len(PoS_3_vh_008_run1_accuracies) // annotating_points):
			if not skipped_1:
				skipped_1 = True
				continue
			plt.annotate(PoS_3_vh_008_run1_accuracies[accuracy_iter], xy=(accuracy_iter, PoS_3_vh_008_run1_accuracies[accuracy_iter]), size=12)


plt.legend(loc='l', bbox_to_anchor=(0.32,0.7))
plt.xlabel('Communication Round')
plt.ylabel('Global Accuracy')
plt.title('Comparison of Global Model Accuracy')
# plt.title('Global Model Accuracy Comparisons Before and After Introducing Noices through vanilla FedAvg Communication Rounds On MNIST Dataset Using MNIST_CNN')
plt.show()
print()