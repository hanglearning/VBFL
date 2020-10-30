# figure 9

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os import listdir
import sys

# 20 devices
switch_1 = sys.argv[1]
switch_2 = sys.argv[2]
switch_3 = sys.argv[3]
no_switch_1 = sys.argv[4]
no_switch_2 = sys.argv[5]
no_switch_3 = sys.argv[6]

all_rounds_switch_1 = sorted([f for f in listdir(switch_1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_switch_2 = sorted([f for f in listdir(switch_2) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_switch_3 = sorted([f for f in listdir(switch_3) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_no_switch_1 = sorted([f for f in listdir(no_switch_1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_no_switch_2 = sorted([f for f in listdir(no_switch_2) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_no_switch_3 = sorted([f for f in listdir(no_switch_3) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = 100
plt.figure(dpi=250)

# get num of devices with their maliciousness
	
switch_1_accuracies = {}
switch_2_accuracies = {}
switch_3_accuracies = {}
no_switch_1_accuracies = {}
no_switch_2_accuracies = {}
no_switch_3_accuracies = {}

for log_var in ["switch_1", "switch_2", "switch_3", "no_switch_1", "no_switch_2", "no_switch_3"]:
	# get global accuracy
	for log_file_folder in vars()[f'all_rounds_{log_var}']:
		if len(log_file_folder) > 8:
			continue
		try:
			file = open(f"{vars()[log_var]}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		except:
			log_file_folder = f"comm_{int(log_file_folder.split('_')[-1])-1}"
			file = open(f"{vars()[log_var]}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_'):
				device_idx = line.split(":")[0].split(" ")[0]
				accuracy = round(float(line.split(":")[-1]), 3)
				b_or_m = line.split(":")[0].split(" ")[-1]
				whole_device_idx = f'{device_idx} {b_or_m}'
				if not whole_device_idx in vars()[f'{log_var}_accuracies'].keys():
					vars()[f'{log_var}_accuracies']
					vars()[f'{log_var}_accuracies'][whole_device_idx] = [accuracy]
				else:
					vars()[f'{log_var}_accuracies'][whole_device_idx].append(accuracy)

for device_idx, accuracies in no_switch_1_accuracies.items():
	plt.plot(range(draw_comm_rounds), accuracies, color='blue')

for device_idx, accuracies in switch_1_accuracies.items():
	plt.plot(range(draw_comm_rounds), accuracies, color='orange')

orange_patch = mpatches.Patch(color='orange', label='Global Accuracies With Role Switch ON')
blue_patch = mpatches.Patch(color='blue', label='Global Accuracies With Role Switch OFF')

plt.legend(handles=[orange_patch, blue_patch], loc='best', prop={'size': 15})
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracies of Communication Rounds')
plt.title('Global Model Accuracy Comparisons')
# plt.title('Global Model Accuracy Comparisons Before and After Introducing Noices through vanilla FedAvg Communication Rounds On MNIST Dataset Using MNIST_CNN')
plt.show()
print()