# figure 7

import matplotlib.pyplot as plt
from os import listdir
import sys

# 20 devices
log_folder_VFL_0 = sys.argv[1]
log_folder_VFL_3 = sys.argv[2]
log_folder_PoS_0_vh_1 = sys.argv[3]
log_folder_PoS_0_vh_008 = sys.argv[4]
log_folder_PoS_3_vh_008_run1 = sys.argv[5]
log_folder_PoS_3_vh_008_run2 = sys.argv[6]
log_folder_PoS_3_vh_008_run3 = sys.argv[7]

draw_comm_rounds = 100

all_rounds_VFL_0 = sorted([f for f in listdir(log_folder_VFL_0) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_VFL_3 = sorted([f for f in listdir(log_folder_VFL_3) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

all_rounds_PoS_0_vh_1 = sorted([f for f in listdir(log_folder_PoS_0_vh_1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_0_vh_008 = sorted([f for f in listdir(log_folder_PoS_0_vh_008) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_3_vh_008_run1 = sorted([f for f in listdir(log_folder_PoS_3_vh_008_run1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_3_vh_008_run2 = sorted([f for f in listdir(log_folder_PoS_3_vh_008_run2) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_PoS_3_vh_008_run3 = sorted([f for f in listdir(log_folder_PoS_3_vh_008_run3) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = 100
plt.figure(dpi=250)

# get num of devices with their maliciousness
	
VFL_0_accuracies = []
VFL_3_accuracies = []
PoS_0_vh_1_accuracies = []
PoS_0_vh_008_accuracies = []
PoS_3_vh_008_run1_accuracies = []
PoS_3_vh_008_run2_accuracies = []
PoS_3_vh_008_run3_accuracies = []

# get global accuracy VFL_0
for log_file in all_rounds_VFL_0:
	file = open(f"{log_folder_VFL_0}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			VFL_0_accuracies.append(accuracy)
			break

# get global accuracy VFL_3
for log_file in all_rounds_VFL_3:
	file = open(f"{log_folder_VFL_3}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			VFL_3_accuracies.append(accuracy)
			break

# get global accuracy PoS_0_vh_1
for log_file_folder in all_rounds_PoS_0_vh_1:
	file = open(f"{log_folder_PoS_0_vh_1}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			PoS_0_vh_1_accuracies.append(accuracy)
			break

# get global accuracy PoS_0_vh_008
for log_file_folder in all_rounds_PoS_0_vh_008:
	if len(log_file_folder) > 8:
		continue
	file = open(f"{log_folder_PoS_0_vh_008}/{log_file_folder}/accuracy_{log_file_folder}.txt","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device_1'):
			accuracy = round(float(line.split(":")[-1]), 3)
			PoS_0_vh_008_accuracies.append(accuracy)
			break

# get global accuracy PoS_3_vh_008
runs = ['PoS_3_vh_008_run1', 'PoS_3_vh_008_run2', 'PoS_3_vh_008_run3']
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


# draw graphs over all available comm rounds
plt.plot(range(draw_comm_rounds), VFL_0_accuracies, label=r'vanilla FL all 20 legitimate $d$s', c='C1')

plt.plot(range(draw_comm_rounds), PoS_0_vh_1_accuracies, label=r'PoS all 20 legitimate $d$s, $vh_j^v=1$', c='C2')

plt.plot(range(draw_comm_rounds), PoS_0_vh_008_accuracies, label=r'PoS all 20 legitimate $d$s, $vh_j^v=0.08$', c='C4')

plt.plot(range(draw_comm_rounds), PoS_3_vh_008_run1_accuracies, label=r'PoS 3/20 malicious $d$s, $vh_j^v=0.08$ run1', c='C3')

plt.plot(range(draw_comm_rounds), VFL_3_accuracies, label=f'vanilla FL 3/20 malicious $d$s', c='C0')

# annotations
# annotating_points = 5
# for accuracy_iter in range(len(VFL_0_accuracies)):
# 	if not accuracy_iter % (len(VFL_0_accuracies) // annotating_points):
# 		plt.annotate(VFL_0_accuracies[accuracy_iter], xy=(accuracy_iter, VFL_0_accuracies[accuracy_iter]), size=12)
# for accuracy_iter in range(len(VFL_3_accuracies)):
# 	if not accuracy_iter % (len(VFL_3_accuracies) // annotating_points):
# 		plt.annotate(VFL_3_accuracies[accuracy_iter], xy=(accuracy_iter, VFL_3_accuracies[accuracy_iter]), size=12)
# for accuracy_iter in range(len(PoS_3_vh_008_run1_accuracies)):
# 	if not accuracy_iter % (len(PoS_3_vh_008_run1_accuracies) // annotating_points):
# 		plt.annotate(PoS_3_vh_008_run1_accuracies[accuracy_iter], xy=(accuracy_iter, PoS_3_vh_008_run1_accuracies[accuracy_iter]), size=12)


plt.legend(loc='l', bbox_to_anchor=(0.39,0.75), prop={'size': 9})
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracies of Communication Rounds')
plt.title('Global Model Accuracy Comparisons')
# plt.title('Global Model Accuracy Comparisons Before and After Introducing Noices through vanilla FedAvg Communication Rounds On MNIST Dataset Using MNIST_CNN')
plt.show()

# figure 8

plt.clf()

plt.plot(range(draw_comm_rounds), PoS_3_vh_008_run1_accuracies, label=r'PoS 3/20 malicious $d$s, $vh_j^v=0.08$ run1', color='r')
for run_iter in range(len(runs)):
	if run_iter == 0:
		continue
	run = runs[run_iter]
	plt.plot(range(draw_comm_rounds), vars()[f'{run}_accuracies'], label=fr'PoS 3/20 malicious $d$s $vh_j^v=0.08$, run{run_iter+1}')
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracies of Communication Rounds')
plt.title('Global Model Accuracy Comparisons')
plt.legend(loc='b')
# plt.show()