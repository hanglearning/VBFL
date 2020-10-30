import matplotlib.pyplot as plt
from os import listdir
import sys

log_folder_1 = sys.argv[1]
log_folder_2 = sys.argv[2]


all_rounds_log_files_1 = sorted([f for f in listdir(log_folder_1) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
all_rounds_log_files_2 = sorted([f for f in listdir(log_folder_2) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

draw_comm_rounds = len(all_rounds_log_files_1) if len(all_rounds_log_files_1) < len(all_rounds_log_files_2) else len(all_rounds_log_files_2)

draw_comm_rounds = 49

device_global_accuracies_across_rounds_1 = []
device_global_accuracies_across_rounds_2 = []

# record global accuracies
for log_file in all_rounds_log_files_1:
	file = open(f"{log_folder_1}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device'):
			accuracy = round(float(line.split(":")[-1]), 3)
			device_global_accuracies_across_rounds_1.append(accuracy)
			break

for log_file in all_rounds_log_files_2:
	file = open(f"{log_folder_2}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('device'):
			accuracy = round(float(line.split(":")[-1]), 3)
			device_global_accuracies_across_rounds_2.append(accuracy)
			break

plt.xticks(range(draw_comm_rounds), [i for i in range(1, draw_comm_rounds + 1)], rotation=90)
# draw global accuracies graph
plt.plot(range(draw_comm_rounds), device_global_accuracies_across_rounds_1[:draw_comm_rounds], label=f'global accuracy 3M/20 -vh 1.0')

plt.plot(range(draw_comm_rounds), device_global_accuracies_across_rounds_2[:draw_comm_rounds], label=f'global accuracy 3M/20 -vh 0.08')

annotating_points = 20
for accuracy_iter in range(len(device_global_accuracies_across_rounds_1)):
	if not accuracy_iter % (len(device_global_accuracies_across_rounds_1) // annotating_points):
		plt.annotate(device_global_accuracies_across_rounds_1[accuracy_iter], xy=(accuracy_iter, device_global_accuracies_across_rounds_1[accuracy_iter]), size=8)
for accuracy_iter in range(len(device_global_accuracies_across_rounds_2)):
	if not accuracy_iter % (len(device_global_accuracies_across_rounds_2) // annotating_points):
		plt.annotate(device_global_accuracies_across_rounds_2[accuracy_iter], xy=(accuracy_iter, device_global_accuracies_across_rounds_2[accuracy_iter]), size=8)

plt.legend(loc='best')
plt.xlabel('Comm Round')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('PoS 3/20 malicious before and after validators involved')
plt.show()
print()