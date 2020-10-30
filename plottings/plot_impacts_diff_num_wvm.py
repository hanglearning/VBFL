
import matplotlib.pyplot as plt
from os import listdir
import sys
from os import path

# 20 devices
run1 = sys.argv[1]
run2 = sys.argv[2]
run3 = sys.argv[3]
run4 = sys.argv[4]
run5 = sys.argv[5]
run6 = sys.argv[6]
run7 = sys.argv[7]

log_vars = ["8_9_3", "9_8_3", "10_7_3", "11_6_3", "12_5_3", "13_4_3", "14_3_3"]

for log_var_iter in range(len(log_vars)):
	log_var = log_vars[log_var_iter]
	vars()[f"{log_var}_global_accuracy"] = []
	vars()[f"{log_var}_log_folder_path"] = vars()[f'run{log_var_iter+1}']

draw_comm_rounds = 100

all_rounds = [f'comm_{i}' for i in range(1, draw_comm_rounds+1)]

plt.figure(dpi=250)

# get global accuracy PoS_3_vh_008
for log_var_iter in range(len(log_vars)):
	log_var = log_vars[log_var_iter]
	for round_iter in all_rounds:
		log_file_path = f"{vars()[f'{log_var}_log_folder_path']}/{round_iter}/accuracy_{round_iter}.txt"
		new_round_iter = round_iter
		while not path.exists(log_file_path):
			new_round_iter = f"comm_{int(new_round_iter.split('_')[-1])-1}"
			log_file_path = f"{vars()[f'{log_var}_log_folder_path']}/{new_round_iter}/accuracy_{new_round_iter}.txt"
		file = open(log_file_path,"r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_1'):
				accuracy = round(float(line.split(":")[-1]), 3)
				vars()[f'{log_var}_global_accuracy'].append(accuracy)
				break

for log_var in log_vars:
	plt.plot(range(draw_comm_rounds), vars()[f'{log_var}_global_accuracy'], label=log_var)

plt.legend(loc='b')
# plt.legend(loc='l', bbox_to_anchor=(0.39,0.75), prop={'size': 9})
plt.xlabel('Communication Rounds')
# plt.ylabel('Accuracies of Communication Rounds')
# plt.title('Global Model Accuracy Comparisons')
plt.title('Global Model Accuracy all with 3 Malicious vh=0.08')
plt.show()
