# figure 10
import matplotlib.pyplot as plt
import sys
import matplotlib.lines as mlines
from os import path

rewards_off_1 = sys.argv[1]
rewards_off_2 = sys.argv[2]
rewards_off_3 = sys.argv[3]
rewards_on_1 = sys.argv[4]
rewards_on_2 = sys.argv[5]
rewards_on_3 = sys.argv[6]

plt.figure(dpi=250)
draw_comm_rounds = 100

all_rounds = [f'comm_{i}' for i in range(1, 101)]

rewards_off_1_miners = []
rewards_off_2_miners = []
rewards_off_3_miners = []
rewards_on_1_miners = []
rewards_on_2_miners = []
rewards_on_3_miners = []

log_vars = ["rewards_off_1", "rewards_off_2", "rewards_off_3", "rewards_on_1", "rewards_on_2", "rewards_on_3"]

counter = 0

for log_var in log_vars:
	# record malicious miner rounds
	for log_file_folder in all_rounds:
		try:
			file = open(f"{vars()[log_var]}/{log_file_folder}/stake_{log_file_folder}.txt","r")
		except:
			continue
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('PoS_block_mined_by'):
				# if line.split(":")[-1].split(" ")[-1] == 'B':
				# 	counter += 1
				if line.split(":")[-1].split(" ")[-1] == 'M':
					vars()[f'{log_var}_miners'].append(int(log_file_folder.split('_')[-1]))

print(counter)
x_axis_labels = ["off_1", "off_2", "off_3", "on_1", "on_2", "on_3"]
plt.xticks(range(len(x_axis_labels)), x_axis_labels)

for log_var_iter in range(len(log_vars)):
	log_var = log_vars[log_var_iter]
	for point in vars()[f'{log_var}_miners']:
		plt.scatter(log_var_iter, point, label=x_axis_labels[log_var_iter])

# uncomment for 3/20 Malicious
# plt.scatter(3, 20, label="on_1", color='white')
# plt.scatter(4, 20, label="on_2", color='white')
# plt.scatter(5, 20, label="on_3", color='white')

plt.xlabel('Worker Rewards Settings')
plt.ylabel('Communication Rounds')
plt.title('VBFL-PoS Effectiveness on Miner Selection for 6/20 Malicious')

# plt.show()

# figure 11 Stake Curve

rewards_off_1_stakes = {}
rewards_off_2_stakes = {}
rewards_off_3_stakes = {}
rewards_on_1_stakes = {}
rewards_on_2_stakes = {}
rewards_on_3_stakes = {}

for log_var in log_vars:
	for log_file_folder in all_rounds:
		log_file_path = f"{vars()[log_var]}/{log_file_folder}/stake_{log_file_folder}.txt"
		while not path.exists(log_file_path):
			log_file_folder = f"comm_{int(log_file_folder.split('_')[-1])-1}"
			log_file_path = f"{vars()[log_var]}/{log_file_folder}/stake_{log_file_folder}.txt"
		file = open(log_file_path,"r")
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			if line.startswith('device_'):
				device_idx = line.split(":")[0].split(" ")[0]
				stake = int(line.split(":")[-1])
				b_or_m = line.split(":")[0].split(" ")[-1]
				whole_device_idx = f'{device_idx} {b_or_m}'
				if not whole_device_idx in vars()[f'{log_var}_stakes'].keys():
					vars()[f'{log_var}_stakes'][whole_device_idx] = [stake]
				else:
					vars()[f'{log_var}_stakes'][whole_device_idx].append(stake)

plt.clf()

log_vars = ["rewards_off_1", "rewards_off_2", "rewards_off_3", "rewards_on_1", "rewards_on_2", "rewards_on_3"]
axs_iters_y = [0,1,2,0,1,2]
fig, axs = plt.subplots(2, 3, sharex=True)
import matplotlib.ticker as mticker    
for log_var_iter in range(len(log_vars)):
	log_var = log_vars[log_var_iter]
	if log_var_iter in [0, 1, 2]:
		# the first row
		x = 0
	else:
		x = 1
	# set ylim and sharey
	if x == 0:
		axs[x, axs_iters_y[log_var_iter]].set_ylim(0, 1.2)
	else:
		axs[x, axs_iters_y[log_var_iter]].set_ylim(0, 1000)
	# hide y axis if not first column
	if axs_iters_y[log_var_iter] != 0:
		axs[x, axs_iters_y[log_var_iter]].get_yaxis().set_visible(False)
	# set label
	if axs_iters_y[log_var_iter] == 0:
		axs[x, axs_iters_y[log_var_iter]].set_ylabel('Total Stakes')
	if x == 1:
		axs[x, axs_iters_y[log_var_iter]].set_xlabel(r'$R_j$')
	axs[x, axs_iters_y[log_var_iter]].set_title(f'Stake Accum {log_var[8:]}')
	axs[x, axs_iters_y[log_var_iter]].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fk'))
	for device_idx, stakes in vars()[f'{log_var}_stakes'].items():
		if device_idx.split(' ')[-1] == 'M':
			axs[x, axs_iters_y[log_var_iter]].plot(range(draw_comm_rounds), [_/1000 for _ in stakes], color='red')
		else:
			axs[x, axs_iters_y[log_var_iter]].plot(range(draw_comm_rounds), [_/1000 for _ in stakes], color='green')

	red_line = mlines.Line2D([], [], color='red', label=r"$dm$")
	green_line = mlines.Line2D([], [], color='green', label=r'$dl$')

	axs[x, axs_iters_y[log_var_iter]].legend(handles=[red_line, green_line], loc='best', prop={'size': 10})
	# Stake Accumulation Curve
	# plt.xlabel('Communication Rounds')
	# plt.ylabel('Total Stake')


plt.show()

# show individually
# log_vars = ["rewards_off_1", "rewards_off_2", "rewards_off_3", "rewards_on_1", "rewards_on_2", "rewards_on_3"]

# for log_var in log_vars:
#     plt.clf()
#     for device_idx, stakes in vars()[f'{log_var}_stakes'].items():
#         if device_idx.split(' ')[-1] == 'M':
#             plt.plot(range(draw_comm_rounds), stakes, color='red')
#         else:
#             plt.plot(range(draw_comm_rounds), stakes, color='green')

#     red_line = mlines.Line2D([], [], color='red', label=r"Malicious $d$s")
#     green_line = mlines.Line2D([], [], color='green', label=r'Legitimate $d$s')

#     plt.legend(handles=[red_line, green_line], loc='best', prop={'size': 15})

#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Total Stake')
#     plt.title(f'Stake Accumulation Curve {log_var[8:]}')
#     plt.show()