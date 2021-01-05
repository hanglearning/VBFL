# vad (validation accuracy difference)

import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

false_positive_malious_nodes_inside_slipped = sys.argv[1]
true_positive_good_nodes_inside_correct = sys.argv[2]


malious_nodes_inside_slipped = {}
good_nodes_inside_correct = {}

draw_comm_rounds = 10
plt.figure(dpi=250)

file = open(false_positive_malious_nodes_inside_slipped,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if 'r' in line:
		acc_diff = round(float(line.split(' ')[0]),2)
		comm_round = int(line.split(' ')[-1])
		if comm_round > 30:
			break
		if comm_round not in malious_nodes_inside_slipped.keys():
			malious_nodes_inside_slipped[comm_round] = [acc_diff]
		else:
			malious_nodes_inside_slipped[comm_round].append(acc_diff)

file = open(true_positive_good_nodes_inside_correct,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if 'r' in line:
		acc_diff = round(float(line.split(' ')[0]),2)
		comm_round = int(line.split(' ')[-1])
		if comm_round > 30:
			break
		if comm_round not in good_nodes_inside_correct.keys():
			good_nodes_inside_correct[comm_round] = [acc_diff]
		else:
			good_nodes_inside_correct[comm_round].append(acc_diff)

for comm_round, malious_nodes_acc_diff in malious_nodes_inside_slipped.items():

	for point in malious_nodes_acc_diff:
		plt.scatter(comm_round, point, c='r')

for comm_round, good_nodes_acc_diff in good_nodes_inside_correct.items():
	for point in good_nodes_acc_diff:
		plt.scatter(comm_round, point, c='g')

plt.axhline(y=0.08, linestyle='dashed', color='orange')
plt.yticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.08, 0.1, 0.15, 0.2], ["-0.20", "-0.15", "-0.10", "-0.05", "0.00", "0.05", "0.08", "0.10", "0.15", "0.20"])
# plt.text(18, 0.08, r'$vh_{j}^{v}$=0.08', fontsize=15, va='bottom', ha='right')

plt.xlabel('Communication Round')
plt.ylabel(r'$vad = A^v(L_j^v(1)) - A^v(L_j^w(n))$', fontsize = 12)
plt.title("Validation Accuracy Difference")
line1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")
line2 = Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="green")


# red_circle = mpatches.Circle(color='red', label='Malicious Workers')
# blue_patch = mpatches.Circle(color='green', label='Legitimate Workers')

plt.legend((line1, line2), ('Malicious Workers', 'Legitimate Workers'), loc='best')
plt.show()



