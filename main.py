# fedavg from https://github.com/WHDY/FedAvg/
# TODO redistribute offline() based on very transaction, not at the beginning of every loop
# TODO when accepting transactions, check comm_round must be in the same, that is to assume by default they only accept the transactions that are in the same round, so the final block contains only updates from the same round
# TODO subnets - potentially resolved by let each miner sends block 
# TODO let's not remove peers because of offline as peers may go back online later, instead just check if they are online or offline. Only remove peers if they are malicious. In real distributed network, remove peers if cannot reach for a long period of time
# assume no resending transaction mechanism if a transaction is lost due to offline or time out. most of the time, unnecessary because workers always send the newer updates, if it's not the last worker's updates
# assume just skip verifying a transaction if offline, in reality it may continue to verify what's left
# PoS also uses resync chain - the chain with highter stake
# only focus on catch malicious worker
# TODO need to make changes in these functions on Sunday
#pow_resync_chain
#update_model_after_chain_resync
# TODO miner sometimes receives worker transactions directly for unknown reason - discard tx if it's not the correct type
# TODO a chain is invalid if a malicious block is identified after this miner is identified as malicious
# TODO Do not associate with blacklisted node. This may be done already.
# TODO KickR continuousness should skip the rounds when nodes are not selected as workers
# TODO update forking log after loading network snapshots
# TODO in reuqest_to_download, forgot to check for maliciousness of the block miner
# future work
# TODO - non-even dataset distribution

import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DevicesInNetwork
from Block import Block
from Blockchain import Blockchain

# set program execution time for logging purpose
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"
# for running on Google Colab
# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{date_time}"
# NETWORK_SNAPSHOTS_BASE_FOLDER = "/content/drive/MyDrive/BFA/snapshots"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg_Simulation")

# debug attributes
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0, help='only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0, help='currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.')
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='resume from the path of saved network_snapshots; only provide the date')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='save frequency of the network_snapshot')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2, help='in case of saving space, keep only the recent specified number of snapshops; 0 means keep all')

# FL attributes
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD", help='optimizer to be used, by default implementing stochastic gradient descent')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to devices')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=100, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='numer of the devices in the simulation network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across devices when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
parser.add_argument('-nv', '--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5, help='local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified')

# blockchain system consensus attributes
parser.add_argument('-ur', '--unit_reward', type=int, default=1, help='unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6, help="a worker or validator device is kicked out of the device's peer list(put in black list) if it's identified as malicious for this number of rounds")
parser.add_argument('-lo', '--lazy_worker_knock_out_rounds', type=int, default=10, help="a worker device is kicked out of the device's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="if set to 0, meaning miners are using PoS")

# blockchain FL validator/miner restriction tuning parameters
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0, help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each device will just perform same amount(-le) of epochs per round like in FedAvg paper")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0, help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"), help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS")
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0, help="a threshold value of accuracy difference to determine malicious worker")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0, help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_validator_on', type=int, default=0, help="let malicious validator flip voting result")


# distributed system attributes
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds a device is online')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1, help="This variable is used to simulate transmission delay. Default value 1 means every device is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a device will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0, help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1, help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

# simulation attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help="hard assign number of roles in the network, order by worker, validator and miner. e.g. 12,5,3 assign 12 workers, 5 validators and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1, help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-cs', '--check_signature', type=int, default=1, help='if set to 0, all signatures are assumed to be verified to save execution time')

# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')

if __name__=="__main__":

	# create logs/ if not exists
	if not os.path.exists('logs'):
		os.makedirs('logs')

	# get arguments
	args = parser.parse_args()
	args = args.__dict__
	
	# detect CUDA
	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# pre-define system variables
	latest_round_num = 0

	''' If network_snapshot is specified, continue from left '''
	if args['resume_path']:
		if not args['save_network_snapshots']:
			print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
		network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
		latest_network_snapshot_file_name = sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')], key = lambda fn: int(fn.split('_')[-1]) , reverse=True)[0]
		print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
		print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
		latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
		devices_in_network = pickle.load(open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
		devices_list = list(devices_in_network.devices_set.values())
		log_files_folder_path = f"logs/{args['resume_path']}"
		# for colab
		# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
		# original arguments file
		args_used_file = f"{log_files_folder_path}/args_used.txt"
		file = open(args_used_file,"r") 
		log_whole_text = file.read() 
		lines_list = log_whole_text.split("\n")
		for line in lines_list:
			# abide by the original specified rewards
			if line.startswith('--unit_reward'):
				rewards = int(line.split(" ")[-1])
			# get number of roles
			if line.startswith('--hard_assign'):
				roles_requirement = line.split(" ")[-1].split(',')
			# get mining consensus
			if line.startswith('--pow_difficulty'):
				mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
		# determine roles to assign
		try:
			workers_needed = int(roles_requirement[0])
		except:
			workers_needed = 1
		try:
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
		except:
			miners_needed = 1
	else:
		''' SETTING UP FROM SCRATCH'''
		
		# 0. create log_files_folder_path if not resume
		os.mkdir(log_files_folder_path)

		# 1. save arguments used
		with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
			f.write("Command line arguments used -\n")
			f.write(' '.join(sys.argv[1:]))
			f.write("\n\nAll arguments used -\n")
			for arg_name, arg in args.items():
				f.write(f'\n--{arg_name} {arg}')
				
		# 2. create network_snapshot folder
		if args['save_network_snapshots']:
			network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
			os.mkdir(network_snapshot_save_path)

		# 3. assign system variables
		# for demonstration purposes, this reward is for every rewarded action
		rewards = args["unit_reward"]
		
		# 4. get number of roles needed in the network
		roles_requirement = args['hard_assign'].split(',')
		# determine roles to assign
		try:
			workers_needed = int(roles_requirement[0])
		except:
			workers_needed = 1
		try:
			validators_needed = int(roles_requirement[1])
		except:
			validators_needed = 1
		try:
			miners_needed = int(roles_requirement[2])
		except:
			miners_needed = 1

		# 5. check arguments eligibility

		num_devices = args['num_devices']
		num_malicious = args['num_malicious']
		
		if num_devices < workers_needed + miners_needed + validators_needed:
			sys.exit("ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")

		if num_devices < 3:
			sys.exit("ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

		
		if num_malicious:
			if num_malicious > num_devices:
				sys.exit("ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network")
			else:
				print(f"Malicious nodes vs total devices set to {num_malicious}/{num_devices} = {(num_malicious/num_devices)*100:.2f}%")

		# 6. create neural net based on the input model name
		net = None
		if args['model_name'] == 'mnist_2nn':
			net = Mnist_2NN()
		elif args['model_name'] == 'mnist_cnn':
			net = Mnist_CNN()

		# 7. assign GPU(s) if available to the net, otherwise CPU
		# os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
		if torch.cuda.device_count() > 1:
			net = torch.nn.DataParallel(net)
		print(f"{torch.cuda.device_count()} GPUs are available to use!")
		net = net.to(dev)

		# 8. set loss_function
		loss_func = F.cross_entropy

		# 9. create devices in the network
		devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size = args['batchsize'], learning_rate =  args['learning_rate'], loss_func = loss_func, opti = args['optimizer'], num_devices=num_devices, network_stability=args['network_stability'], net=net, dev=dev, knock_out_rounds=args['knock_out_rounds'], lazy_worker_knock_out_rounds=args['lazy_worker_knock_out_rounds'], shard_test_data=args['shard_test_data'], miner_acception_wait_time=args['miner_acception_wait_time'], miner_accepted_transactions_size_limit=args['miner_accepted_transactions_size_limit'], validator_threshold=args['validator_threshold'], pow_difficulty=args['pow_difficulty'], even_link_speed_strength=args['even_link_speed_strength'], base_data_transmission_speed=args['base_data_transmission_speed'], even_computation_power=args['even_computation_power'], malicious_updates_discount=args['malicious_updates_discount'], num_malicious=num_malicious, noise_variance=args['noise_variance'], check_signature=args['check_signature'], not_resync_chain=args['destroy_tx_in_block'])
		del net
		devices_list = list(devices_in_network.devices_set.values())

		# 10. register devices and initialize global parameterms
		for device in devices_list:
			# set initial global weights
			device.init_global_parameters()
			# helper function for registration simulation - set devices_list and aio
			device.set_devices_dict_and_aio(devices_in_network.devices_set, args["all_in_one"])
			# simulate peer registration, with respect to device idx order
			device.register_in_the_network()
		# remove its own from peer list if there is
		for device in devices_list:
			device.remove_peers(device)

		# 11. build logging files/database path
		# create log files
		open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'w').close()
		open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'w').close()
		open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'w').close()
		open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
		# open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
		# open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
		open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'w').close()

		# 12. setup the mining consensus
		mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

	# create malicious worker identification database
	conn = sqlite3.connect(f'{log_files_folder_path}/malicious_wokrer_identifying_log.db')
	conn_cursor = conn.cursor()
	conn_cursor.execute("""CREATE TABLE if not exists  malicious_workers_log (
	device_seq text,
	if_malicious integer,
	correctly_identified_by text,
	incorrectly_identified_by text,
	in_round integer,
	when_resyncing text
	)""")

	# VBFL starts here
	for comm_round in range(latest_round_num + 1, args['max_num_comm']+1):
		# create round specific log folder
		log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
		if os.path.exists(log_files_folder_path_comm_round):
			print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
			shutil.rmtree(log_files_folder_path_comm_round)
		os.mkdir(log_files_folder_path_comm_round)
		# free cuda memory
		if dev == torch.device("cuda"):
			with torch.cuda.device('cuda'):
				torch.cuda.empty_cache()
		print(f"\nCommunication round {comm_round}")
		comm_round_start_time = time.time()
		# (RE)ASSIGN ROLES
		workers_to_assign = workers_needed
		miners_to_assign = miners_needed
		validators_to_assign = validators_needed
		workers_this_round = []
		miners_this_round = []
		validators_this_round = []
		random.shuffle(devices_list)
		for device in devices_list:
			if workers_to_assign:
				device.assign_worker_role()
				workers_to_assign -= 1
			elif miners_to_assign:
				device.assign_miner_role()
				miners_to_assign -= 1
			elif validators_to_assign:
				device.assign_validator_role()
				validators_to_assign -= 1
			else:
				device.assign_role()
			if device.return_role() == 'worker':
				workers_this_round.append(device)
			elif device.return_role() == 'miner':
				miners_this_round.append(device)
			else:
				validators_this_round.append(device)
			# determine if online at the beginning (essential for step 1 when worker needs to associate with an online device)
			device.online_switcher()

		''' DEBUGGING CODE '''
		if args['verbose']:

			# show devices initial chain length and if online
			for device in devices_list:
				if device.is_online():
					print(f'{device.return_idx()} {device.return_role()} online - ', end='')
				else:
					print(f'{device.return_idx()} {device.return_role()} offline - ', end='')
				# debug chain length
				print(f"chain length {device.return_blockchain_object().return_chain_length()}")
		
			# show device roles
			print(f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
			print("\nworkers this round are")
			for worker in workers_this_round:
				print(f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
			print("\nminers this round are")
			for miner in miners_this_round:
				print(f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
			print("\nvalidators this round are")
			for validator in validators_this_round:
				print(f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
			print()

			# show peers with round number
			print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
			for device_seq, device in devices_in_network.devices_set.items():
				peers = device.return_peers()
				print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
				for peer in peers:
					print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
				print()
			print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

		''' DEBUGGING CODE ENDS '''

		# re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought device may go offline somewhere in the previous round and their variables were not therefore reset
		for miner in miners_this_round:
			miner.miner_reset_vars_for_new_round()
		for worker in workers_this_round:
			worker.worker_reset_vars_for_new_round()
		for validator in validators_this_round:
			validator.validator_reset_vars_for_new_round()

		# DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for worker, this will affect the order of dataset portions to be trained)
		random.shuffle(workers_this_round)
		random.shuffle(miners_this_round)
		random.shuffle(validators_this_round)
		
		''' workers, validators and miners take turns to perform jobs '''

		print(''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')
		for worker_iter in range(len(workers_this_round)):
			worker = workers_this_round[worker_iter]
			# resync chain(block could be dropped due to fork from last round)
			if worker.resync_chain(mining_consensus):
				worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
			# worker (should) perform local update and associate
			print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} will associate with a validator and a miner, if online...")
			# worker associates with a miner to accept finally mined block
			if worker.online_switcher():
				associated_miner = worker.associate_with_device("miner")
				if associated_miner:
					associated_miner.add_device_to_association(worker)
				else:
					print(f"Cannot find a qualified miner in {worker.return_idx()} peer list.")
			# worker associates with a validator to send worker transactions
			if worker.online_switcher():
				associated_validator = worker.associate_with_device("validator")
				if associated_validator:
					associated_validator.add_device_to_association(worker)
				else:
					print(f"Cannot find a qualified validator in {worker.return_idx()} peer list.")
		
		print(''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			# resync chain
			if validator.resync_chain(mining_consensus):
				validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			# associate with a miner to send post validation transactions
			if validator.online_switcher():
				associated_miner = validator.associate_with_device("miner")
				if associated_miner:
					associated_miner.add_device_to_association(validator)
				else:
					print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")
			# validator accepts local updates from its workers association
			associated_workers = list(validator.return_associated_workers())
			if not associated_workers:
				print(f"No workers are associated with validator {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} for this communication round.")
				continue
			validator_link_speed = validator.return_link_speed()
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online...")
			# records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
			records_dict = dict.fromkeys(associated_workers, None)
			for worker, _ in records_dict.items():
				records_dict[worker] = {}
			# used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
			transaction_arrival_queue = {}
			# workers local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
			if args['miner_acception_wait_time']:
				print(f"miner wati time is specified as {args['miner_acception_wait_time']} seconds. let each worker do local_updates till time limit")
				for worker_iter in range(len(associated_workers)):
					worker = associated_workers[worker_iter]
					if not worker.return_idx() in validator.return_black_list():
						# TODO here, also add print() for below miner's validators
						print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')	 
						total_time_tracker = 0
						update_iter = 1
						worker_link_speed = worker.return_link_speed()
						lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
						while total_time_tracker < validator.return_miner_acception_wait_time():
							# simulate the situation that worker may go offline during model updates transmission to the validator, based on per transaction
							if worker.online_switcher():
								local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round)
								unverified_transaction = worker.return_local_updates_and_signature(comm_round)
								# size in bytes, usually around 35000 bytes per transaction
								unverified_transactions_size = getsizeof(str(unverified_transaction))
								transmission_delay = unverified_transactions_size/lower_link_speed
								if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
									# last transaction sent passes the acception time window
									break
								records_dict[worker][update_iter] = {}
								records_dict[worker][update_iter]['local_update_time'] = local_update_spent_time
								records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
								records_dict[worker][update_iter]['local_update_unverified_transaction'] = unverified_transaction
								records_dict[worker][update_iter]['local_update_unverified_transaction_size'] = unverified_transactions_size
								if update_iter == 1:
									total_time_tracker = local_update_spent_time + transmission_delay
								else:
									total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + local_update_spent_time + transmission_delay
								records_dict[worker][update_iter]['arrival_time'] = total_time_tracker
								if validator.online_switcher():
									# accept this transaction only if the validator is online
									print(f"validator {validator.return_idx()} has accepted this transaction.")
									transaction_arrival_queue[total_time_tracker] = unverified_transaction
								else:
									print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
							else:
								# worker goes offline and skip updating for one transaction, wasted the time of one update and transmission
								wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time(args['optimizer'])
								wasted_update_params_size = getsizeof(str(wasted_update_params))
								wasted_transmission_delay = wasted_update_params_size/lower_link_speed
								if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
									# wasted transaction "arrival" passes the acception time window
									break
								records_dict[worker][update_iter] = {}
								records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
								if update_iter == 1:
									total_time_tracker = wasted_update_time + wasted_transmission_delay
									print(f"worker goes offline and wasted {total_time_tracker} seconds for a transaction")
								else:
									total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + wasted_update_time + wasted_transmission_delay
							update_iter += 1
			else:
				 # did not specify wait time. every associated worker perform specified number of local epochs
				for worker_iter in range(len(associated_workers)):
					worker = associated_workers[worker_iter]
					if not worker.return_idx() in validator.return_black_list():
						print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')	 
						if worker.online_switcher():
							local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round, local_epochs=args['default_local_epochs'])
							worker_link_speed = worker.return_link_speed()
							lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
							unverified_transaction = worker.return_local_updates_and_signature(comm_round)
							unverified_transactions_size = getsizeof(str(unverified_transaction))
							transmission_delay = unverified_transactions_size/lower_link_speed
							if validator.online_switcher():
								transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
								print(f"validator {validator.return_idx()} has accepted this transaction.")
							else:
								print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
						else:
							print(f"worker {worker.return_idx()} offline and unable do local updates")
					else:
						print(f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
			validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
			# in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
			validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))
			
			# broadcast to other validators
			if transaction_arrival_queue:
				validator.validator_broadcast_worker_transactions()
			else:
				print("No transactions have been received by this validator, probably due to workers and/or validators offline or timeout while doing local updates or transmitting updates, or all workers are in validator's black list.")


		print(''' Step 2.5 - with the broadcasted workers transactions, validators decide the final transaction arrival order \n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_worker_transactions()
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				self_validator_link_speed = validator.return_link_speed()
				for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
					broadcasting_validator_link_speed = broadcasting_validator_record['source_validator_link_speed']
					lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed
					for arrival_time_at_broadcasting_validator, broadcasted_transaction in broadcasting_validator_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
			else:
				print(f"validator {validator.return_idx()} {validator_iter+1}/{len(validators_this_round)} did not receive any broadcasted worker transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			final_transactions_arrival_queue = sorted({**validator.return_unordered_arrival_time_accepted_worker_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
			print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")


		print(''' Step 3 - validators do self and cross-validation(validate local updates from workers) by the order of transaction arrival time.\n''')
		for validator_iter in range(len(validators_this_round)):
			validator = validators_this_round[validator_iter]
			final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
			if final_transactions_arrival_queue:
				# validator asynchronously does one epoch of update and validate on its own test set
				local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(args['optimizer'])
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is validating received worker transactions...")
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					if validator.online_switcher():
						# validation won't begin until validator locally done one epoch of update and validation(worker transactions will be queued)
						if arrival_time < local_validation_time:
							arrival_time = local_validation_time
						validation_time, post_validation_unconfirmmed_transaction = validator.validate_worker_transaction(unconfirmmed_transaction, rewards, log_files_folder_path, comm_round, args['malicious_validator_on'])
						if validation_time:
							validator.add_post_validation_transaction_to_queue((arrival_time + validation_time, validator.return_link_speed(), post_validation_unconfirmmed_transaction))
							print(f"A validation process has been done for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}")
					else:
						print(f"A validation process is skipped for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()} due to validator offline.")
			else:
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} did not receive any transaction from worker or validator in this round.")

		print(''' Step 4 - validators send post validation transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			# resync chain
			if miner.resync_chain(mining_consensus):
				miner.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' post-validation transactions...")
			associated_validators = list(miner.return_associated_validators())
			if not associated_validators:
				print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
				continue
			self_miner_link_speed = miner.return_link_speed()
			validator_transactions_arrival_queue = {}
			for validator_iter in range(len(associated_validators)):
				validator = associated_validators[validator_iter]
				print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction...")
				post_validation_transactions_by_validator = validator.return_post_validation_transactions_queue()
				post_validation_unconfirmmed_transaction_iter = 1
				for (validator_sending_time, source_validator_link_spped, post_validation_unconfirmmed_transaction) in post_validation_transactions_by_validator:
					if validator.online_switcher() and miner.online_switcher():
						lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
						transmission_delay = getsizeof(str(post_validation_unconfirmmed_transaction))/lower_link_speed
						validator_transactions_arrival_queue[validator_sending_time + transmission_delay] = post_validation_unconfirmmed_transaction
						print(f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}")
					else:
						print(f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of devices or both offline.")
					post_validation_unconfirmmed_transaction_iter += 1
			miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
			miner.miner_broadcast_validator_transactions()

		print(''' Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order\n ''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
			self_miner_link_speed = miner.return_link_speed()
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
			accepted_broadcasted_transactions_arrival_queue = {}
			if accepted_broadcasted_validator_transactions:
				# calculate broadcasted transactions arrival time
				for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
					broadcasting_miner_link_speed = broadcasting_miner_record['source_device_link_speed']
					lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed
					for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record['broadcasted_transactions'].items():
						transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
						accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
			else:
				print(f"miner {miner.return_idx()} {miner_iter+1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round.")
			# mix the boardcasted transactions with the direct accepted transactions
			final_transactions_arrival_queue = sorted({**miner.return_unordered_arrival_time_accepted_validator_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
			miner.set_candidate_transactions_for_final_mining_queue(final_transactions_arrival_queue)
			print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions.")
		
		print(''' Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time and record the transactions in the candidate block according to the limit size. Also mine and propagate the block.\n''')
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			final_transactions_arrival_queue = miner.return_final_candidate_transactions_mining_queue()
			valid_validator_sig_candidate_transacitons = []
			invalid_validator_sig_candidate_transacitons = []
			begin_mining_time = 0
			new_begin_mining_time = begin_mining_time
			if final_transactions_arrival_queue:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is verifying received validator transactions...")
				time_limit = miner.return_miner_acception_wait_time()
				size_limit = miner.return_miner_accepted_transactions_size_limit()
				for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
					if miner.online_switcher():
						if time_limit:
							if arrival_time > time_limit:
								break
						if size_limit:
							if getsizeof(str(valid_validator_sig_candidate_transacitons+invalid_validator_sig_candidate_transacitons)) > size_limit:
								break
						# verify validator signature of this transaction
						verification_time, is_validator_sig_valid = miner.verify_validator_transaction(unconfirmmed_transaction)
						if verification_time:
							if is_validator_sig_valid:
								validator_info_this_tx = {
								'validator': unconfirmmed_transaction['validation_done_by'],
								'validation_rewards': unconfirmmed_transaction['validation_rewards'],
								'validation_time': unconfirmmed_transaction['validation_time'],
								'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
								'validator_signature': unconfirmmed_transaction['validator_signature'],
								'update_direction': unconfirmmed_transaction['update_direction'],
								'miner_device_idx': miner.return_idx(),
								'miner_verification_time': verification_time,
								'miner_rewards_for_this_tx': rewards}
								# validator's transaction signature valid
								found_same_worker_transaction = False
								for valid_validator_sig_candidate_transaciton in valid_validator_sig_candidate_transacitons:
									if valid_validator_sig_candidate_transaciton['worker_signature'] == unconfirmmed_transaction['worker_signature']:
										found_same_worker_transaction = True
										break
								if not found_same_worker_transaction:
									valid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
									del valid_validator_sig_candidate_transaciton['validation_done_by']
									del valid_validator_sig_candidate_transaciton['validation_rewards']
									del valid_validator_sig_candidate_transaciton['update_direction']
									del valid_validator_sig_candidate_transaciton['validation_time']
									del valid_validator_sig_candidate_transaciton['validator_rsa_pub_key']
									del valid_validator_sig_candidate_transaciton['validator_signature']
									valid_validator_sig_candidate_transaciton['positive_direction_validators'] = []
									valid_validator_sig_candidate_transaciton['negative_direction_validators'] = []
									valid_validator_sig_candidate_transacitons.append(valid_validator_sig_candidate_transaciton)
								if unconfirmmed_transaction['update_direction']:
									valid_validator_sig_candidate_transaciton['positive_direction_validators'].append(validator_info_this_tx)
								else:
									valid_validator_sig_candidate_transaciton['negative_direction_validators'].append(validator_info_this_tx)
								transaction_to_sign = valid_validator_sig_candidate_transaciton
							else:
								# validator's transaction signature invalid
								invalid_validator_sig_candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
								invalid_validator_sig_candidate_transaciton['miner_verification_time'] = verification_time
								invalid_validator_sig_candidate_transaciton['miner_rewards_for_this_tx'] = rewards
								invalid_validator_sig_candidate_transacitons.append(invalid_validator_sig_candidate_transaciton)
								transaction_to_sign = invalid_validator_sig_candidate_transaciton
							# (re)sign this candidate transaction
							signing_time = miner.sign_candidate_transaction(transaction_to_sign)
							new_begin_mining_time = arrival_time + verification_time + signing_time
					else:
						print(f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline.")
						new_begin_mining_time = arrival_time
					begin_mining_time = new_begin_mining_time if new_begin_mining_time > begin_mining_time else begin_mining_time
				transactions_to_record_in_block = {}
				transactions_to_record_in_block['valid_validator_sig_transacitons'] = valid_validator_sig_candidate_transacitons
				transactions_to_record_in_block['invalid_validator_sig_transacitons'] = invalid_validator_sig_candidate_transacitons
				# put transactions into candidate block and begin mining
				# block index starts from 1
				start_time_point = time.time()
				candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, transactions=transactions_to_record_in_block, miner_rsa_pub_key=miner.return_rsa_pub_key())
				# mine the block
				miner_computation_power = miner.return_computation_power()
				if not miner_computation_power:
					block_generation_time_spent = float('inf')
					miner.set_block_generation_time_point(float('inf'))
					print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
					continue
				recorded_transactions = candidate_block.return_transactions()
				if recorded_transactions['valid_validator_sig_transacitons'] or recorded_transactions['invalid_validator_sig_transacitons']:
					print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the block...")
					# return the last block and add previous hash
					last_block = miner.return_blockchain_object().return_last_block()
					if last_block is None:
						# will mine the genesis block
						candidate_block.set_previous_block_hash(None)
					else:
						candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
					# mine the candidate block by PoW, inside which the block_hash is also set
					mined_block = miner.mine_block(candidate_block, rewards)
				else:
					print("No transaction to mine for this block.")
					continue
				# unfortunately may go offline while propagating its block
				if miner.online_switcher():
					# sign the block
					miner.sign_block(mined_block)
					miner.set_mined_block(mined_block)
					# record mining time
					block_generation_time_spent = (time.time() - start_time_point)/miner_computation_power
					miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
					print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
					# immediately propagate the block
					miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
				else:
					print(f"Unfortunately, {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated.")
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round.")

		print(''' Step 6 - miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated devices to download this block''')
		forking_happened = False
		# comm_round_block_gen_time regarded as the time point when the winning miner mines its block, calculated from the beginning of the round. If there is forking in PoW or rewards info out of sync in PoS, this time is the avg time point of all the appended time by any device
		comm_round_block_gen_time = []
		for miner_iter in range(len(miners_this_round)):
			miner = miners_this_round[miner_iter]
			unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
			# add self mined block to the processing queue and sort by time
			this_miner_mined_block = miner.return_mined_block()
			if this_miner_mined_block:
				unordered_propagated_block_processing_queue[miner.return_block_generation_time_point()] = this_miner_mined_block
			ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
			if ordered_all_blocks_processing_queue:
				if mining_consensus == 'PoW':
					print("\nselect winning block based on PoW")
					# abort mining if propagated block is received
					print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block...")
					for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
						verified_block, verification_time = miner.verify_block(block_to_verify, block_to_verify.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == miner.return_idx():
								print(f"Miner {miner.return_idx()} is adding its own mined block.")
							else:
								print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
							if miner.online_switcher():
								miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
							if miner.return_the_added_block():
								# requesting devices in its associations to download this block
								miner.request_to_download(verified_block, block_arrival_time + verification_time)
								break								
				else:
					# PoS
					candidate_PoS_blocks = {}
					print("select winning block based on PoS")
					# filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
					for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
						if block_arrival_time < args['miner_pos_propagated_block_wait_time']:
							candidate_PoS_blocks[devices_in_network.devices_set[block_to_verify.return_mined_by()].return_stake()] = block_to_verify
					high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
					# for PoS, requests every device in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
					for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
						verified_block, verification_time = miner.verify_block(PoS_candidate_block, PoS_candidate_block.return_mined_by())
						if verified_block:
							block_mined_by = verified_block.return_mined_by()
							if block_mined_by == miner.return_idx():
								print(f"Miner {miner.return_idx()} with stake {stake} is adding its own mined block.")
							else:
								print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
							if miner.online_switcher():
								miner.add_block(verified_block)
							else:
								print(f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
							if miner.return_the_added_block():
								# requesting devices in its associations to download this block
								miner.request_to_download(verified_block, block_arrival_time + verification_time)
								break
				miner.add_to_round_end_time(block_arrival_time + verification_time)
			else:
				print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet.")
		# CHECK FOR FORKING
		added_blocks_miner_set = set()
		for device in devices_list:
			the_added_block = device.return_the_added_block()
			if the_added_block:
				print(f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
				added_blocks_miner_set.add(the_added_block.return_mined_by())
				block_generation_time_point = devices_in_network.devices_set[the_added_block.return_mined_by()].return_block_generation_time_point()
				# commented, as we just want to plot the legitimate block gen time, and the wait time is to avoid forking. Also the logic is wrong. Should track the time to the slowest worker after its global model update
				# if mining_consensus == 'PoS':
				# 	if args['miner_pos_propagated_block_wait_time'] != float("inf"):
				# 		block_generation_time_point += args['miner_pos_propagated_block_wait_time']
				comm_round_block_gen_time.append(block_generation_time_point)
		if len(added_blocks_miner_set) > 1:
			print("WARNING: a forking event just happened!")
			forking_happened = True
			with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file:
				file.write(f"Forking in round {comm_round}\n")
		else:
			print("No forking event happened.")
			

		print(''' Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious nodes identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round''')
		all_devices_round_ends_time = []
		for device in devices_list:
			if device.return_the_added_block() and device.online_switcher():
				# collect usable updated params, malicious nodes identification, get rewards and do local udpates
				processing_time = device.process_block(device.return_the_added_block(), log_files_folder_path, conn, conn_cursor)
				device.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
				device.add_to_round_end_time(processing_time)
				all_devices_round_ends_time.append(device.return_round_end_time())

		print(''' Logging Accuracies by Devices ''')
		for device in devices_list:
			device.accuracy_this_round = device.validate_model_weights()
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if device.return_is_malicious() else "B"
				file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.accuracy_this_round}\n")

		# logging time, mining_consensus and forking
		# get the slowest device end time
		comm_round_spent_time = time.time() - comm_round_start_time
		with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
			# corner case when all miners in this round are malicious devices so their blocks are rejected
			try:
				comm_round_block_gen_time = max(comm_round_block_gen_time)
				file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
			except:
				no_block_msg = "No valid block has been generated this round."
				print(no_block_msg)
				file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
				with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file2:
					# TODO this may be caused by "no transaction to mine" for the miner. Forgot to check for block miner's maliciousness in request_to_downlaod()
					file2.write(f"No valid block in round {comm_round}\n")
			try:
				slowest_round_ends_time = max(all_devices_round_ends_time)
				file.write(f"slowest_device_round_ends_time: {slowest_round_ends_time}\n")
			except:
				# corner case when all transactions are rejected by miners
				file.write("slowest_device_round_ends_time: No valid block has been generated this round.\n")
				with open(f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", 'a') as file2:
					no_valid_block_msg = f"No valid block in round {comm_round}\n"
					if file2.readlines()[-1] != no_valid_block_msg:
						file2.write(no_valid_block_msg)
			file.write(f"mining_consensus: {mining_consensus} {args['pow_difficulty']}\n")
			file.write(f"forking_happened: {forking_happened}\n")
			file.write(f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n")
		conn.commit()

		# if no forking, log the block miner
		if not forking_happened:
			legitimate_block = None
			for device in devices_list:
				legitimate_block = device.return_the_added_block()
				if legitimate_block is not None:
					# skip the device who's been identified malicious and cannot get a block from miners
					break
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				if legitimate_block is None:
					file.write("block_mined_by: no valid block generated this round\n")
				else:
					block_mined_by = legitimate_block.return_mined_by()
					is_malicious_node = "M" if devices_in_network.devices_set[block_mined_by].return_is_malicious() else "B"
					file.write(f"block_mined_by: {block_mined_by} {is_malicious_node}\n")
		else:
			with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
				file.write(f"block_mined_by: Forking happened\n")

		print(''' Logging Stake by Devices ''')
		for device in devices_list:
			device.accuracy_this_round = device.validate_model_weights()
			with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
				is_malicious_node = "M" if device.return_is_malicious() else "B"
				file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n")

		# a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
		if args['destroy_tx_in_block']:
			for device in devices_list:
				last_block = device.return_blockchain_object().return_last_block()
				if last_block:
					last_block.free_tx()

		# save network_snapshot if reaches save frequency
		if args['save_network_snapshots'] and (comm_round == 1 or comm_round % args['save_freq'] == 0):
			if args['save_most_recent']:
				paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
				if len(paths) > args['save_most_recent']:
					for _ in range(len(paths) - args['save_most_recent']):
						# make it 0 byte as os.remove() moves file to the bin but may still take space
						# https://stackoverflow.com/questions/53028607/how-to-remove-the-file-from-trash-in-drive-in-colab
						open(paths[_], 'w').close() 
						os.remove(paths[_])
			snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
			print(f"Saving network snapshot to {snapshot_file_path}")
			pickle.dump(devices_in_network, open(snapshot_file_path, "wb"))