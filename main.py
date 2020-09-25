# fedavg from https://github.com/WHDY/FedAvg/
# TODO DELETE ALL receive_rewards() and only do it after a block is appended! AND whenever resync chain recalculate rewards!!!
# TODO redistribute offline() based on very transaction, not at the beginning of every loop
# TODO when accepting transactions, check comm_round must be in the same, that is to assume by default they only accept the transactions that are in the same round, so the final block contains only updates from the same round
# TODO subnets - potentially resolved by let each miner sends block 
# TODO let's not remove peers because of offline as peers may go back online later, instead just check if they are online or offline. Only remove peers if they are malicious. In real distributed network, remove peers if cannot reach for a long period of time

# PoS also uses PoW resync chain - the longest chain

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
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DevicesInNetwork
from Block import Block
from Blockchain import Blockchain

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg_Simulation")

# program attributes
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=0, help='print verbose debug log')

# FL attributes
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=1000, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-nd', '--num_devices', type=int, default=100, help='numer of the devices in the simulation network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across devices when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
parser.add_argument('-le', '--default_local_epochs', type=int, default=3, help='local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified')

# blockchain system consensus attributes
parser.add_argument('-gr', '--general_rewards', type=int, default=1, help='rewards for providing data, verification of one transaction, mining and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=5, help='a device is kicked out of the network if its accuracy shows decreasing for the number of rounds recorded by a winning validator')
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="if set to 0, meaning miners are using PoS")

# blockchain FL restriction attributes
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0, help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each device will just perform same amount(-le) of epochs per round like in FedAvg paper")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0, help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")

# debug attributes
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='hard assign number of roles in the network, order by worker, miner and validator')
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds a device is online')
parser.add_argument('-aio', '--all_in_one', type=int, default=0, help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1, help="This variable is used to simulate transmission delay. Default value 1 means every device is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a device will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0, help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed, which further determines the transmission delay - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1, help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")


# parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')
# parser.add_argument('-vh', '--validator_threshold', type=float, default=0.1, help="a threshold value of accuracy difference to determine malicious worker")

if __name__=="__main__":

    ''' SETTING UP '''
    # 1. set program running time for logging purpose
    date_time = datetime.now().strftime("%d%m%Y_%H%M%S")

    args = parser.parse_args()
    args = args.__dict__

    # 2. for demonstration purposes, this reward is for every rewarded action
    rewards = args["general_rewards"]

    # 3. get number of roles needed in the network
    roles_requirement = args['hard_assign'].split(',')
    
    try:
        workers_needed = int(roles_requirement[0])
    except:
        workers_needed = 1
    
    try:
        miners_needed = int(roles_requirement[1])
    except:
        miners_needed = 1
    
    try:
        validators_needed = int(roles_requirement[2])
    except:
        validators_needed = 1

    # 4. check arguments eligibility

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

    # 5. create neural net based on the input model name
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    # 6. assign GPUs if available and prepare the net
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    print(f"{torch.cuda.device_count()} GPUs are available to use!")
    net = net.to(dev)

    # 6. set loss_function and optimizer
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    # 7. create devices in the network
    devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size = args['batchsize'], loss_func = loss_func, opti = opti, num_devices=num_devices, network_stability=args['network_stability'], net=net, dev=dev, knock_out_rounds=args['knock_out_rounds'], shard_test_data=args['shard_test_data'], miner_acception_wait_time=args['miner_acception_wait_time'], miner_accepted_transactions_size_limit=args['miner_accepted_transactions_size_limit'], pow_difficulty=args['pow_difficulty'], even_link_speed_strength=args['even_link_speed_strength'], base_data_transmission_speed=args['base_data_transmission_speed'], even_computation_power=args['even_computation_power'], num_malicious=num_malicious)
    devices_list = list(devices_in_network.devices_set.values())

    # 8. register devices and initialize global parameterms
    for device in devices_list:
        # set initial global weights
        device.init_global_parameters()
        # helper function for registration simulation - set devices_list and aio
        device.set_devices_list_and_aio(devices_list, args["all_in_one"])
        # simulate peer registration, with respect to device idx order
        device.register_in_the_network()
    # remove its own from peer list if there is
    for device in devices_list:
        device.remove_peers(device)

    # 9. build a dict to record device accuracies accross rounds
    devices_accuracies_records = {}
    for device_seq, device in devices_in_network.devices_set.items():
        devices_accuracies_records[device_seq] = {}

    # 10. make chechpoint save path
    # if not os.path.isdir(args['save_path']):
    #     os.mkdir(args['save_path'])

    # BlockFL starts here
    for comm_round in range(1, args['max_num_comm']+1):
        print(f"\nCommunication round {comm_round}")

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
            # REMOVED AS OFFLINE PEERS WILL NOT BE DISCARDED ANY MORE - register with a peer in case peer list empty


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

        print(''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) ''')
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            # PoW resync chain - can perform at any time
            if worker.pow_resync_chain():
                worker.update_model_after_chain_resync()
            # worker (should) perform local update and associate
            print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} associating with a validator and a miner, if online...")
            # worker associates with a miner to accept finally mined block
            if worker.online_switcher():
                associated_miner = worker.associate_with_device("miner")
                if not associated_miner:
                    print(f"Cannot find a qualified miner in {worker.return_idx()} peer list.")
                    continue
                else:
                    print(f"Worker {worker.return_idx()} associates with miner {associated_miner.return_idx()}")
                associated_miner.add_device_to_association(worker)
            # worker associates with a validator
            if worker.online_switcher():
                associated_validator = worker.associate_with_device("validator")
                if not associated_validator:
                    print(f"Cannot find a qualified validator in {worker.return_idx()} peer list.")
                    continue
                else:
                    print(f"Worker {worker.return_idx()} associates with validator {associated_validator.return_idx()}")
                associated_validator.add_device_to_association(worker)

        
        print(''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            # PoW resync chain
            if validator.pow_resync_chain():
                validator.update_model_after_chain_resync()
            # validator accepts local updates from its workers association
            validator_link_speed = validator.return_link_speed()
            print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online...")
            associated_workers = validator.return_associated_workers()
            if not associated_workers:
                print(f"No workers are associated with validator {validator.return_idx()} for this communication round.")
                continue
            # records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}
            # used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
            transaction_arrival_queue = {}
            # workers local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
            for worker in associated_workers:
                if not worker.return_idx() in validator.return_black_list():
                    if args['miner_acception_wait_time']:
                        print(f"miner wati time has specified as {args['miner_acception_wait_time']} seconds. let each worker do local_updates till time limit")             
                        total_time_tracker = 0
                        update_iter = 1
                        worker_link_speed = worker.return_link_speed()
                        lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                        while total_time_tracker < validator.return_miner_acception_wait_time():
                            # simulate the situation that worker may go offline during model updates transmission to the validator, based on per transaction
                            if worker.online_switcher():
                                local_update_spent_time = worker.worker_local_update(rewards)
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
                                if validators.online_switcher():
                                    # accept this transaction only if the validator is online
                                    transaction_arrival_queue[total_time_tracker] = unverified_transaction
                            else:
                                # TODO - add print()
                                # worker goes offline and skip updating for one transaction, wasted the time of one update and transmission
                                wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time()
                                wasted_update_params_size = getsizeof(str(wasted_update_params))
                                wasted_transmission_delay = wasted_update_params_size/lower_link_speed
                                if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
                                    # wasted transaction "arrival" passes the acception time window
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = wasted_update_time + wasted_transmission_delay
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + wasted_update_time + wasted_transmission_delay
                            update_iter += 1
                    else:
                        # did not specify wait time. every associated worker perform specified number of local epochs
                        for worker in associated_workers:
                            if worker.online_switcher():
                                local_update_spent_time = worker.worker_local_update(rewards, local_epochs=args['default_local_epochs'])
                                worker_link_speed = worker.return_link_speed()
                                lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                                unverified_transaction = worker.return_local_updates_and_signature(comm_round)
                                unverified_transactions_size = getsizeof(str(unverified_transaction))
                                transmission_delay = unverified_transactions_size/lower_link_speed
                                if validators.online_switcher():
                                    transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
                else:
                    print(f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")

            
                        
                # workers local updates done by time limit
                # begin self-verification, parimarily due to validated_size_limit may be specified
                # sort arrival time of all possible transactions
                #ordered_transaction_arrival_queue = sorted(transaction_arrival_queue.items())
                #validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
                
                    # else:
                    #     print(f"transaction arrived at {transaction_record[0]}s by worker {by_worker.return_idx()} at local epoch {epoch_seq} did not pass the signature verification.")
                    
            

                #ordered_transaction_arrival_queue = sorted(transaction_arrival_queue.items())
            validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
            # in case validator off line for accepting broadcasted transactions but can later back on line to validate the transactions itself receives
            validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))

            if not transaction_arrival_queue:
                print("Workers offline or disconnected while transmitting updates, or no transaction has been verified or been received due to time-out by this validator.")
                continue
        
            # associate with a miner to send validated transactions
            associated_miner = validator.associate_with_device("miner")
            if not associated_miner:
                print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")
                continue
            associated_miner.add_device_to_association(validator)

            # broadcast to other validators
            # may go offline at any point
            if validator.online_switcher() and transaction_arrival_queue:
                validator.validator_broadcast_worker_transactions()
            if validator.is_online():
                validator.online_switcher()

        ''' Step 2.5 - with the broadcasted workers transactions, validators decide the final transaction arrival order '''
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if validator.is_online():
                print(f"{validator.return_idx()} - validator calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
                accepted_broadcasted_validator_transactions = validator.return_accepted_broadcasted_worker_transactions()
                self_validator_link_speed = validator.return_link_speed()
                # calculate broadcasted transactions arrival time
                accepted_broadcasted_transactions_arrival_queue = {}
                if accepted_broadcasted_validator_transactions:
                    for broadcasting_validator_record in accepted_broadcasted_validator_transactions:
                        broadcasting_validator_link_speed = broadcasting_validator_record['source_device_link_speed']
                        lower_link_speed = self_validator_link_speed if self_validator_link_speed < broadcasting_validator_link_speed else broadcasting_validator_link_speed

                        for arrival_time_at_broadcasting_validator, broadcasted_transaction in broadcasting_validator_record['broadcasted_transactions'].items():
                            transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
                            accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_validator] = broadcasted_transaction
                # mix the boardcasted transactions with the direct accepted transactions
                final_transactions_arrival_queue = sorted({**validator.return_unordered_arrival_time_accepted_worker_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
                validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
                # inited both vars to empty dicts so should be fine now
                # if validator.return_unordered_arrival_time_accepted_worker_transactions() and accepted_broadcasted_transactions_arrival_queue:
                #     final_transactions_arrival_queue = sorted({**validator.return_unordered_arrival_time_accepted_worker_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
                #     validator.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
                # elif validator.return_unordered_arrival_time_accepted_worker_transactions():
                #     validator.set_transaction_for_final_validating_queue(sorted(validator.return_unordered_arrival_time_accepted_worker_transactions()))
                # elif accepted_broadcasted_transactions_arrival_queue:
                #     (sorted(accepted_broadcasted_transactions_arrival_queue))
                # else:
                #     print(f"{validator.return_idx()} - validator does not have any transaction recorded in this round.")
            else:
                print(f"{validator.return_idx()} - validator {worker_iter+1}/{len(workers_this_round)} is offline")


        ''' Step 3 - validators do self and cross-validation(evaluate local updates from workers) by the order of transaction arrival time. Validator only record the signature verified transactions'''
        print()
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if not validator.is_online():
                # give a chance for validator to go back online and run its errands
                validator.online_switcher()
                if validator.is_back_online():
                    if validator.pow_resync_chain():
                        validator.update_model_after_chain_resync()
            if validator.is_online():
                final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
                if final_transactions_arrival_queue:
                    for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                        validation_time, sig_verified_unconfirmmed_transaction = validator.validate_worker_transaction(unconfirmmed_transaction, rewards)
                        if validation_time:
                            # beginning_time_for_miner: validation_transaction
                            validator.add_sig_verified_transaction_to_queue((arrival_time + validation_time, validator.return_link_speed(), sig_verified_unconfirmmed_transaction))
                            print(f"A validation process has been done for the transaction from worker {sig_verified_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}")
                else:
                    print(f"{validator.return_idx()} - validator {worker_iter+1}/{len(workers_this_round)} did not receive any transactions from worker or validator in this round.")
                    continue
                # validator.return_sig_verified_transactions_queue()

        ''' Step 4 - validators send signature verified transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists'''
        print()
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain():
                    miner.update_model_after_chain_resync()
                # miner accepts local updates from its workers association
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' sig verified transactions...")
                potential_offline_validators = set()
                associated_validators = miner.return_associated_validators()
                if not associated_validators:
                    print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                self_miner_link_speed = miner.return_link_speed()
                validator_transactions_arrival_queue = {}
                for validator in associated_validators:
                    if validator.is_online():
                        sig_verified_transactions_by_validator = validator.return_sig_verified_transactions_queue()
                        for (validator_sending_time, source_validator_link_spped, sig_verified_unconfirmmed_transaction) in sig_verified_transactions_by_validator:
                            lower_link_speed = self_miner_link_speed if self_miner_link_speed < source_validator_link_spped else source_validator_link_spped
                            transmission_delay = getsizeof(str(sig_verified_unconfirmmed_transaction))/lower_link_speed
                            # validator_transactions_arrival_queue[validator_sending_time + transmission_delay] = {'broadcasting_miner_link_speed': miner.return_link_speed(), 'received_validator_transaction': sig_verified_unconfirmmed_transaction}
                            validator_transactions_arrival_queue[validator_sending_time + transmission_delay] = sig_verified_unconfirmmed_transaction
                        # miner.add_unconfirmmed_transaction(validator.return_local_updates_and_signature(comm_round), validator.return_idx())
                    else:
                        potential_offline_validators.add(validator)
                        if args["verbose"]:
                            print(f"validator {validator.return_idx()} is offline when accepting sig verified transaction. Removed from peer list.")
                miner.remove_peers(potential_offline_validators)
                miner.set_unordered_arrival_time_accepted_validator_transactions(validator_transactions_arrival_queue)
                miner.miner_broadcast_validator_transactions()
            #     if not miner.return_unconfirmmed_transactions():
            #         print("Workers offline or disconnected while transmitting updates.")
            #         continue
            #     # broadcast to other miners
            #     # may go offline at any point
            #     if miner.online_switcher() and miner.return_unconfirmmed_transactions():
            #         miner.broadcast_transactions()
            #     if miner.is_online():
            #         miner.online_switcher()
            # else:
            #     print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue

        ''' Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order '''
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.is_online():
                print(f"{miner.return_idx()} - miner calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions...")
                accepted_broadcasted_validator_transactions = miner.return_accepted_broadcasted_validator_transactions()
                self_miner_link_speed = miner.return_link_speed()
                # calculate broadcasted transactions arrival time
                accepted_broadcasted_transactions_arrival_queue = {}
                if accepted_broadcasted_validator_transactions:
                    for broadcasting_miner_record in accepted_broadcasted_validator_transactions:
                        broadcasting_miner_link_speed = broadcasting_miner_record['source_device_link_speed']
                        lower_link_speed = self_miner_link_speed if self_miner_link_speed < broadcasting_miner_link_speed else broadcasting_miner_link_speed

                        for arrival_time_at_broadcasting_miner, broadcasted_transaction in broadcasting_miner_record['broadcasted_transactions'].items():
                            transmission_delay = getsizeof(str(broadcasted_transaction))/lower_link_speed
                            accepted_broadcasted_transactions_arrival_queue[transmission_delay + arrival_time_at_broadcasting_miner] = broadcasted_transaction
                # mix the boardcasted transactions with the direct accepted transactions
                final_transactions_arrival_queue = sorted({**miner.return_unordered_arrival_time_accepted_validator_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
                miner.set_transaction_for_final_mining_queue(final_transactions_arrival_queue)
                # inited both vars to empty dicts so should be fine now
                # if miner.return_unordered_arrival_time_accepted_worker_transactions() and accepted_broadcasted_transactions_arrival_queue:
                #     final_transactions_arrival_queue = sorted({**miner.return_unordered_arrival_time_accepted_worker_transactions(), **accepted_broadcasted_transactions_arrival_queue}.items())
                #     miner.set_transaction_for_final_validating_queue(final_transactions_arrival_queue)
                # elif miner.return_unordered_arrival_time_accepted_worker_transactions():
                #     miner.set_transaction_for_final_validating_queue(sorted(miner.return_unordered_arrival_time_accepted_worker_transactions()))
                # elif accepted_broadcasted_transactions_arrival_queue:
                #     (sorted(accepted_broadcasted_transactions_arrival_queue))
                # else:
                #     print(f"{miner.return_idx()} - miner does not have any transaction recorded in this round.")
            else:
                print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")


        
        ''' Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time, and record the sig verified transactions in the candidate block according to the limit size. Also mine the block.'''
        print()
        # block_generation_time_spent = {}
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                # give a chance for miner to go back online and run its errands
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                final_transactions_arrival_queue = miner.return_final_transactions_mining_queue()
                candidate_transacitons = []
                begin_mining_time = 0
                if final_transactions_arrival_queue:
                    time_limit = miner.return_miner_acception_wait_time()
                    size_limit = miner.return_miner_accepted_transactions_size_limit()
                    for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                        if time_limit:
                            if arrival_time > time_limit:
                                break
                        if size_limit:
                            if getsizeof(str(candidate_transacitons)) > size_limit:
                                break
                        # verify validator signature of this transaction
                        verification_time = miner.verify_validator_transaction(unconfirmmed_transaction)
                        if verification_time:
                            validator_info_this_tx = {
                            'validator': unconfirmmed_transaction['validation_done_by'],
                            'validation_rewards': unconfirmmed_transaction['validation_rewards'],
                            'validation_time': unconfirmmed_transaction['validation_time'],
                            'validator_rsa_pub_key': unconfirmmed_transaction['validator_rsa_pub_key'],
                            'validator_signature': unconfirmmed_transaction['validator_signature'],
                            'update_direction': unconfirmmed_transaction['update_direction'],
                            'miner_verification_time': verification_time,
                            'miner_rewards_for_this_tx': rewards}
                            # validator's transaction signature valid
                            found_same_worker_transaction = False
                            for candidate_transaciton in candidate_transacitons:
                                if candidate_transaciton['worker_signature'] == unconfirmmed_transaction['worker_signature']:
                                    found_same_worker_transaction = True
                                    break
                            if not found_same_worker_transaction:
                                candidate_transaciton = copy.deepcopy(unconfirmmed_transaction)
                                del candidate_transaciton['validation_done_by']
                                del candidate_transaciton['validation_rewards']
                                del candidate_transaciton['update_direction']
                                del candidate_transaciton['validation_time']
                                del candidate_transaciton['validator_rsa_pub_key']
                                del candidate_transaciton['validator_signature']
                                candidate_transaciton['positive_direction_validator'] = []
                                candidate_transaciton['negative_direction_validator'] = []
                                candidate_transacitons.append(candidate_transaciton)
                            if unconfirmmed_transaction['update_direction']:
                                candidate_transaciton['positive_direction_validator'].append(validator_info_this_tx)
                            else:
                                candidate_transaciton['negative_direction_validator'].append(validator_info_this_tx)
                            # (re)sign this candidate transaction
                            signing_time = miner.sign_candidate_transaction(candidate_transaciton)
                            new_begining_mining_time = arrival_time + verification_time + signing_time
                            begin_mining_time = new_begining_mining_time if new_begining_mining_time > begin_mining_time else begin_mining_time
                # print(candidate_transacitons)
                # put transactions into candidate block and begin mining
                # block index starts from 1
                start_time_point = time.time()
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, transactions=candidate_transacitons, miner_rsa_pub_key=miner.return_rsa_pub_key())
                # mine the block
                if candidate_block.return_transactions():
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
                # unfortunately may go offline
                if miner.online_switcher():
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
                    # record mining time
                    miner_computation_power = miner.return_computation_power()
                    if miner_computation_power:
                        # block_generation_time_spent[miner] = begin_mining_time + (time.time() - start_time_point)/miner_computation_power
                        block_generation_time_spent = (time.time() - start_time_point)/miner_computation_power
                        miner.set_block_generation_time_point(begin_mining_time + block_generation_time_spent)
                        print(f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds.")
                    else:
                        block_generation_time_spent = float('inf')
                        miner.set_block_generation_time_point(float('inf'))
                        print(f"{miner.return_idx()} - miner mines a block in INFINITE time...")
                    # immediately propagate the block
                    miner.propagated_the_block(miner.return_block_generation_time_point(), mined_block)
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # TODO PoS here??

        ''' Step 6 - miners decide if adding a propagated block or its own mined block, and request its associated devices to download this block'''
        # should not depend on min time selection
        # abort mining if propagated block is received
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                # give a chance for miner to go back online and run its errands
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # add self mined block to the processing queue and sort by time
                unordered_propagated_block_processing_queue = miner.return_unordered_propagated_block_processing_queue()
                unordered_propagated_block_processing_queue[miner.return_block_generation_time_point()] = miner.return_mined_block()
                ordered_all_blocks_processing_queue = sorted(unordered_propagated_block_processing_queue.items())
                if ordered_all_blocks_processing_queue:
                    for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
                        # sending miner in this case its the miner who mined and propagated this block
                        verified_block, verification_time = miner.verify_block(block_to_verify, block_to_verify.return_mined_by())
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(f"Miner {miner.return_idx()} is adding its own mined block.")
                            else:
                                print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}.")
                            miner.add_block(verified_block)
                            # requesting devices in its associations to download this block
                            miner.request_to_download(verified_block, block_arrival_time + verification_time)
                            break
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # CHECK FOR FORKING
        added_blocks_miner_set = set()
        for device in devices_list:
            the_added_block = device.return_the_added_block()
            if the_added_block:
                print(f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}")
                added_blocks_miner_set.add(the_added_block.return_mined_by())
        if len(added_blocks_miner_set) > 1:
            cont = input("WARNING: a forking event just happened!\nPress any key to continue")
        else:
            print("No forking event happened.")

        # update model
        # if PoW, do not track rewards by individual devices and only calculate its own, just skip the code for now
        added_blocks_miner_set = set()
        for device in devices_list:
            if device.return_the_added_block():
                pass
        # if PoS, resync then update model due to rewards calculation
        
        mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

        if mining_consensus == 'PoW':
            # select winning block based on PoW
            try:
                winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
            except:
                print("No worker block is generated in this round. Skip to the next round.")
                continue
        else:
            # select winning block based on PoS
            winning_miner = None
            highest_stake = -1
            for miner in miners_this_round:
                if miner.return_mined_block():
                    if miner.return_stake() > highest_stake:
                        highest_stake = miner.return_stake()
                        winning_miner = miner
        
        print(f"\n{winning_miner.return_idx()} is the winning miner by {mining_consensus}")
        block_to_propagate = winning_miner.return_mined_block()
        # winning miner receives mining rewards
        # winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        # IGNORE SUBNETS, where propagated block will be tossed
        # Subnets should be found by finding connected nodes in a graph
        # IN REALITY, FORK MAY HAPPEN AT THIS MOMENT
        # actually, in this system fork can still happen - two nodes have the same length of different chain for their peers in different network group to sync. But they should eventually catch up
        # winning miner adds this block to its own chain
        winning_miner.add_block(block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its worker block.")

        # miner propagate the winning block (just let other miners in its peer list receive it, verify it and add to the blockchain)
        # update peer list
        if not winning_miner.update_peer_list(args['verbose']):
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. propagated block ")
                continue
        # miners_this_round will be updated to the miners in the peer list of the winnning miner and the winning miner itself
        miners_in_winning_miner_subnet = winning_miner.return_miners_eligible_to_continue()

        print()
        if miners_in_winning_miner_subnet:
            if args["verbose"]:
                print("Miners in the winning miners subnet are")
                for miner in miners_in_winning_miner_subnet:
                    print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
                miners_in_other_nets = set(miners_this_round).difference(miners_in_winning_miner_subnet)
                if miners_in_other_nets:
                    print("These miners in other subnets will not get this propagated block.")
                    for miner in miners_in_other_nets:
                        print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
        else:
            if args["verbose"]:
                print("THIS SHOULD NOT GET CALLED AS THERE IS AT LEAST THE WINNING MINER ITSELF IN THE LIST.")

        # debug_propagated_block_list = []
        # last_block_hash = {}
        # miners accept propagated block
        print()
        miners_in_winning_miner_subnet = list(miners_in_winning_miner_subnet)
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is accepting propagated worker block.")
                miner.receive_propagated_block(block_to_propagate)
                if miner.return_propagated_block():
                    verified_block = miner.verify_block(miner.return_propagated_block(), winning_miner.return_idx())
                    #last_block_hash[miner.return_idx()] = {}
                    if verified_block:
                        # if verified_block.return_block_idx() != 1:
                        #     last_block_hash[miner.return_idx()]['block_idx'] = miner.return_blockchain_object().return_last_block().return_block_idx()
                        #     last_block_hash[miner.return_idx()]['block_hash'] = miner.return_blockchain_object().return_last_block_hash()
                        #     last_block_hash[miner.return_idx()]['block_str'] = str(sorted(miner.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                        miner.add_block(verified_block)
                        # debug_propagated_block_list.append(True)
                        pass
                    else:
                        # if block_to_propagate.return_block_idx() != 1:
                        #     last_block_hash[miner.return_idx()]['block_idx'] = miner.return_blockchain_object().return_last_block().return_block_idx()
                        #     last_block_hash[miner.return_idx()]['block_hash'] = miner.return_blockchain_object().return_last_block_hash()
                        #     last_block_hash[miner.return_idx()]['block_str'] = str(sorted(miner.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                        #     debug_propagated_block_list.append(False)
                        #     miner.verify_block(miner.return_propagated_block())
                        miner.toss_propagated_block()
                        print("Received propagated worker block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")
        # print(debug_propagated_block_list)
        # print()
        
        # worker_last_block_hash = {}
        # miner requests worker to download block
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting its workers to download a new worker block.")
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    if not associated_workers:
                        print(f"No workers are associated with miner {miner.return_idx()} to accept the worker block.")
                        continue
                    for worker in associated_workers:
                        print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting worker {worker.return_idx()} to download...")
                        if not worker.is_online():
                            worker.online_switcher()
                            if worker.is_back_online():
                                if worker.pow_resync_chain():
                                    worker.update_model_after_chain_resync()
                        if worker.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            worker.receive_block_from_miner(block_to_send, miner.return_idx())
                            verified_block = worker.verify_block(worker.return_received_block_from_miner(), miner.return_idx())
                            if verified_block:
                                # if verified_block.return_block_idx() != 1:
                                #     worker_last_block_hash[worker.return_idx()]['block_idx'] = worker.return_blockchain_object().return_last_block().return_block_idx()
                                #     worker_last_block_hash[worker.return_idx()]['block_hash'] = worker.return_blockchain_object().return_last_block_hash()
                                #     worker_last_block_hash[worker.return_idx()]['block_str'] = str(sorted(worker.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                                worker.add_block(verified_block)
                                pass
                            else:
                                # if block_to_send.return_block_idx() != 1:
                                #     worker.verify_block(worker.return_received_block_from_miner())
                                #     worker_last_block_hash[worker.return_idx()]['block_idx'] = worker.return_blockchain_object().return_last_block().return_block_idx()
                                #     worker_last_block_hash[worker.return_idx()]['block_hash'] = worker.return_blockchain_object().return_last_block_hash()
                                #     worker_last_block_hash[worker.return_idx()]['block_str'] = str(sorted(worker.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                                worker.toss_received_block()
                                print("Received block from the associated miner is not valid or does not fit its chain. Pass to the next worker.")
                            worker.online_switcher()
            miner.online_switcher()
                            
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        print()
        # workers do global updates
        for worker in workers_this_round:
            if not worker.is_online():
                worker.online_switcher()
                if worker.is_back_online():
                    if worker.pow_resync_chain():
                        worker.update_model_after_chain_resync()
            if worker.is_online():
                print(f'Worker {worker.return_idx()} is doing global update...')
                if worker.return_received_block_from_miner():
                    worker.global_update()
                    accuracy = worker.evaluate_model_weights()
                    accuracy = float(accuracy)
                    worker.set_accuracy_this_round(accuracy)
                    report_msg = f'Worker {worker.return_idx()} at the communication round {comm_round+1} with chain length {worker.return_blockchain_object().return_chain_length()} has accuracy: {accuracy}\n'
                    print(report_msg)
                    devices_accuracies_records[worker.return_idx()][f'round_{comm_round}'] = accuracy
                    worker.online_switcher()
                else:
                    print(f'No block has been sent to worker {worker.return_idx()}. Skipping global update.\n')
        
        # record accuries in log file
        log_file_path = f"logs/accuracy_report_{date_time}.txt"
        open(log_file_path, 'w').close()
        for device_idx, accuracy_records in devices_accuracies_records.items():
            accuracy_list = []
            for accuracy in accuracy_records.values():
                accuracy_list.append(accuracy)
            with open(log_file_path, "a") as file:
                file.write(f"{device_idx} : {accuracy_list}\n")
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # TODO validator may also be evil. how to validate validators?
        # or maybe in this specific settings, since devices take turns to become validators and not specifically set to some certain memvbers, we believe most of the members in the system want to benefit the whole community and trust validators by default
        # after all, they are still taking chances to send their validations to the miners
        # workers send their accuracies to validators and validators record the accuracies in a block
        # iterating validator is easier than iterating worker because of the creation of the validator block

        print("Begin validator rounds.")
        # validators request accuracies from the workers in their peer lists
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if validator.is_online():
                # update peer list
                if not validator.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(validator, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if validator.pow_resync_chain():
                    validator.update_model_after_chain_resync()
                last_block_on_validator_chain = validator.return_blockchain_object().return_last_block()
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} accepting workers' accuracies...")
                # check last block
                if last_block_on_validator_chain == None or last_block_on_validator_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                online_workers_in_peer_list = validator.return_online_workers()
                if not online_workers_in_peer_list:
                    print(f"Cannot find online workers in {validator.return_idx()} peer list.")
                    continue

                # validator_candidate_block = Block(idx=validator.blockchain.return_chain_length()+1, is_validator_block=True)
                for worker in online_workers_in_peer_list:
                    smart_contract_worker_upload_accuracy_to_validator(worker, validator)
                # validator.record_worker_performance_in_block(validator_candidate_block, comm_round, args["general_rewards"])
                # associate with a miner
                associated_miner = validator.associate_with_miner()
                if not associated_miner:
                    print(f"Cannot find a miner in {validator.return_idx()} peer list.")
                    continue
                finally_no_associated_miner = False
                # check if the associated miner is online
                while not associated_miner.is_online():
                    validator.remove_peers(associated_miner)
                    associated_miner = validator.associate_with_miner()
                    if not associated_miner:
                        finally_no_associated_miner = True
                        break
                if finally_no_associated_miner:
                    print(f"Cannot find a online miner in {validator.return_idx()} peer list.")
                    continue
                else:
                    print(f"Validator {validator.return_idx()} associated with miner {associated_miner.return_idx()}")
                associated_miner.add_validator_to_association(validator)
                # may go offline
                validator.online_switcher()
            else:
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is offline") 
                
        # miners accept validators' transactions and broadcast to other miners
        print()
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain():
                    miner.update_model_after_chain_resync()
                # miner accepts validator transactions
                miner.miner_reset_vars_for_new_validation_round()
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' transactions...")
                potential_offline_validators = set()
                associated_validators = miner.return_associated_validators()
                if not associated_validators:
                    print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for validator in associated_validators:
                    if validator.is_online():
                        miner.add_unconfirmmed_transaction(validator.return_validations_and_signature(comm_round), validator.return_idx())
                    else:
                        potential_offline_validators.add(validator)
                        if args["verbose"]:
                            print(f"validator {validator.return_idx()} is offline when accepting transaction. Removed from peer list.")
                miner.remove_peers(potential_offline_validators)
                if not miner.return_unconfirmmed_transactions():
                    print("Validators offline or disconnected while transmitting validations.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher() and miner.return_unconfirmmed_transactions():
                    miner.broadcast_transactions()
                if miner.is_online():
                    miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")

        # miners do self and cross-validation(only validating signature at this moment)
        print()
        block_generation_time_spent = {}
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                last_block_on_miner_chain = miner.return_blockchain_object().return_last_block()
                # check last block
                # though miner could still mine this block, but if it finds itself cannot add this mined block, it's unwilling to mine
                if last_block_on_miner_chain==None or last_block_on_miner_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                start_time = time.time()
                # block index starts from 1
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key(), is_validator_block=True)
                # self verification
                unconfirmmed_transactions = miner.return_unconfirmmed_transactions()
                if unconfirmmed_transactions:
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing self verification...")
                else:
                    print(f"\nNo recorded transactions by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do self verification.")
                for unconfirmmed_transaction in miner.return_unconfirmmed_transactions():
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['mining_rewards'] = rewards
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        # TODO put outside
                    miner.receive_rewards(rewards)
                # cross verification
                accepted_broadcasted_transactions = miner.return_accepted_broadcasted_transactions()
                if accepted_broadcasted_transactions:
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing cross verification...")
                else:
                    print(f"No broadcasted transactions have been recorded by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do cross verification.")
                for unconfirmmed_transactions in miner.return_accepted_broadcasted_transactions():
                    for unconfirmmed_transaction in unconfirmmed_transactions:
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards) 
                # mine the block
                if candidate_block.return_transactions():
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the validator block...")
                    # add previous hash(last block had been checked above)
                    candidate_block.set_previous_block_hash(miner.return_blockchain_object().return_last_block().compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.proof_of_work(candidate_block)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline
                if miner.online_switcher():
                    # record mining time
                    try:
                        block_generation_time_spent[miner] = (time.time() - start_time)/(miner.return_computation_power())
                        print(f"{miner.return_idx()} - miner mines a validator block in {block_generation_time_spent[miner]} seconds.")
                    except:
                        block_generation_time_spent[miner] = float('inf')
                        print(f"{miner.return_idx()} - miner mines a validator block in INFINITE time...")
                    mined_block.set_mining_rewards(rewards)
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        # select the winning miner and broadcast its mined block
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No validator block is generated in this round. Skip to the next round.")
            continue
        
        print(f"\n{winning_miner.return_idx()} is the winning miner for the validator block this round.")
        validator_block_to_propagate = winning_miner.return_mined_block()
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        winning_miner.add_block(validator_block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its validator block.")

        if not winning_miner.update_peer_list(args['verbose']):
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. propagated block ")
                continue

        miners_in_winning_miner_subnet = winning_miner.return_miners_eligible_to_continue()
            
        print()
        if miners_in_winning_miner_subnet:
            if args["verbose"]:
                print("Miners in the winning miners subnet are")
                for miner in miners_in_winning_miner_subnet:
                    print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
                miners_in_other_nets = set(miners_this_round).difference(miners_in_winning_miner_subnet)
                if miners_in_other_nets:
                    print("These miners in other subnets will not get this propagated block.")
                    for miner in miners_in_other_nets:
                        print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
        else:
            if args["verbose"]:
                print("THIS SHOULD NOT GET CALLED AS THERE IS AT LEAST THE WINNING MINER ITSELF IN THE LIST.")
        
        print()
        miners_in_winning_miner_subnet = list(miners_in_winning_miner_subnet)
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is accepting propagated validator block.")
                miner.receive_propagated_validator_block(validator_block_to_propagate)
                if miner.return_propagated_validator_block():
                    verified_block = miner.verify_block(miner.return_propagated_validator_block(), winning_miner.return_idx())
                    #last_block_hash[miner.return_idx()] = {}
                    if verified_block:
                        miner.add_block(verified_block)
                        pass
                    else:
                        miner.toss_ropagated_validator_block()
                        print("Received propagated validator block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # miner requests worker and validator to download the validator block
        # does not matter if the miner did not append block above and send its last block, as it won't be verified
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting its workers and validators to download a new validator block.")
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    associated_validators = miner.return_associated_validators()
                    associated_devices = associated_workers.union(associated_validators)
                    if not associated_devices:
                        print(f"No devices are associated with miner {miner.return_idx()} to accept the validator block.")
                        continue
                    associated_devices = list(associated_devices)
                    for device_iter in range(len(associated_devices)):
                        device = associated_devices[device_iter]
                        print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting device {device_iter+1}/{len(associated_devices)} {device.return_idx()} - {device.return_role()} to download...")
                        if not device.is_online():
                            device.online_switcher()
                            if device.is_back_online():
                                if device.pow_resync_chain():
                                    device.update_model_after_chain_resync()
                        if device.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            device.reset_received_block_from_miner_vars()
                            device.receive_block_from_miner(block_to_send, miner.return_idx())
                            verified_block = device.verify_block(device.return_received_block_from_miner(), miner.return_idx())
                            if verified_block:
                                device.add_block(verified_block)
                                pass
                            else:
                                device.toss_received_block()
                                print("Received block from the associated miner is not valid or does not fit its chain. Pass to the next worker.")
                            device.online_switcher()
                            if device.return_has_added_block():
                                block_to_propagate = device.return_blockchain_object().return_last_block()
                                if block_to_propagate.is_validator_block():
                                    device.operate_on_validator_block()
                    print(f"\nminer {miner.return_idx()} is processing this validator block.")
                    miner.operate_on_validator_block()

                
