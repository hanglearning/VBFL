# fedavg from https://github.com/WHDY/FedAvg/

import os
import sys
import argparse
#from tqdm import tqdm
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nd', '--num_devices', type=int, default=100, help='numer of the devices in the simulation network')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
#parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
#parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=1000, help='maximum number of communication rounds, may terminate early if converges')
# parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-ns', '--network_stability', type=float, default=0.8, help='the odds a device is online')
parser.add_argument('-gr', '--general_rewards', type=int, default=1, help='rewards for providing data, verification of one transaction, mining and so forth')
parser.add_argument('-v', '--verbose', type=int, default=0, help='print verbose debug log')
parser.add_argument('-aio', '--all_in_one', type=int, default=0, help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=5, help='a device is kicked out of the network if its accuracy shows decreasing for the number of rounds recorded by a winning validator')
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='hard assign number of roles in the network, order by worker, miner and validator')
# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across devices when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
# parser.add_argument('-vh', '--validator_threshold', type=float, default=0.1, help="a threshold value of accuracy difference to determine malicious worker")
# use time window and size limit together to determine how many epochs a worker can perform and how many can a validator accept
parser.add_argument('-vt', '--validator_acception_wait_time', type=float, default=0.0, help="default time window for valitors to accept transactions, in seconds. Either this or -le must be specified.")
# parser.add_argument('-vs', '--validator_sig_validated_transactions_size_limit', type=float, default=0.0, help="default total size of the transactions a validator can accept. this partly determines the final block size. Requires -vt to be specified! 0 means no size limit.")
# parser.add_argument('-vss', '--validator_size_stop', type=float, default=35000.0, help="when validator_sig_validated_transactions_size_limit is specified, this value is used to determine that when the remaining buffer of the validator is less than this value, validator stops accepting transactions")
parser.add_argument('-le', '--default_local_epochs', type=int, default=1, help='local train epoch. Train local model by this num of epochs for each worker, if -vt is not specified')
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0, help="default time window for miners to accept transactions, in seconds. 0 means no time limit. either this or -ml has to be specified, or both")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=35000.0, help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")
parser.add_argument('-ls', '--link_speed', type=int, default=0, help="This variable is used to simulate transmission delay. Default value 0 means no transmission delay. If set to 1, link speed is randomly initiated between 0 and 1, meaning seconds spent to transmit 70k bytes - during experiment, one transaction is around 35k bytes")
parser.add_argument('-cp', '--computation_power', type=int, default=0, help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 0 means evenly assign computation power. If set to 1, power is randomly initiated as an int between 0 and 4, both included.")
# def flattern_2d_to_1d(arr):
#     final_set = set()
#     for sub_arr in arr:
#         for ele in sub_arr:
#             final_set.add(sub_arr)
#     return final_set

# def find_sub_nets():
#     # sort of DFS
#     sub_nets = []
#     for device_seq, device in devices_in_network.devices_set.items():
#         sub_net = set()
#         checked_device = flattern_2d_to_1d(sub_nets)
#         while device not in checked_device and not in sub_net:
#             sub_net.add(device)
#             for peer in device.return_peers():
#                 device = peer
#         sub_nets.append(sub_net)


# TODO write logic here as the control should not be in device class, must be outside
def smart_contract_worker_upload_accuracy_to_validator(worker, validator):
    validator.accept_accuracy(worker, rewards)

# TODO should be flexible depending on loose/hard assign
# TODO since we now allow devices to go back online, may discard this function
def check_network_eligibility(check_online=False):
    # num_online_workers = 0
    # num_online_miners = 0
    # num_online_validators = 0
    # for worker in workers_this_round:
    #     if worker.is_online():
    #         num_online_workers += 1
    # for miner in miners_this_round:
    #     if miner.is_online():
    #         num_online_miners += 1
    # for validator in validators_this_round:
    #     if validator.is_online():
    #         num_online_validators += 1
    ineligible = False
    if len(workers_this_round) == 0:
        print('There is no workers online in this round, ', end='')
        ineligible = True
    elif len(miners_this_round) == 0:
        print('There is no miners online in this round, ', end='')
        ineligible = True
    elif len(validators_this_round) == 0:
        print('There is no validators online in this round, ', end='')
        ineligible = True
    if ineligible:
        print('which is ineligible for the network to continue operating.')
        return False
    return True

def register_in_the_network(registrant, check_online=False):
    potential_registrars = set(devices_in_network.devices_set.values())
    # it cannot register with itself
    potential_registrars.discard(registrant)        
    # pick a registrar
    registrar = random.sample(potential_registrars, 1)[0]
    if check_online:
        if not registrar.is_online():
            online_registrars = set()
            for registrar in potential_registrars:
                if registrar.is_online():
                    online_registrars.add(registrar)
            if not online_registrars:
                return False
            registrar = random.sample(online_registrars, 1)[0]
    # registrant add registrar to its peer list
    registrant.add_peers(registrar)
    # this device sucks in registrar's peer list
    registrant.add_peers(registrar.return_peers())
    # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
    registrar.add_peers(registrant)
    return True

def register_by_aio(device):
    device.add_peers(set(devices_in_network.devices_set.values()))

if __name__=="__main__":

    # program running time for logging purpose
    date_time = datetime.now().strftime("%d%m%Y_%H%M%S")

    args = parser.parse_args()
    args = args.__dict__

    # for demonstration purposes, this reward is for everything
    rewards = args["general_rewards"]

    # get number of roles needed in the network
    roles_requirement = args['hard_assign'].split(',')
    
    try:
        workers_needed = int(roles_requirement[0])
    except:
        workers_needed = 0
    
    try:
        miners_needed = int(roles_requirement[1])
    except:
        miners_needed = 0
    
    try:
        validators_needed = int(roles_requirement[2])
    except:
        validators_needed = 0

    if args['num_devices'] < workers_needed + miners_needed + validators_needed:
        sys.exit("ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")

    # check eligibility
    if args['num_devices'] < 2:
        sys.exit("ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

    num_malicious = args['num_malicious']
    num_devices = args['num_devices']
    if num_malicious:
        if num_malicious > num_devices:
            sys.exit("ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network")
        else:
            print(f"Malicious nodes vs total devices set to {num_devices}/{num_devices} = {(num_devices/num_devices)*100:.2f}%")

    # # make chechpoint save path
    # if not os.path.isdir(args['save_path']):
    #     os.mkdir(args['save_path'])

    # create neural net based on the input model name
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    # assign GPUs if available and prepare the net
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    print(f"{torch.cuda.device_count()} GPUs are available to use!")
    net = net.to(dev)

    # set loss_function and optimizer
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
    
    # get validator transaction acception limit
    # validator_acception_wait_time = args['validator_acception_wait_time']
    # validator_sig_validated_transactions_size_limit = args['validator_sig_validated_transactions_size_limit']

    # if not (validator_acception_wait_time or validator_sig_validated_transactions_size_limit):
    #     sys.exit("ERROR: either -vl or -vl has to be specified, or both.")
    
    # get miner transaction acception limit
    # miner_acception_wait_time = args['miner_acception_wait_time']
    # miner_accepted_transactions_size_limit = args['miner_accepted_transactions_size_limit']

    # if not (miner_acception_wait_time or miner_accepted_transactions_size_limit):
    #     sys.exit("ERROR: either -mt or -ml has to be specified, or both.")

    # TODO - # of malicious nodes, non-even dataset distribution
    # create devices in the network
    devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size = args['batchsize'], loss_func = loss_func, opti = opti, num_devices=args['num_devices'], network_stability=args['network_stability'], net=net, dev=dev, knock_out_rounds=args['knock_out_rounds'], shard_test_data=args['shard_test_data'], validator_acception_wait_time=args['validator_acception_wait_time'], validator_sig_validated_transactions_size_limit=args['validator_sig_validated_transactions_size_limit'], validator_size_stop = args['validator_size_stop'], miner_acception_wait_time=args['miner_acception_wait_time'], miner_accepted_transactions_size_limit=args['miner_accepted_transactions_size_limit'], even_link_speed=args['link_speed'], even_computation_power=args['computation_power'], num_malicious=args['num_malicious'])
    # test_data_loader = devices_in_network.test_data_loader

    devices_list = list(devices_in_network.devices_set.values())

    for device in devices_list:
        # set initial global weights
        device.init_global_parameters()
        # simulate peer registration, with respect to device idx order
        if not args["all_in_one"]:
            register_in_the_network(device)
        else:
            register_by_aio(device)

    # remove its own from peer list if there is
    for device in devices_list:
        device.remove_peers(device)

    # build a dict to record worker accuracies for different rounds
    workers_accuracies_records = {}
    for device_seq, device in devices_in_network.devices_set.items():
        workers_accuracies_records[device_seq] = {}

    # FL starts here
    for comm_round in range(1, args['max_num_comm']+1):
        print(f"\nCommunication round {comm_round}")
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # assign role first, and then simulate if online or offline
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
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
            device.online_switcher()
            #     # though back_online, resync chain when they are performing tasks
            #     if args['verbose']:
            #         print(f'{device.return_idx()} {device.return_role()} online - ', end='')
            # else:
            #     if args['verbose']:
            #         print(f'{device.return_idx()} {device.return_role()} offline - ', end='')
            # # debug chain length
            # if args['verbose']:
            #     print(f"chain length {device.return_blockchain_object().return_chain_length()}")
            # debug
        
        if not check_network_eligibility():
            print("Go to the next round and re-assign role.\n")
            continue
        
        # shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        if args['verbose']:
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\nminers this round are")
            for miner in miners_this_round:
                print(f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            if validators_this_round:
                print("\nvalidators this round are")
                for validator in validators_this_round:
                    print(f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
            else:
                print("\nThere are no validators this round.")
        
        if args['verbose']:
            print(f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print()

        # debug peers
        if args['verbose']:
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():
                peers = device.return_peers()
                print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        # re-init round vars
        for miner in miners_this_round:
            if miner.is_online():
                miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            if worker.is_online():
                worker.worker_reset_vars_for_new_round()
        for validator in validators_this_round:
            if validator.is_online():
                validator.validator_reset_vars_for_new_round()
        
        # workers, validators and miners take turns to perform jobs
        
        ''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code of step 2) '''
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            if worker.is_online():
                # update peer list
                if not worker.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if worker.pow_resync_chain(args['verbose']):
                    worker.update_model_after_chain_resync()
                # worker perform local update
                print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} performing local updates...")
                # worker associates with a miner to accept finally mined block
                associated_miner = worker.associate_with_device("miner")
                if not associated_miner:
                    print(f"Cannot find a qualified miner in {worker.return_idx()} peer list.")
                    continue
                associated_miner.add_device_to_association(worker)
                # worker associates with a validator
                associated_validator = worker.associate_with_device("validator")
                if not associated_validator:
                    print(f"Cannot find a qualified validator in {worker.return_idx()} peer list.")
                    continue
                associated_validator.add_device_to_association(worker)
                # simulate the situation that worker may go offline during model updates transmission
                worker.online_switcher()
            else:
                print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} is offline")
        
        ''' Step 2 - validators accept local updates and broadcast to other validators (workers local_updates() are called in this step. If validator limit transaction size, self-verification has to be performed in this loop)'''
        print()
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if validator.is_online():
                # update peer list
                if not validator.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(validator, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if validator.pow_resync_chain(args['verbose']):
                    validator.update_model_after_chain_resync()
                # validator accepts local updates from its workers association
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = validator.return_associated_workers()
                if not associated_workers:
                    print(f"No workers are associated with validator {validator.return_idx()} for this communication round.")
                    continue
                # workers local_updates() called here as their updates are restrained by validators' acception time and size
                # determine the transactions to accept
                if args['validator_acception_wait_time']:
                    # wati time has specified. let each worker does local_updates till time limit
                    validator_link_speed = validator.return_link_speed()
                    print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} link speed {validator.return_link_speed()}")
                    records_dict = dict.fromkeys(associated_workers, None)
                    for worker, _ in records_dict.items():
                        records_dict[worker] = {}
                    # used for easy sorting
                    transaction_arrival_queue = {}
                    for worker in associated_workers:
                        if not worker.return_idx() in validator.return_black_list():
                            if worker.is_online():
                                total_time_tracker = 0
                                epoch_seq = 1
                                while total_time_tracker < validator.return_validator_acception_wait_time():
                                    local_update_spent_time = worker.worker_local_update(rewards)
                                    worker_link_speed = worker.return_link_speed()
                                    unconfirmed_transaction = worker.return_local_updates_and_signature(comm_round)
                                    # size in bytes, usually around 35000 bytes per transaction
                                    unconfirmed_transactions_size = getsizeof(str(unconfirmed_transaction))
                                    transmission_delay = (unconfirmed_transactions_size/70000)/validator_link_speed if validator_link_speed < worker_link_speed else (unconfirmed_transactions_size/70000)/worker_link_speed
                                    if local_update_spent_time + transmission_delay > validator.return_validator_acception_wait_time():
                                        # last transaction sent passes the acception time window
                                        break
                                    records_dict[worker][epoch_seq] = {}
                                    records_dict[worker][epoch_seq]['local_update_time'] = local_update_spent_time
                                    records_dict[worker][epoch_seq]['transmission_delay'] = transmission_delay
                                    records_dict[worker][epoch_seq]['local_update_unconfirmed_transaction'] = unconfirmed_transaction
                                    records_dict[worker][epoch_seq]['local_update_unconfirmed_transaction_size'] = unconfirmed_transactions_size
                                    if epoch_seq == 1:
                                        total_time_tracker = local_update_spent_time + transmission_delay
                                    else:
                                        total_time_tracker = total_time_tracker - records_dict[worker][epoch_seq - 1]['transmission_delay'] + local_update_spent_time + transmission_delay
                                    records_dict[worker][epoch_seq]['arrival_time'] = total_time_tracker
                                    transaction_arrival_queue[total_time_tracker] = {}
                                    transaction_arrival_queue[total_time_tracker]['worker'] = worker
                                    transaction_arrival_queue[total_time_tracker]['epoch'] = epoch_seq
                                    epoch_seq += 1
                            else:
                                potential_offline_workers.add(worker)
                                if args["verbose"]:
                                    print(f"worker {worker.return_idx()} is offline when accepting transaction. Removed from peer list.")
                        else:
                            print(f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
                    # workers local updates done by time limit
                    # begin self-verification, parimarily due to validated_size_limit may be specified
                    # sort arrival time of all possible transactions
                    ordered_transaction_arrival_queue = sorted(transaction_arrival_queue.items())
                    # sig_verified_transactions = []
                    transaction_size_tracker = 0
                    for transaction_record in ordered_transaction_arrival_queue:
                        by_worker = transaction_record[-1]['worker']
                        epoch_seq = transaction_record[-1]['epoch']
                        transaction_to_verify = records_dict[by_worker][epoch_seq]['local_update_unconfirmed_transaction']
                        if validator.verify_transaction_by_signature(transaction_to_verify):
                            # toss if there are previously accepted transactions from the same worker and the epoch seq was behind
                            currently_accepted_sig_verified_transaction = copy.copy(validator.return_sig_verified_transactions())
                            for already_accepted_transaction in currently_accepted_sig_verified_transaction:
                                if transaction_to_verify['rsa_pub_key'] == already_accepted_transaction['rsa_pub_key']:
                                    # if accepted transaction is from an earlier epoch, toss previously accepted one
                                    if already_accepted_transaction['local_total_epoch'] < transaction_to_verify['local_total_epoch']:
                                        # toss
                                        validator.remove(already_accepted_transaction)
                                        transaction_size_tracker -= records_dict[by_worker][already_accepted_transaction['local_total_epoch']]['local_update_unconfirmed_transaction_size']
                                        # add
                                        validator.add_sig_verified_transaction(transaction_to_verify, by_worker.return_idx())
                                        transaction_size_tracker += records_dict[by_worker][epoch_seq]['local_update_unconfirmed_transaction_size']
                                    else:
                                        # received an obsolte transaction
                                        pass
                                else:
                                    # first time see this worker
                                    validator.add_sig_verified_transaction(transaction_to_verify, by_worker.return_idx())
                                    transaction_size_tracker += records_dict[by_worker][epoch_seq]['local_update_unconfirmed_transaction_size']
                            # # if size is limited
                            size_limit = args['validator_sig_validated_transactions_size_limit']
                            if size_limit:
                                # check if buffer full to stop accepting. allow a bit surpass
                                if transaction_size_tracker > size_limit:
                                    print("Size limit has been reached. Stop accepting further transactions.")
                                    break
                            else:
                                # go go go
                                continue
                        else:
                            print(f"transaction arrived at {transaction_record[0]}s by worker {by_worker.return_idx()} at local epoch {epoch_seq} did not pass the signature verification.")
                else:
                    # did not specify wait time. every associated worker perform specified number of local epochs
                    for worker in associated_workers:
                        if not worker.return_idx() in validator.return_black_list():
                            if worker.is_online():
                                pass
                            else:
                                potential_offline_workers.add(worker)
                                if args["verbose"]:
                                    print(f"worker {worker.return_idx()} is offline when accepting transaction. Removed from peer list.")
                        else:
                            print(f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
                validator.remove_peers(potential_offline_workers)
                if not validator.return_sig_verified_transactions():
                    print("Workers offline or disconnected while transmitting updates, or no transaction has been verified by this validator.")
                    continue
                # broadcast to other validators
                # may go offline at any point
                if validator.online_switcher() and validator.return_sig_verified_transactions():
                    validator.broadcast_sig_verified_transaction()
                if validator.is_online():
                    validator.online_switcher()
            else:
                print(f"{validator.return_idx()} - validator {worker_iter+1}/{len(workers_this_round)} is offline")

        ''' Step 3 - validators do self and cross-validation(evaluate local updates from workers), and associate with a miner'''
        print()
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if not validator.is_online():
                # give a chance for validator to go back online and run its errands
                validator.online_switcher()
                if validator.is_back_online():
                    if validator.pow_resync_chain(args['verbose']):
                        validator.update_model_after_chain_resync()
            if validator.is_online():
                # self verification
                unconfirmmed_transactions = validator.return_unconfirmmed_transactions()
                if unconfirmmed_transactions:
                    print(f"\n{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} doing self verification...")
                else:
                    print(f"\nNo recorded transactions by {validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} will not do self verification.")
                # determine unconfirmmed_transaction validating order by taking transmission delay into account
                transaction_total_delay_records = {}
                validator_link_speed = validator.return_link_speed()
                for unconfirmmed_transaction in unconfirmmed_transactions:
                    source_worker_idx = unconfirmmed_transaction['worker_device_idx']
                    if not source_worker_idx in validator.return_black_list():
                        upload_worker_link_speed = devices_list[source_worker_idx],return_link_speed()
                        transmission_delay = getsizeof(str(unconfirmmed_transaction))/validator_link_speed if validator_link_speed < upload_worker_link_speed else getsizeof(str(unconfirmmed_transaction))/upload_worker_link_speed
                        total_delay = unconfirmmed_transaction['local_update_spent_time'] + transmission_delay
                        transaction_total_delay_records[total_delay] = unconfirmmed_transaction
                    else:
                        print(f"worker {source_worker_idx} in validator's black list. transaction not validated.")
                unconfirmmed_transaction_processing_queue = []
                for total_delay in sorted(transaction_total_delay_records):
                    unconfirmmed_transaction_processing_queue.append(transaction_total_delay_records[total_delay])
                for unconfirmmed_transaction in unconfirmmed_transaction_processing_queue:
                    # verify signature
                    if validator.verify_transaction_by_signature(unconfirmmed_transaction):
                        # validate local updates
                        # TODO DOING NOW
                        if_malicious = validator.validate_worker_local_update(unconfirmmed_transaction)
                        unconfirmmed_transaction['tx_verified_by'] = validator.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['mining_rewards'] = rewards
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                    validator.receive_rewards(rewards)
                # cross verification
                accepted_broadcasted_transactions = validator.return_accepted_broadcasted_transactions()
                if accepted_broadcasted_transactions:
                    print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} doing cross verification...")
                else:
                    print(f"No broadcasted transactions have been recorded by {validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} will not do cross verification.")
                for unconfirmmed_transactions in accepted_broadcasted_transactions:
                    for unconfirmmed_transaction in unconfirmmed_transactions:
                        if validator.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = validator.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            validator.receive_rewards(rewards) 
                # mine the block
                if candidate_block.return_transactions():
                    print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} mining the worker block...")
                    # return the last block and add previous hash
                    last_block = validator.return_blockchain_object().return_last_block()
                    if last_block is None:
                        # mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = validator.proof_of_work(candidate_block)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline
                if validator.online_switcher():
                    # record mining time
                    try:
                        block_generation_time_spent[validator] = (time.time() - start_time)/(validator.return_computation_power())
                        print(f"{validator.return_idx()} - validator mines a worker block in {block_generation_time_spent[validator]} seconds.")
                    except:
                        block_generation_time_spent[validator] = float('inf')
                        print(f"{validator.return_idx()} - validator mines a worker block in INFINITE time...")
                    mined_block.set_mining_rewards(rewards)
                    # sign the block
                    validator.sign_block(mined_block)
                    validator.set_mined_block(mined_block)
            else:
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is offline.")

        # miners accept local updates and broadcast to other miners
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
                if miner.pow_resync_chain(args['verbose']):
                    miner.update_model_after_chain_resync()
                # miner accepts local updates from its workers association
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = miner.return_associated_workers()
                if not associated_workers:
                    print(f"No workers are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for worker in associated_workers:
                    if worker.is_online():
                        miner.add_unconfirmmed_transaction(worker.return_local_updates_and_signature(comm_round), worker.return_idx())
                    else:
                        potential_offline_workers.add(worker)
                        if args["verbose"]:
                            print(f"worker {worker.return_idx()} is offline when accepting transaction. Removed from peer list.")
                miner.remove_peers(potential_offline_workers)
                if not miner.return_unconfirmmed_transactions():
                    print("Workers offline or disconnected while transmitting updates.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher() and miner.return_unconfirmmed_transactions():
                    miner.broadcast_transactions()
                if miner.is_online():
                    miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue

        # miners do self and cross-validation(only validating signature at this moment)
        # time spent included in the block_generation_time
        print()
        block_generation_time_spent = {}
        # TODO miner also performs as a validator
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                # give a chance for miner to go back online and run its errands
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                start_time = time.time()
                # block index starts from 1
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key())
                # self verification
                unconfirmmed_transactions = miner.return_unconfirmmed_transactions()
                if unconfirmmed_transactions:
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing self verification...")
                else:
                    print(f"\nNo recorded transactions by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do self verification.")
                for unconfirmmed_transaction in unconfirmmed_transactions:
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['mining_rewards'] = rewards
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        miner.receive_rewards(rewards)
                # cross verification
                accepted_broadcasted_transactions = miner.return_accepted_broadcasted_transactions()
                if accepted_broadcasted_transactions:
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing cross verification...")
                else:
                    print(f"No broadcasted transactions have been recorded by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do cross verification.")
                for unconfirmmed_transactions in accepted_broadcasted_transactions:
                    for unconfirmmed_transaction in unconfirmmed_transactions:
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards) 
                # mine the block
                if candidate_block.return_transactions():
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the worker block...")
                    # return the last block and add previous hash
                    last_block = miner.return_blockchain_object().return_last_block()
                    if last_block is None:
                        # mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
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
                        print(f"{miner.return_idx()} - miner mines a worker block in {block_generation_time_spent[miner]} seconds.")
                    except:
                        block_generation_time_spent[miner] = float('inf')
                        print(f"{miner.return_idx()} - miner mines a worker block in INFINITE time...")
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
        # PoS - change Blockchain diff to 0 and choose the miner with the max rewards
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No worker block is generated in this round. Skip to the next round.")
            continue
        
        print(f"\n{winning_miner.return_idx()} is the winning miner for the worker block this round.")
        block_to_propagate = winning_miner.return_mined_block()
        # winning miner receives mining rewards
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        # IGNORE SUBNETS, where propagated block will be tossed
        # Subnets should be found by finding connected nodes in a graph
        # IN REALITY, FORK MAY HAPPEN AT THIS MOMENT
        # actually, in this system fork can still happen - two nodes have the same length of different chain for their peers in different network group to sync. But they should eventually catch up
        # winning miner adds this block to its own chain
        winning_miner.add_block(block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its worker block.")

        # miner propogate the winning block (just let other miners in its peer list receive it, verify it and add to the blockchain)
        # update peer list
        if not winning_miner.update_peer_list(args['verbose']):
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
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
                    if miner.pow_resync_chain(args['verbose']):
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
                    if miner.pow_resync_chain(args['verbose']):
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
                                if worker.pow_resync_chain(args['verbose']):
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
                    if worker.pow_resync_chain(args['verbose']):
                        worker.update_model_after_chain_resync()
            if worker.is_online():
                print(f'Worker {worker.return_idx()} is doing global update...')
                if worker.return_received_block_from_miner():
                    worker.global_update()
                    accuracy = worker.evaluate_updated_weights()
                    accuracy = float(accuracy)
                    worker.set_accuracy_this_round(accuracy)
                    report_msg = f'Worker {worker.return_idx()} at the communication round {comm_round+1} with chain length {worker.return_blockchain_object().return_chain_length()} has accuracy: {accuracy}\n'
                    print(report_msg)
                    workers_accuracies_records[worker.return_idx()][f'round_{comm_round}'] = accuracy
                    worker.online_switcher()
                else:
                    print(f'No block has been sent to worker {worker.return_idx()}. Skipping global update.\n')
        
        # record accuries in log file
        log_file_path = f"logs/accuracy_report_{date_time}.txt"
        open(log_file_path, 'w').close()
        for device_idx, accuracy_records in workers_accuracies_records.items():
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
                if validator.pow_resync_chain(args['verbose']):
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
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain(args['verbose']):
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
                    if miner.pow_resync_chain(args['verbose']):
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
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
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
                    if miner.pow_resync_chain(args['verbose']):
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
                    if miner.pow_resync_chain(args['verbose']):
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
                                if device.pow_resync_chain(args['verbose']):
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

                
