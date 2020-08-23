# fedavg from https://github.com/WHDY/FedAvg/

import os
import sys
import argparse
#from tqdm import tqdm
import numpy as np
import random
import time
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
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
#parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
#parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=1000, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-ns', '--network_stability', type=float, default=0.8, help='the odds a device is online')
parser.add_argument('-gr', '--general_rewards', type=int, default=1, help='rewards for verification of one transaction, mining and so forth')
parser.add_argument('-v', '--verbose', type=int, default=0, help='print verbose debug log')

def check_network_eligibility(check_online=False):
    num_online_workers = 0
    num_online_miners = 0
    num_online_validators = 0
    for worker in workers_this_round:
        if worker.is_online():
            num_online_workers += 1
    for miner in miners_this_round:
        if miner.is_online():
            num_online_miners += 1
    for validator in validators_this_round:
        if validator.is_online():
            num_online_validators += 1
    ineligible = False
    if num_online_workers == 0:
        print('There is no workers online in this round, ', end='')
        ineligible = True
    elif num_online_miners == 0:
        print('There is no miners online in this round, ', end='')
        ineligible = True
    elif num_online_validators == 0:
        print('There is no validators online in this round, ', end='')
        ineligible = True
    if ineligible:
        print('which is ineligible for the network to continue operating.')
        return False
    return True

def register_in_the_network(registrant, check_online=False):
    potential_registrars = set(devices_in_network.devices_set.keys())
    # it cannot register with itself
    potential_registrars.discard(registrant.return_idx())        
    # pick a registrar
    registrar_idx = random.sample(potential_registrars, 1)[0]
    registrar = devices_in_network.devices_set[registrar_idx]
    if check_online:
        if not registrar.is_online():
            online_registrars_idxes = set()
            for registrar_idx in potential_registrars:
                if devices_in_network.devices_set[registrar_idx].is_online():
                    online_registrars_idxes.add(registrar_idx)
            if not online_registrars_idxes:
                return False
            registrar_idx = random.sample(online_registrars_idx, 1)[0]
            registrar = devices_in_network.devices_set[registrar_idx]
    # registrant add registrar to its peer list
    registrant.add_peers(registrar)
    # this device sucks in registrar's peer list
    registrant.add_peers(registrar.return_peers())
    # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
    registrar.add_peers(registrant)
    # remove itself if there is(should not be here)
    # registrant.remove_peers(registrant.return_idx())
    return True

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    # check eligibility
    if args['num_devices'] < 3:
        sys.exit("ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and one validator to start the operation.\nSystem aborted.")

    # make chechpoint save path
    if not os.path.isdir(args['save_path']):
        os.mkdir(args['save_path'])

    # create neural net based on the input model name
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    # assign GPUs if available and prepare nn
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    print(f"{torch.cuda.device_count()} GPUs are available to use!")
    net = net.to(dev)

    # set loss_function and optimizer
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    # create devices in the network
    devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], num_devices=args['num_devices'], network_stability=args['network_stability'], dev=dev)
    test_data_loader = devices_in_network.test_data_loader

    global_weights = net.state_dict()
    for device_seq, device in devices_in_network.devices_set.items():
        # set initial global weights
        device.init_global_weights(global_weights)
        # simulate peer registration, with respect to device idx order 
        register_in_the_network(device)

    # remove its own from peer list if there is
    for device_seq, device in devices_in_network.devices_set.items():
        device.remove_peers(device)

    # debug peers
    if args['verbose']:
        for device_seq, device in devices_in_network.devices_set.items():
            peers = device.return_peers()
            print(f'{device_seq} has peer list ', end='')
            for peer in peers:
                print(peer.return_idx(), end=', ')
            print()

    # FL starts here
    for comm_round in range(args['max_num_comm']):
        print(f"Communicate round {comm_round+1}")
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # assign role first, and then simulate if on or off line
        for device_seq, device in devices_in_network.devices_set.items():
            device.assign_role()
            if device.return_role() == 'worker':
                workers_this_round.append(device)
            elif device.return_role() == 'miner':
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
            if device.online_switcher():
                if args['verbose']:
                    print(f'{device.return_idx()} {device.return_role()} is online')
        if not check_network_eligibility():
            print("Go to the next round.\n")
            continue
        # shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)
        if args['verbose']:
            print(f"There are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")

        # re-init round vars
        for miner in miners_this_round:
            if miner.is_online():
                miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            if worker.is_online():
                worker.worker_reset_vars_for_new_round()

        # incase no device is online for this communication round
        #  no_device_online = False
        
        # workers, miners and validators take turns to perform jobs
        # workers
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            if worker.is_online():
                # update peer list
                if not worker.update_peer_list():
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        go_to_next_round = True
                        break
                # PoW resync chain
                worker.pow_resync_chain()
                # worker perform local update
                print(f"This is {device.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} performing local updates...")
                worker.worker_local_update(args['batchsize'], net, loss_func, opti, global_weights)
                # worker associates with a miner
                associated_miner_idx = worker.associate_with_miner()
                if not associated_miner_idx:
                    print("Cannot find a miner in its peer list.")
                    continue
                associated_miner = devices_in_network.devices_set[associated_miner_idx]
                if associated_miner.is_online():
                    associated_miner.add_worker_to_association(worker)
                else:
                    worker.remove_peers(associated_miner_idx)
                # may go offline during model updates transmission
                worker.online_switcher() 
        
        if not check_network_eligibility():
            print("Go to the next round.\n")
            continue
        
        # miners accept local updates and broadcast to other miners
        for miner in miners_this_round:
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list():
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        go_to_next_round = True
                        break
                # PoW resync chain
                miner.pow_resync_chain()
                # miner accepts local updates from its workers association
                print(f"This is {miner.return_idx()} - miner {miner_iter+1}/{len(workers_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = miner.return_associated_workers()
                if not associated_workers:
                    print("No workers are assigned with this miner for this communication round.")
                    continue
                for worker in miner.return_associated_workers():
                    if worker.is_online():
                        miner.add_unconfirmmed_transaction({'worker_device_id': worker.return_idx(), 'local_updates': worker.return_local_updates()})
                    else:
                        potential_offline_workers.add(worker)
                miner.remove_peers(potential_offline_workers)
                if not miner.return_unconfirmmed_transactions():
                    print("Workers disconnected while transmitting updates.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher():
                    miner.broadcast_updates()
                if miner.is_online():
                    miner.online_switcher()
        if not check_network_eligibility():
            print("Go to the next round.\n")
            continue

        # miners do self and cross-validation(only validating signature at this moment)
        # time spent included in the block_generation_time
        block_generation_time_spent = {}
        for miner in miners_this_round:
            candidate_block = Block(idx=miner.blockchain.return_chain_length())
            if miner.is_online():
                start_time = time.time()
                # self verification
                for unconfirmmed_transaction in miner.return_unconfirmmed_transactions():
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['rewards'] = args["general_rewards"]
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        miner.receive_rewards(args["general_rewards"])
                # cross verification
                for unconfirmmed_transaction in miner.return_broadcasted_transactions():
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['rewards'] = args["general_rewards"]
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        miner.receive_rewards(args["general_rewards"]) 
                # mine the block
                if candidate_block.return_transactions():
                    # return the last block and add previous hash
                    last_block = miner.blockchain.return_last_block()
                    if last_block is None:
                        # mine the genesis block
                        candidate_block.set_previous_hash(None)
                    else:
                        candidate_block.set_previous_hash(last_block.compute_hash(hash_whole_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.proof_of_work(candidate_block)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline
                if miner.online_switcher():
                    # record mining time
                    block_generation_time_spent[miner] = (time.time() - start_time)/(miner.return_computation_power())
                    mined_block.set_mining_rewards(args["general_rewards"])
                    miner.receive_rewards(args["general_rewards"])
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)

        if not check_network_eligibility():
            print("Go to the next round.\n")
            continue
        # select the winning miner and broadcast its mined block
        winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        block_to_propagate = miner.return_mined_block()

        # miner propogate the winning block (just let other miners receive it, verify it and add to the blockchain)
        for miner in miners_this_round:
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                miner.receive_propagated_block(block_to_propagate)
                if miner.verify_and_add_block(miner.return_propagated_block()):
                    pass
                else:
                    miner.toss_propagated_block()
                    print("Received propagated block is invalid. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_swticher()
        
        # miner requests worker to download block
        for miner in miners_this_round:
            if miner.is_online():
                if miner.return_propagated_block():
                    for worker in miner.return_associated_workers():
                        block_to_send = miner.blockchain.get_last_block()
                        if worker.online():
                            worker.receive_block_from_miner(block_to_send)
                            if worker.verify_and_add_block(block_to_send):
                                pass
                            else:
                                worker.toss_received_block()
                                print("Received block from the associated miner is not valid. Pass to the next worker.")
                            worker.online_switcher()
                            
        if not check_network_eligibility():
            print("Go to the next round.\n")
            continue
        # miner requests worker to download block
        for worker in workers_this_round:
            if worker.is_online():
                if worker.return_received_block_from_miner():
                    block_to_operate = worker.blockchain.return_last_block()
                    # avg the gradients
                    transactions = block_to_operate.return_transactions()

        # TODO
        '''
        miner
        3. average gradients
        worker
        1. download and update to global model
        2. exe smart contract - use its own data to calculate accuracy
        3. send accuracy to validator
        validator
        1. calculate belief degree
        2. record interaction frequency
        3. calculate final trustworthiness of the worker
        '''
        

                
            
               

                
