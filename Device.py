# fedavg from https://github.com/WHDY/FedAvg/

import os
import argparse
#from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DevicesInNetwork
from block import Block
from blockchain import Blockchain

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nd', '--num_of_devices', type=int, default=100, help='numer of the devices in the simulation network')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
#parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
#parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=1000, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-ns', '--network_stability', type=float, default=0.7, help='the odds a device is online')
parser.add_argument('-v', '--verbose', type=int, default=0, help='print verbose debug log')

def register_in_the_network(registrant_idx, registrant, num_of_devices_in_network, check_online=False):
    # device index starts from 1
    registrar_idx = random.randint(1, num_of_devices_in_network)
    registrar = devices_in_network.devices_set[f'device_{registrar_idx}']
    if check_online:
        all_devices = set([f'device_{i}' for i in range(1, num_of_devices_in_network+1)])
        while not registrar.is_online():
            all_devices.remove(registrar)
            if not all_devices:
                return False
            registrar_idx = random.sample(all_devices, 1)[0]
            registrar = devices_in_network.devices_set[registrar_idx]
    # registrar add this device to its peer list
    registrar.add_peers(registrant_idx)
    # this device sucks in registrar's peer list
    registrant.add_peers(registrar.get_peers())
    return True

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

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
    devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], num_of_devices=args['num_of_devices'], network_stability=args['network_stability'], dev=dev)
    test_data_loader = myClients.test_data_loader

    global_weights = net.state_dict()
    for device_seq, device in devices_in_network.devices_set.items():
        # set initial global weights
        device.init_global_weights(global_weights)
        # simulate peer registration, with respect to device idx order 
        register_in_the_network(device_seq, device, args['num_of_devices'])
    
    # debug peers
    if args['verbose']:
        for device_seq, device in devices_in_network.devices_set.items():
            print(f'{device_seq} has peer list {device.get_peers()}')

    # FL starts here
    for comm_round in range(args['max_num_comm']):
        print(f"Communicate round {comm_round+1}")
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # assign role first, and then simulate if on or off line
        for device_seq, device in devices_in_network.devices_set.items():
            device.assign_role()
            device.online_switcher()
            if device.get_role() == 'w':
                workers_this_round.append(device)
            elif device.get_role() == 'm':
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
        # shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)
        if args['verbose']:
            print(f"There are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")

        # re-init round vars
        for miner in miners_this_round:
            miner.clear_worker_association()
            miner.clear_broadcasted_transactions()
            miner.clear_unconfirmmed_transactions()
        # incase no device is online for this communication round
        no_device_online = False
        
        # workers, miners and validators take turns to perform jobs
        # workers
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            if worker.is_online():
                # update peer list
                if not worker.update_peer_list():
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(work.get_idx(), worker, args['num_of_devices'], check_online=True):
                        print("No devices found in the network online in this communication round.")
                        no_device_online = True
                        break
                # PoW resync chain
                worker.pow_resync_chain()
                # worker perform local update
                print(f"This is device {device.get_idx()} - worker {worker_iter+1}/{len(workers_this_round)} performing local updates...")
                worker.worker_local_update(args['batchsize'], net, loss_func, opti, global_parameters)
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
        # to save time, other devices won't check; jump to the next round
        if no_device_online:
            continue
        
        # miners accept local updates and broadcast to other miners
        block_generation_time
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list():
                    # peer_list_empty, randomly register with a online node
                    register_in_the_network(miner.get_idx(), miner, args['num_of_devices'], check_online=True)
                # PoW resync chain
                miner.pow_resync_chain()
                # miner accepts local updates from its workers association
                print(f"This is device {miner.get_idx()} - miner {miner_iter+1}/{len(workers_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = miner.get_associated_workers()
                if not associated_workers:
                    print("No workers are assigned with this miner for this communication round.")
                    continue
                for worker in miner.get_associated_workers():
                    if worker.is_online():
                        miner.add_unconfirmmed_transaction({'worker_device_id': worker.get_idx(), 'local_updates': worker.get_local_updates()})
                    else:
                        potential_offline_workers.add(worker)
                miner.remove_peers(potential_offline_workers)
                if not miner.get_unconfirmmed_transactions():
                    print("Workers disconnected while transmitting updates.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher():
                    miner.broadcast_updates()
                miner.online_switcher()

        # miners do self and cross-validation(only validating signature at this moment)
        # time spent included in the block_generation_time
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            candidate_block
            if miner.is_online():
                # self verification
                for unconfirmmed_transaction in miner.get_unconfirmmed_transactions():
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        
                    

                
