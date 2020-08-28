import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
import random
import copy
# https://cryptobook.nakov.com/digital-signatures/rsa-sign-verify-examples
from Crypto.PublicKey import RSA
from hashlib import sha256
from Blockchain import Blockchain

class Device:
    def __init__(self, idx, assigned_train_ds, assigned_test_ds, local_batch_size, loss_func, opti, network_stability, net, dev):
        self.idx = idx
        self.train_ds = assigned_train_ds
        self.test_ds = assigned_test_ds
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.opti = opti
        self.network_stability = network_stability
        self.net = net
        self.dev = dev
        self.train_dl = None
        self.test_dl = None
        self.local_train_parameters = None
        self.global_parameters = None
        # used to assign role to the device
        self.role = None
        ''' simulating hardware equipment strength, such as good processors and RAM capacity
        # for workers, meaning the number of epochs it can perform for a communication round
        # for miners, its PoW time will be shrink by this value of times
        # for validators, haha! # TODO
        '''
        self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.on_line = False
        self.rewards = 0
        self.blockchain = Blockchain()
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        # a flag to phase out this device for a certain comm round(simulation friendly)
        # self.phased_out = False
        ''' For workers '''
        self.received_block_from_miner = None
        ''' For miners '''
        self.associated_worker_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        # self.block_to_add = None

    ''' Common Methods '''
    def return_idx(self):
        return self.idx

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def init_global_parameters(self):
        self.global_parameters = self.net.state_dict()

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def return_peers(self):
        return self.peer_list
    
    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def assign_role(self):
        # equal probability
        # TODO give validator back later
        role_choice = random.randint(0, 1)
        if role_choice == 0:
            self.role = "worker"
        elif role_choice == 1:
            self.role = "miner"
        else:
            self.role = "validator"
        
    def return_role(self):
        return self.role

    def online_switcher(self):
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.on_line = True
        else:
            self.on_line = False
        return self.on_line

    def is_online(self):
        return self.on_line

    # def phase_out(self):
    #     self.phased_out = True

    # def reset_phase_out(self):
    #     self.phased_out = True

    # def is_not_phased_out(self):
    #     return self.phased_out
    
    def update_peer_list(self):
        online_peers = set()
        offline_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
            else:
                offline_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove offline peers
        self.remove_peers(offline_peers)
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def return_blockchain_object(self):
        return self.blockchain

    def check_pow_proof(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check.return_pow_proof()
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * Blockchain.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_whole_block=True):
                    pass
                else:
                    # if not self.check_pow_proof(block):
                    #     print("block_to_check string")
                    #     print(str(sorted(block.__dict__.items())).encode('utf-8'))

                    # print(block.return_previous_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_whole_block=True))
                    # print(f"index {chain_to_check.index(block) - 1}")
                    # print("pre", block.return_previous_hash())
                    # print(chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_whole_block=True))
                    return False
        return True

    def pow_resync_chain(self):
        longest_chain = None
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                curr_chain_len = self.return_blockchain_object().return_chain_length()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
        if longest_chain:
            self.blockchain.replace_chain(longest_chain.return_chain_structure())
            print(f"{self.return_idx()} chain resynced")

    def receive_rewards(self, rewards):
        self.rewards += rewards
    
    def return_computation_power(self):
        return self.computation_power

    ''' Worker '''
    # TODO change to computation power
    def worker_local_update(self):
        print(f"computation power {self.computation_power}, performing {self.computation_power} epoch(s)")
        self.net.load_state_dict(self.global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        for epoch in range(self.computation_power):
            # print(f"epoch {epoch+1}")
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
        print("Done")
        self.local_train_parameters = self.net.state_dict()

    def return_local_updates_and_signature(self):
        return {"local_updates_params": self.local_train_parameters, "signature": self.sign_updates()}

    def associate_with_miner(self):
        online_miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == 'miner':
                    online_miners_in_peer_list.add(peer)
        if not online_miners_in_peer_list:
            return False
        self.worker_associated_miner = random.sample(online_miners_in_peer_list, 1)[0]
        return self.worker_associated_miner

    def sign_updates(self):
        return {"pub_key": self.public_key, "modulus": self.modulus, "signature": self.sign_msg(self.local_train_parameters)}

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None

    def receive_block_from_miner(self, received_block):
        self.received_block_from_miner = copy.deepcopy(received_block)

    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner

    def global_update(self, num_participants, sum_parameters):
        for var in self.global_parameters:
            self.global_parameters[var] = (sum_parameters[var] / num_participants)
    
    def evaluate_updated_weights(self):
        with torch.no_grad():
            self.net.load_state_dict(self.global_parameters, strict=True)
            sum_accu = 0
            num = 0
            self.test_dl = DataLoader(self.test_ds, batch_size=self.local_batch_size, shuffle=True)
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            return sum_accu / num

    ''' miner '''
    def add_worker_to_association(self, worker_device):
        self.associated_worker_set.add(worker_device)

    def return_associated_workers(self):
        return self.associated_worker_set
    
    def add_unconfirmmed_transaction(self, unconfirmmed_transaction):
        self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def accept_broadcasted_transactions(self, broadcasted_transactions):
        self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))

    def broadcast_updates(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == 'miner':
                    peer.accept_broadcasted_transactions(self.unconfirmmed_transactions)

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def sign_block(self, mined_block):
        mined_block.add_signature(self.sign_msg(mined_block.__dict__))

    def verify_transaction_by_signature(self, transaction_to_verify):
        local_updates_params = transaction_to_verify['local_updates']["local_updates_params"]
        modulus = transaction_to_verify['local_updates']["signature"]["modulus"]
        pub_key = transaction_to_verify['local_updates']["signature"]["pub_key"]
        signature = transaction_to_verify['local_updates']["signature"]["signature"]
        # verify
        hash = int.from_bytes(sha256(str(local_updates_params).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.return_idx())
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        while not current_hash.startswith('0' * Blockchain.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_hash(current_hash)
        return candidate_block

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def receive_propagated_block(self, received_propagated_block):
        self.received_propagated_block = copy.deepcopy(received_propagated_block)
    
    def return_propagated_block(self):
        return self.received_propagated_block
        
    def toss_propagated_block(self):
        self.received_propagated_block = None

    def verify_block(self, block_to_verify):
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
            return False
        last_block = self.blockchain.return_last_block()
        if last_block is not None:
            # check if the previous_hash referred in the block and the hash of latest block in the chain match.
            last_block_hash = last_block.compute_hash(hash_whole_block=True)
            if block_to_verify.return_previous_hash() != last_block_hash:
                # debug
                #print("last_block")
                #print(str(sorted(last_block.__dict__.items())).encode('utf-8'))
                #print("block_to_verify")
                #print(str(sorted(block_to_verify.__dict__.items())).encode('utf-8'))
                #debug
                return False
        # All verifications done.
        # ???When syncing by calling consensus(), rebuilt block doesn't have this field. add the block hash after verifying
			# block_to_verify.set_hash()
        return block_to_verify

    def add_block(self, block_to_add):
        self.blockchain.append_block(block_to_add)

    # def set_block_to_add(self, block_to_add):
    #     self.block_to_add = block_to_add

    # def return_block_to_add(self):
    #     return self.block_to_add
    
    def miner_reset_vars_for_new_round(self):
        self.associated_worker_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
#        self.block_to_add = None


class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, batch_size, loss_func, opti, num_devices, network_stability, net, dev):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = {}
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        # distribute dataset
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the devices
    def data_set_balanced_allocation(self):
        # read dataset
        mnist_dataset = DatasetLoad(self.data_set_name, self.is_iid)
        # perpare training data
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label
        # perpare test data
        #test_data = torch.tensor(mnist_dataset.test_data)
        #test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
        # self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)
        test_data = mnist_dataset.test_data
        test_label = mnist_dataset.test_label
        # shard dataset and distribute among devices
        shard_size = mnist_dataset.train_data_size // self.num_devices // 2
        shards_id = np.random.permutation(mnist_dataset.train_data_size // shard_size)
        for i in range(self.num_devices):
            # make it more random by introducing two shards
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            data_shards1 = test_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = test_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = test_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = test_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_test_label = np.argmax(local_test_label, axis=1)
            # assign data to a device and put in the devices set
            device_idx = f'device_{i+1}'
            a_device = Device(device_idx, TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label)), TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), self.batch_size, self.loss_func, self.opti, self.default_network_stability, self.net, self.dev)
            # device index starts from 1
            self.devices_set[device_idx] = a_device