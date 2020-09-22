import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
from DatasetLoad import AddGaussianNoise
import random
import copy
import time
# https://cryptobook.nakov.com/digital-signatures/rsa-sign-verify-examples
from more_itertools import split_when
from Crypto.PublicKey import RSA
from hashlib import sha256
from Blockchain import Blockchain

class Device:
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, loss_func, opti, network_stability, net, dev, validator_acception_wait_time, miner_acception_wait_time, miner_accepted_transactions_size_limit, even_link_speed, even_computation_power, is_malicious, knock_out_rounds):
        self.idx = idx
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.opti = opti
        self.network_stability = network_stability
        self.net = net
        self.dev = dev
        self.train_dl = None
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None
        # used to assign role to the device
        self.role = None
        if even_link_speed:
            self.link_speed = 1
        else:
            self.link_speed = random.random()
        ''' simulating hardware equipment strength, such as good processors and RAM capacity. Following recorded times will be shrunk by this value of times
        # for workers, its update time
        # for miners, its PoW time
        # for validators, its validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster devices' transactions will be accepted and verified first
        '''
        if even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.on_line = True
        self.rewards = 0
        self.blockchain = Blockchain()
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        # a flag to phase out this device for a certain comm round(simulation friendly)
        # self.phased_out = False
        # for validation purpose
        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.is_malicious = is_malicious
        ''' For workers '''
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        ''' For miners '''
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        # self.block_to_add = None
        ''' For validators '''
        self.validator_associated_worker_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        self.validator_acception_wait_time = validator_acception_wait_time
        # self.validator_sig_validated_transactions_size_limit = validator_sig_validated_transactions_size_limit
        #self.sig_verified_transactions = None or []
        #self.broadcasted_sig_verified_transactions = None or []
        self.unordered_arrival_time_direct_worker_transactions = None
        self.accepted_broadcasted_validator_transactions = None or []
        self.final_transactions_queue_to_validate = None
        self.sig_verified_transactions_queue = None or []
        # self.validator_size_stop = validator_size_stop
        # self.unconfirmmed_validator_transactions = None or []
        # self.accepted_broadcasted_validator_transactions = None or []
        

    ''' Common Methods '''
    def return_idx(self):
        return self.idx

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def init_global_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
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
        role_choice = random.randint(0, 2)
        if role_choice == 0:
            self.role = "worker"
        elif role_choice == 1:
            self.role = "miner"
        else:
            self.role = "validator"

    # used for hard_assign
    def assign_miner_role(self):
        self.role = "miner"

    def assign_worker_role(self):
        self.role = "worker"

    def assign_validator_role(self):
        self.role = "validator"   
        
    def return_role(self):
        return self.role

    def online_switcher(self):
        self.old_status = self.on_line
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.on_line = True
        else:
            print(f"{self.idx} goes offline.")
            self.on_line = False
        return self.on_line

    def is_back_online(self):
        if self.old_status == False and self.old_status != self.on_line:
            print(f"{self.idx} goes back online.")
            return True

    def is_online(self):
        return self.on_line

    def is_malicious(self):
        return self.is_malicious()

    # def phase_out(self):
    #     self.phased_out = True

    # def reset_phase_out(self):
    #     self.phased_out = True

    # def is_not_phased_out(self):
    #     return self.phased_out

    def return_black_list(self):
        return self.black_list
    
    def update_peer_list(self, verbose=False):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
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
        # remove malicious peers
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        if verbose:
            if old_peer_list == self.peer_list:
                print("Peer list NOT changed.")
            else:
                print("Peer list has been changed.")
                removed_peers = offline_peers.union(potential_malicious_peer_set)
                added_peers = self.peer_list.difference(old_peer_list)
                if removed_peers:
                    print("These peers are removed")
                    for peer in removed_peers:
                        print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                    print()
                if added_peers:
                    print("These peers are added")
                    for peer in added_peers:
                        print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                    print()
                print("Final peer list:")
                for peer in self.peer_list:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
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
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    # if not self.check_pow_proof(block):
                    #     print("block_to_check string")
                    #     print(str(sorted(block.__dict__.items())).encode('utf-8'))

                    # print(block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True))
                    # print(f"index {chain_to_check.index(block) - 1}")
                    # print("pre", block.return_previous_block_hash())
                    # print(chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True))
                    return False
        return True

    def pow_resync_chain(self, verbose=False):
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need machenism to reverse updates by # of blocks
            # chain_structure_before_replace = self.return_blockchain_object().return_chain_structure()
            # for block_iter in range(len(chain_structure_before_replace)):
            #     if chain_structure_before_replace[block_iter].compute_hash(hash_entire_block=True) != longest_chain_structure[block_iter].compute_hash(hash_entire_block=True):
            #         break
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            if verbose:
                print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True
        if verbose:
            print("Chain not resynced.")
        return False

    def receive_rewards(self, rewards):
        self.rewards += rewards
    
    def return_computation_power(self):
        return self.computation_power

    def verify_block(self, block_to_verify, source_miner):
        if source_miner in self.black_list:
            print(f"The miner sending this block {source_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False
        if block_to_verify.return_mined_by() in self.black_list:
            print(f"The miner {block_to_verify.return_mined_by()} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.idx} is not verified.")
            return False
        # check if the signature is valid
        signature_dict = block_to_verify.return_miner_pub_key()
        modulus = signature_dict["modulus"]
        pub_key = signature_dict["pub_key"]
        signature = block_to_verify.return_signature()
        # verify signature
        block_to_verify_before_sign = copy.deepcopy(block_to_verify)
        block_to_verify_before_sign.remove_signature_for_verification()
        hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        if hash != hashFromSignature:
            print(f"Signature of the block from miner {self.idx} is not verified.")
            return False
        last_block = self.return_blockchain_object().return_last_block()
        if last_block is not None:
            # check if the previous_hash referred in the block and the hash of latest block in the chain match.
            last_block_hash = last_block.compute_hash(hash_entire_block=True)
            if block_to_verify.return_previous_block_hash() != last_block_hash:
                print(f"Block from miner {self.idx} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. Not verified.")
                # debug
                #print("last_block")
                #print(str(sorted(last_block.__dict__.items())).encode('utf-8'))
                #print("block_to_verify")
                #print(str(sorted(block_to_verify.__dict__.items())).encode('utf-8'))
                #debug
                return False
        # All verifications done.
        # ???When syncing by calling consensus(), rebuilt block doesn't have this field. add the block hash after verifying
			# block_to_verify.set_pow_proof()
        print(f"Block accepted from miner {source_miner} mined by {block_to_verify.return_mined_by()} has been verified!")
        return block_to_verify

    def add_block(self, block_to_add):
        if block_to_add.is_validator_block():
            last_block_on_validator_chain = self.return_blockchain_object().return_last_block()
            # check last block
            if last_block_on_validator_chain == None or last_block_on_validator_chain.is_validator_block():
                print("validator block cannot be appended to a validator block.")
                return False
        self.return_blockchain_object().append_block(block_to_add)
        block_type = "validator_block" if block_to_add.is_validator_block() else "worker_block"
        print(f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a {block_type} to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        self.has_added_block = True
        return True

    def reset_received_block_from_miner_vars(self):
        self.has_added_block = False
        self.received_block_from_miner = None

    def return_has_added_block(self):
        return self.has_added_block        

    def operate_on_validator_block(self, passed_in_validator_block=None):
        old_black_list = copy.copy(self.black_list)
        if not passed_in_validator_block:
            block_to_operate = self.return_blockchain_object().return_last_block()
        else:
            # use when chain resynced
            block_to_operate = passed_in_validator_block
        if block_to_operate.return_mined_by() in self.black_list:
            print(f"Block miner {block_to_operate.return_mined_by()} is in {self.idx}'s black list. operate_on_validator_block terminated.")
        if not block_to_operate.is_validator_block():
            print(f"Block {block_to_operate.return_block_idx()} is a worker block. Skip getting validations from this block.")
            return
        voted_worker_accuracy_this_round = {}
        transactions = block_to_operate.return_transactions()
        for transaction in transactions:
            if transaction['validator_device_idx'] in self.black_list:
                print(f"Validator {transaction['validator_device_idx']} of this transaction is in {self.idx}'s black list. This transaction is ignored.")
                continue
            else:
                print(f"Processing transaction from validator {transaction['validator_device_idx']}...")
            accuracies_this_round = transaction['accuracies_this_round']
            for worker_idx, worker_accuracy in accuracies_this_round.items():
                # voting system
                if not worker_idx in voted_worker_accuracy_this_round.keys():
                    voted_worker_accuracy_this_round[worker_idx] = {}
                    voted_worker_accuracy_this_round[worker_idx][worker_accuracy] = 1
                else:
                    if worker_accuracy in  voted_worker_accuracy_this_round[worker_idx].keys():
                        voted_worker_accuracy_this_round[worker_idx][worker_accuracy] += 1
                    else:
                        voted_worker_accuracy_this_round[worker_idx][worker_accuracy] = 1
        # recognize the accuracy recorded by most of the validators
        for worker_idx, worker_accuracy in voted_worker_accuracy_this_round.items():
            accuracy_recorded_by_most_validators = list({k: v for k, v in sorted(worker_accuracy.items(), key=lambda item: item[1], reverse=True)})[0]
            if not worker_idx in self.worker_accuracy_accross_records.keys():
                self.worker_accuracy_accross_records[worker_idx] = []
            self.worker_accuracy_accross_records[worker_idx].append(accuracy_recorded_by_most_validators)
        # check and add to black_list
        for worker_idx, worker_accuracy in self.worker_accuracy_accross_records.items():            
            if not worker_idx in self.black_list:
                decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
                for decreasing_accuracie in decreasing_accuracies:
                    if len(decreasing_accuracie) >= self.knock_out_rounds:
                        self.black_list.add(worker_idx)
                        print(f"{worker_idx} has been put in {self.idx}'s black list")
                        break
                        # add malicious devices to the black list
        new_black_list_peers = self.black_list.difference(old_black_list)
        print(f"validator block {block_to_operate.return_block_idx()} has been processed.")
        if new_black_list_peers:
            print("These peers are put into the black_list")
            for peer in new_black_list_peers:
                print(f"d_{peer.split('_')[-1]}", end=', ')
            print()
        else:
            print("No new peers are put into the black_list")


    # def update_model_after_chain_resync(self, chain_diff_at_index):
    #     for block in self.return_blockchain_object().return_chain_structure()[chain_diff_at_index:]:
    #         if not block.is_validator_block():
    #             self.global_update(passed_in_block_to_operate=block)
    #         else:
    #             self.operate_on_validator_block(block)
    def update_model_after_chain_resync(self):
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            if not block.is_validator_block():
                self.global_update(passed_in_block_to_operate=block)
            else:
                self.operate_on_validator_block(passed_in_validator_block=block)

    ''' Worker '''
    def add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                m.weight.add_(torch.randn(m.weight.size()) * 0.1)
                
    # TODO change to computation power
    def worker_local_update(self, rewards, local_epochs=1):
        print(f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed,3)}")
        self.net.load_state_dict(self.global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        # for epoch in range(self.computation_power):
            # print(f"epoch {epoch+1}")
        self.local_update_time = time.time()
        # local worker update by specified epochs
        # usually, if validator acception time is specified, local_epochs should be 1
        self.local_updates_rewards_per_transaction = 0
        for epoch in range(local_epochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                self.local_updates_rewards_per_transaction += rewards * (label.shape[0])
                # self.receive_rewards(rewards)
                # should calculate rewards only in the block. do it when block is appended
            self.local_total_epoch += 1
        # local update one epoch done
        try:
            self.local_update_time = (time.time() - self.local_update_time)/self.computation_power
        except:
            self.local_update_time = float('inf')
        # self.net.apply(self.add_noise_to_weights)
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.net.state_dict()
        return self.local_update_time
        
    # def return_local_update_time(self):
    #     return self.local_update_time

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round, "local_updates_params": copy.deepcopy(self.local_train_parameters), "local_updates_rewards": self.local_updates_rewards_per_transaction, "local_iteration(s)_spent_time": self.local_update_time, "local_total_accumulated_epochs_this_round": self.local_total_epoch, "worker_rsa_pub_key": self.return_rsa_pub_key()}
        local_updates_dict["worker_signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        return local_updates_dict

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0

    # def associate_with_validator(self):
    #     validators_in_peer_list = set()
    #     for peer in self.peer_list:
    #         if peer.return_role() == "validator":
    #             if not peer.return_idx() in self.black_list:
    #                 validators_in_peer_list.add(peer)
    #     if not validators_in_peer_list:
    #         return False
    #     self.worker_associated_validator = random.sample(validators_in_peer_list, 1)[0]
    #     return self.worker_associated_validator

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(f"Either the block sending miner {source_miner} or the miner {eceived_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")
            

    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner
    
    def evaluate_model_weights(self, weights_to_eval=None):
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)
            else:
                self.net.load_state_dict(self.global_parameters, strict=True)
            sum_accu = 0
            num = 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            return sum_accu / num

    def global_update(self, passed_in_block_to_operate=None):
        if not passed_in_block_to_operate:
            block_to_operate = self.return_blockchain_object().return_last_block()
        else:
            # use when chain resynced
            block_to_operate = passed_in_block_to_operate
        if block_to_operate.return_mined_by() in self.black_list:
            print(f"Block miner {block_to_operate.return_mined_by()} is in {self.idx}'s black list. global_update terminated.")
        if block_to_operate.is_validator_block():
            print(f"Block {block_to_operate.return_block_idx()} is a validator block. Skip global model updates for this block.")
            return
        # avg the gradients
        sum_parameters = None
        # TODO verify transaction??
        transactions = block_to_operate.return_transactions()
        for transaction in transactions:
            if transaction['worker_device_idx'] in self.black_list:
                print(f"Worker {transaction['worker_device_idx']} of this transaction is in {self.idx}'s black list. This transaction is ignored.")
                continue
            if self.verify_transaction_by_signature(transaction, ignore_miner=True):
                # print(f"The signature of the transaction from worker {transaction['worker_device_idx']} has been veified. Processing...")
                print("Transaction processing...")
            else:
                # print(f"The signature of the transaction from worker {transaction['worker_device_idx']} is NOT veified. Ignored this transaction.")
                print("Transaction ignored.")
            local_updates_params = transaction['local_updates_params']
            if sum_parameters is None:
                sum_parameters = copy.deepcopy(local_updates_params)
            else:
                for var in sum_parameters:
                    sum_parameters[var] += local_updates_params[var]
        if not sum_parameters:
            # seems all transactions are from malicious nodes
            return
        num_participants = len(transactions)
        for var in self.global_parameters:
            self.global_parameters[var] = (sum_parameters[var] / num_participants)
        print(f"global updates done by block {block_to_operate.return_block_idx()}")
    
    ''' miner '''

    def add_validator_to_association(self, validator_device):
        if not validator_device.return_idx() in self.black_list:
            self.associated_validator_set.add(validator_device)
        else:
            print(f"WARNING: {validator_device.return_idx()} in miner {self.idx}'s black list. Not added by miner.")

    # def return_associated_workers(self):
    #     return self.associated_worker_set
    
    def return_associated_validators(self):
        return self.associated_validator_set
    

    # def add_unconfirmmed_validator_transactions(self, unconfirmmed_validator_transaction):
    #     self.unconfirmmed_validator_transactions.append(copy.deepcopy(unconfirmmed_validator_transaction))

    # def return_unconfirmmed_validator_transactions(self):
    #     return self.unconfirmmed_validator_transactions

    

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    # TODO may have to distinguish btw miner and validator
    def verify_transaction_by_signature(self, transaction_to_verify, ignore_validator=False, ignore_miner=False):
        try:
            transaction_device_idx = transaction_to_verify['worker_device_idx']
        except:
            transaction_device_idx = transaction_to_verify['validator_device_idx']
        if transaction_device_idx in self.black_list:
            print(f"{transaction_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        transaction_before_signed = copy.deepcopy(transaction_to_verify)
        del transaction_before_signed["signature"]
        if ignore_miner:
            del transaction_before_signed["tx_verified_by"]
            del transaction_before_signed["mining_rewards"]
        if ignore_validator:
            del transaction_before_signed["initial_sig_verified_by"]
            del transaction_before_signed["sig_verification_rewards"]
        modulus = transaction_to_verify['rsa_pub_key']["modulus"]
        pub_key = transaction_to_verify['rsa_pub_key']["pub_key"]
        signature = transaction_to_verify["signature"]
        # verify
        hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        if hash == hashFromSignature:
            print(f"Signature of transaction from {transaction_device_idx} is verified by {self.role} {self.idx}!")
            return True
        else:
            print(f"Signature invalid. Transaction from {transaction_device_idx} is NOT verified.")
            return False
    
    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        while not current_hash.startswith('0' * Blockchain.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(received_propagated_block)
            print(f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(received_propagated_validator_block)
            print(f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")
    
    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block
        
    def toss_propagated_block(self):
        self.received_propagated_block = None
        
    def toss_ropagated_validator_block(self):
        self.received_propagated_validator_block = None

    # def set_block_to_add(self, block_to_add):
    #     self.block_to_add = block_to_add

    # def return_block_to_add(self):
    #     return self.block_to_add
    
    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmmed_validator_transactions.clear()
        # self.accepted_broadcasted_validator_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
#        self.block_to_add = None

    # def miner_reset_vars_for_new_validation_round(self):
    #     self.unconfirmmed_transactions.clear()
    #     self.broadcasted_transactions.clear()

    ''' Validator '''
    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.has_added_block = False
        self.validator_associated_miner = None
        self.validator_associated_worker_set.clear()
        #self.sig_verified_transactions.clear()
        #self.broadcasted_sig_verified_transactions.clear()
        self.unordered_arrival_time_direct_worker_transactions = None
        self.final_transactions_queue_to_validate = None
        self.accepted_broadcasted_validator_transactions.clear()
        self.sig_verified_transactions_queue.clear()

    def add_sig_verified_transaction_to_queue(self, transaction_to_add):
        self.sig_verified_transactions_queue.append(transaction_to_add)
    
    def return_sig_verified_transactions_queue(self):
        return self.sig_verified_transactions_queue

    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list

    def accept_accuracy(self, worker, effort_rewards):
        worker_idx = worker.return_idx()
        if not worker_idx in self.black_list:
            worker_accuracy = worker.return_accuracy_this_round()
            # record in transaction
            self.accuracies_this_round[worker_idx] = worker_accuracy
            print(f"Worker {worker.return_idx()}'s accuracy {worker_accuracy} has been recorded by the validator {self.idx}.")
            self.validation_rewards_this_round += 1
            self.receive_rewards(effort_rewards)
        else:
            print(f"Worker {worker.return_idx()} is in the black list of validator {self.idx}. Accuracy not accepted.")
            # record in its own cache
            # if worker_idx in self.worker_accuracy_accross_records.keys():
            #     self.worker_accuracy_accross_records[worker_idx].append(worker_accuracy)
            #     self.receive_rewards(effort_rewards)
            # else:
            #     self.worker_accuracy_accross_records[worker_idx] = [worker_accuracy]
            #     self.receive_rewards(effort_rewards)
    
    def validate_worker_local_update(unconfirmmed_transaction, self):
        pass
        # worker_local_update_params = unconfirmmed_transaction["local_updates_params"]
        # # current global model accuracy
        # validator_net = copy.deepcopy(self.net)
        # dev = copy.deepcopy(self.dev)
        # validator_net.load_state_dict(self.global_parameters, strict=True)
        # sum_accu = 0
        # num = 0
        # for data, label in self.test_dl:
        #     data, label = data.to(dev), label.to(dev)
        #     preds = validator_net(data)
        #     preds = torch.argmax(preds, dim=1)
        #     sum_accu += (preds == label).float().mean()
        #     num += 1
        # current_accuracy = sum_accu / num
        # # accuracy evaluated by worker's update
        # # apply updates to the current global params
        # worker_new_global_params = 
        # sum_parameters = None

        # evaludated_worker_accuracy_by_own_dataset = 

    # def record_worker_performance_in_block(self, validator_candidate_block, comm_round, rewards):
    #     kick_out_device_set = set()
    #     validation_effort_rewards = 0
    #     from more_itertools import split_when
    #     for worker_idx, worker_accuracy in self.worker_accuracy_accross_records.items():
    #         decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
    #         for decreasing_accuracie in decreasing_accuracies:
    #             if decreasing_accuracie >= self.knock_out_rounds:
    #                 kick_out_device_list.add(worker_idx)
    #                 break
    #         self.receive_rewards(rewards)
    #         validation_effort_rewards += rewards
    #     validator_candidate_block.add_validator_transaction({'validator_idx': self.idx, 'round_number': comm_round, 'accuracies_this_round': self.accuracies_this_round, 'kicked_out_devices': kick_out_device_set, 'validation_effort_rewards': validation_effort_rewards})

    def return_validations_and_signature(self, comm_round):
        #kick_out_device_set = set()
        #validation_effort_rewards = 0
        # from more_itertools import split_when
        # for worker_idx, worker_accuracy in self.worker_accuracy_accross_records.items():
        #     decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
        #     for decreasing_accuracie in decreasing_accuracies:
        #         if decreasing_accuracie >= self.knock_out_rounds:
        #             kick_out_device_list.add(worker_idx)
        #             break
        #     self.receive_rewards(rewards)
        #     validation_effort_rewards += rewards
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_number': comm_round, 'accuracies_this_round': copy.deepcopy(self.accuracies_this_round), 'validation_effort_rewards': self.validation_rewards_this_round, "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(f"WARNING: {worker_device.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(miners_in_peer_list, 1)[0]
        return self.validator_associated_miner

    def return_validator_acception_wait_time(self):
        return self.validator_acception_wait_time

    ''' miner and validator '''
    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
        else:
            print(f"WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_device_idx):
        if not souce_device_idx in self.black_list:
            self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))
            print(f"{souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(f"Source device {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is boardcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(self, self.unconfirmmed_transactions)
                    else:
                        print(f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Boardcasting skipped.")

    def accept_broadcasted_transactions(self, source_device, broadcasted_transactions):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))
            print(f"{self.role} {self.idx} has accepted transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")

    ''' worker and validator '''

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_device(self, to_associate_device_role):
        to_associate_devices_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == to_associate_device_role:
                if not peer.return_idx() in self.black_list:
                    to_associate_devices_in_peer_list.add(peer)
        if not to_associate_devices_in_peer_list:
            # there is no device matching the required associated role in this device's peer list
            return False
        # associate with a device from list
        to_associate_device = vars(self)[f'{self.role}_associated_{to_associate_device_role}']
        to_associate_device = random.sample(to_associate_devices_in_peer_list, 1)[0]
        # check if the associated device is online
        while not to_associate_device.is_online():
            # remove from peer list and candidate list
            self.remove_peers(to_associate_device)
            to_associate_devices_in_peer_list.discard(to_associate_device)
            if not to_associate_devices_in_peer_list:
                # no devices required to be associated are online
                return False
            # reselect a device
            to_associate_device = random.sample(validators_in_peer_list, 1)[0]
        print(f"{self.role} {self.idx} associated with {to_associate_device.return_role()} {to_associate_device.return_idx()}")
        return to_associate_device

    ''' validator '''
    # def add_sig_verified_transaction(self, transaction_to_validate, souce_device_idx):
    #     if not souce_device_idx in self.black_list:
    #         self.sig_verified_transactions.append(copy.deepcopy(transaction_to_validate))
    #         print(f"worker {souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
    #     else:
    #         print(f"Source worker {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction will not be accepted.")

    # def remove_sig_verified_transaction(self, transaction_to_remove):
    #     self.sig_verified_transactions.remove(transaction_to_remove)

    # def return_sig_verified_transactions(self):
    #     return self.sig_verified_transactions

    def set_unordered_arrival_time_direct_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_direct_worker_transactions = unordered_transaction_arrival_queue

    def return_unordered_arrival_time_direct_worker_transactions(self):
        return self.unordered_arrival_time_direct_worker_transactions

    def broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "validator":
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is boardcasting received transactions to {peer.return_role()} {peer.return_idx()}.")
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the entire received transactions and later calculate the order the individual transaction's arrival time
                        peer.accept_broadcasted_validator_transactions(self, self.unordered_arrival_time_direct_worker_transactions)
                    else:
                        print(f"Destination validator {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Boardcasting skipped for this dest validator.")

    def accept_broadcasted_validator_transactions(self, source_device, unordered_transaction_arrival_queue_from_source_validator):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.accepted_broadcasted_validator_transactions.append({'source_device_link_speed': source_device.return_link_speed(),'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_validator)})
            print(f"{self.role} {self.idx} has accepted validator transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source validator {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.accepted_broadcasted_validator_transactions

    def set_transaction_for_final_processing_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def return_final_transactions_processing_queue(self):
        return self.final_transactions_queue_to_validate
    # def broadcast_sig_verified_transaction(self):
    #     for peer in self.peer_list:
    #         if peer.is_online():
    #             if peer.return_role() == "validator":
    #                 if not peer.return_idx() in self.black_list:
    #                     print(f"{self.role} {self.idx} is boardcasting signature verified transactions to {peer.return_role()} {peer.return_idx()}.")
    #                     # in the real distributed system, it should be broadcasting transaction one by one. Here we send the entire received sig verified transactions
    #                     peer.accept_broadcasted_sig_verified_transactions(self, self.sig_verified_transactions)
    #                 else:
    #                     print(f"Destination validator {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Boardcasting skipped.")

    # def accept_broadcasted_sig_verified_transactions(self, source_device, broadcasted_transactions):
    #     # discard malicious node
    #     if not source_device.return_idx() in self.black_list:
    #         self.broadcasted_sig_verified_transactions.append(copy.deepcopy(broadcasted_transactions))
    #         print(f"{self.role} {self.idx} has accepted sig verified transactions from {source_device.return_role()} {source_device.return_idx()}")
    #     else:
    #         print(f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")

    def validate_worker_transaction(self, transaction_to_verify, rewards):
        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_verify['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["worker_signature"]
            modulus = transaction_to_verify['worker_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['worker_rsa_pub_key']["pub_key"]
            signature = transaction_to_verify["worker_signature"]
            # begin validation
            verification_time = time.time()
            # 1 - verify signature
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                # return True
            else:
                print(f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")
                return False, False
            # 2 - validate worker's local_updates_params
            # current global model accuracy
            current_accuracy = self.evaluate_model_weights()
            # accuracy evaluated by worker's update
            accuracy_by_worker_update_using_own_data = self.evaluate_model_weights(transaction_to_verify["local_updates_params"])
            transaction_to_verify['validation_by'] = self.idx
            transaction_to_verify['validation_rewards'] = rewards
            # if accuracy decreases by worker's updates, False, otherwise True
            print(f'Current validator model accuracy - {current_accuracy}')
            print(f"After applying worker's update, model accuracy becomes - {accuracy_by_worker_update_using_own_data}")
            if current_accuracy > accuracy_by_worker_update_using_own_data:
                transaction_to_verify['update_direction'] = False
                print(f"NOTE: worker {worker_transaction_device_idx}'s' updates is deemed as suspiciously malicious by validator {self.idx}")
            else:
                transaction_to_verify['update_direction'] = True
                print(f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.idx}")
            verification_time = (time.time() - verification_time)/self.computation_power
            transaction_to_verify['verification_time'] = verification_time
            transaction_to_verify['validator_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            transaction_to_verify["validator_signature"] = self.sign_msg(sorted(transaction_to_verify.items()))
            return verification_time, transaction_to_verify
    # def return_validator_size_stop(self):
    #     return self.validator_size_stop

class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, batch_size, loss_func, opti, num_devices, network_stability, net, dev, knock_out_rounds, shard_test_data, validator_acception_wait_time, miner_acception_wait_time, miner_accepted_transactions_size_limit, even_link_speed, even_computation_power, num_malicious):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = {}
        self.knock_out_rounds = knock_out_rounds
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed = even_link_speed
        self.even_computation_power = even_computation_power
        self.num_malicious = num_malicious
        # distribute dataset
        ''' validator '''
        self.validator_acception_wait_time = validator_acception_wait_time
        # self.validator_sig_validated_transactions_size_limit = validator_sig_validated_transactions_size_limit
        # self.validator_size_stop = validator_size_stop
        ''' miner '''
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        ''' shard '''
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the devices
    def data_set_balanced_allocation(self):
        # read dataset
        mnist_dataset = DatasetLoad(self.data_set_name, self.is_iid)
        
        # perpare training data
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label
        # shard dataset and distribute among devices
        # shard train
        shard_size_train = mnist_dataset.train_data_size // self.num_devices // 2
        shards_id_train = np.random.permutation(mnist_dataset.train_data_size // shard_size_train)

        # perpare test data
        if not self.shard_test_data:
            test_data = torch.tensor(mnist_dataset.test_data)
            test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
            test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        else:
            test_data = mnist_dataset.test_data
            test_label = mnist_dataset.test_label
             # shard test
            shard_size_test = mnist_dataset.test_data_size // self.num_devices // 2
            shards_id_test = np.random.permutation(mnist_dataset.test_data_size // shard_size_test)
        
        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(range(self.num_devices), self.num_malicious)

        for i in range(self.num_devices):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), batch_size=100, shuffle=False)
            # assign data to a device and put in the devices set
            if i in malicious_nodes_set:
                is_malicious = True
                # add Gussian Noise

            device_idx = f'device_{i+1}'
            a_device = Device(device_idx, TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label)), test_data_loader, self.batch_size, self.loss_func, self.opti, self.default_network_stability, self.net, self.dev, self.validator_acception_wait_time, self.miner_acception_wait_time, self.miner_accepted_transactions_size_limit, self.even_link_speed, self.even_computation_power, is_malicious, self.knock_out_rounds)
            # device index starts from 1
            self.devices_set[device_idx] = a_device