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
    def __init__(self, idx, assigned_train_ds, assigned_test_ds, local_batch_size, loss_func, opti, network_stability, net, dev, kick_out_rounds):
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
        self.initial_net_parameters = None
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
        self.kick_out_rounds = kick_out_rounds
        self.worker_accuracy_records = {}
        self.has_added_block = False
        self.acc
        ''' For workers '''
        self.local_updates_rewards = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = None
        ''' For miners '''
        self.associated_worker_set = set()
        self.associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validation_block = None
        # self.block_to_add = None
        ''' For validators '''
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        # self.unconfirmmed_validator_transactions = None or []
        # self.broadcasted_validator_transactions = None or []
        

    ''' Common Methods '''
    def return_idx(self):
        return self.idx

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def return_rsa_pub_key(self):
        return {"modulus": modulus, "public_key": public_key}
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        signature = {"pub_key": self.public_key, "modulus": self.modulus, "signature": signature}
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
        
    def return_role(self):
        return self.role

    def online_switcher(self):
        self.old_status = self.on_line
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.on_line = True
        else:
            self.on_line = False
        return self.on_line

    def is_back_online(self):
        back_online = True if self.old_status == False and self.old_status != self.on_line
        return back_online

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
        # remove malicious peers
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                self.remove_peers(peer)
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
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need machenism to reverse updates by # of blocks
            # chain_structure_before_replace = self.return_blockchain_object().return_chain_structure()
            # for block_iter in range(len(chain_structure_before_replace)):
            #     if chain_structure_before_replace[block_iter].compute_hash(hash_whole_block=True) != longest_chain_structure[block_iter].compute_hash(hash_whole_block=True):
            #         break
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.return_idx()} chain resynced")
            #return block_iter
            return True

    def receive_rewards(self, rewards):
        self.rewards += rewards
    
    def return_computation_power(self):
        return self.computation_power

    def verify_block(self, block_to_verify):
        if block_to_verify.return_mined_by() in self.black_list:
            return False
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
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
            return False
        last_block = self.return_blockchain_object().return_last_block()
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
			# block_to_verify.set_pow_proof()
        return block_to_verify

    def add_block(self, block_to_add):
        if block_to_add.is_validator_block():
            last_block_on_validator_chain = self.return_blockchain_object().return_last_block()
            # check last block
            if last_block_on_validator_chain == None or last_block_on_validator_chain.is_validator_block():
                print("Validation block cannot be appended to a validation block.")
                return False
        self.return_blockchain_object().append_block(block_to_add)
        self.has_added_block = True
        return True

    def reset_has_added_block(self):
        self.has_added_block = False

    def return_has_added_block(self):
        return self.has_added_block        

    def operate_on_validator_block(self, passed_in_validator_block=None):
        if not passed_in_block_to_operate:
            block_to_operate = self.return_blockchain_object().return_last_block()
        else:
            # use when chain resynced
            block_to_operate = passed_in_validator_block
        transactions = block_to_operate.return_transactions()
        for transaction in transactions:
            if transaction['validator_device_idx'] in self.black_list:
                continue
            accuracies_this_round = transaction['accuracies_this_round']
            for worker_idx, worker_accuracy in accuracies_this_round.items():
                if not worker_idx in self.worker_accuracy_records.keys():
                    worker_accuracy_records[worker_idx] = []
                worker_accuracy_records[worker_idx].append(accuracy)
        # check and add to black_list
        from more_itertools import split_when
        for worker_idx, worker_accuracy in self.worker_accuracy_records.items():
            decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
            for decreasing_accuracie in decreasing_accuracies:
                if decreasing_accuracie >= self.kick_out_rounds:
                    self.black_list.add(worker_idx)
                    break
            
        kicked_out_devices = transaction['transaction']
        # add malicious devices to the black list
        self.black_list.update(kicked_out_devices)

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
    # TODO change to computation power
    def worker_local_update(self, rewards):
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
                self.local_updates_rewards += 1
                self.receive_rewards(rewards)
        print("Done")
        self.local_train_parameters = self.net.state_dict()

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_local_updates_and_signature(self, comm_round):
        local_updates_dict = {'worker_device_idx': self.return_idx(), 'round_number': comm_round, "local_updates_params": copy.deepcopy(self.local_train_parameters), "local_updates_rewards": self.local_updates_rewards, "rsa_pub_key": self.return_rsa_pub_key()}
        local_updates_dict["signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        return local_updates_dict

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = None
        self.local_updates_rewards = 0
        self.has_added_block = False

    def receive_block_from_miner(self, received_block):
        if not received_block.return_mined_by() in self.black_list:
            self.received_block_from_miner = copy.deepcopy(received_block)

    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner
    
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

    def global_update(self, passed_in_block_to_operate=None):
        if not passed_in_block_to_operate:
            block_to_operate = self.return_blockchain_object().return_last_block()
        else:
            # use when chain resynced
            block_to_operate = passed_in_block_to_operate
        if block_to_operate.is_validator_block():
            print(f"block {block_to_operate.return_block_idx()} is a validator appened block. skip global model updates for this block")
            return
        # avg the gradients
        sum_parameters = None
        # TODO verify transaction??
        transactions = block_to_operate.return_transactions()
        for transaction in transactions:
            if transaction['worker_device_idx'] in self.black_list:
                continue
            local_updates_params = transaction['local_updates_params']
            if sum_parameters is None:
                sum_parameters = copy.deepcopy(local_updates_params)
            else:
                for var in sum_parameters:
                    sum_parameters[var] += local_updates_params[var]
        num_participants = len(transactions)
        for var in self.global_parameters:
            self.global_parameters[var] = (sum_parameters[var] / num_participants)
        print(f"global updates done by block {block_to_operate.return_block_idx()}")
    
    ''' miner '''
    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)

    def add_validator_to_association(self, validator_device):
        if not validator_device.return_idx() in self.black_list:
            self.associated_validator_set.add(validator_device)

    def return_associated_workers(self):
        return self.associated_worker_set
    
    def return_associated_validators(self):
        return self.associated_validator_set
    
    def add_unconfirmmed_transaction(self, unconfirmmed_transaction):
        self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    # def add_unconfirmmed_validator_transactions(self, unconfirmmed_validator_transaction):
    #     self.unconfirmmed_validator_transactions.append(copy.deepcopy(unconfirmmed_validator_transaction))

    # def return_unconfirmmed_validator_transactions(self):
    #     return self.unconfirmmed_validator_transactions

    def accept_broadcasted_transactions(self, source_miner_idx, broadcasted_transactions):
        # discard malicious node
        if not source_miner_idx in self.black_list:
            self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == 'miner':
                    if not peer.return_idx() in self.black_list:
                        peer.accept_broadcasted_transactions(self.idx, self.unconfirmmed_transactions)

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions


    def verify_transaction_by_signature(self, transaction_to_verify):
        try:
            transaction_device_idx = transaction_to_verify['worker_device_idx']
        except:
            transaction_device_idx = transaction_to_verify['validator_device_idx']
        if transaction_device_idx in self.black_list:
            return False
        transaction_before_signed = copy.deepcopy(transaction_to_verify)
        del transaction_before_signed["signature"]
        modulus = transaction_to_verify['rsa_pub_key']["modulus"]
        pub_key = transaction_to_verify['rsa_pub_key']["pub_key"]
        signature = transaction_to_verify["signature"]
        # verify
        hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items()))).encode('utf-8')).digest(), byteorder='big')
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
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(received_propagated_block)

     def receive_propagated_validation_block(self, received_propagated_validation_block):
        if not received_propagated_validation_block.return_mined_by() in self.black_list:
            self.received_propagated_validation_block = copy.deepcopy(received_propagated_validation_block)
    
    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validation_block(self):
        return self.received_propagated_validation_block
        
    def toss_propagated_block(self):
        self.received_propagated_block = None
        
    def toss_propagated_block(self):
        self.received_propagated_validation_block = None

    # def set_block_to_add(self, block_to_add):
    #     self.block_to_add = block_to_add

    # def return_block_to_add(self):
    #     return self.block_to_add
    
    def miner_reset_vars_for_new_round(self):
        self.associated_worker_set.clear()
        self.associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmmed_validator_transactions.clear()
        # self.broadcasted_validator_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validation_block = None
        self.has_added_block = False
#        self.block_to_add = None

    def miner_reset_vars_for_new_validation_round(self):
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()

    ''' Validator '''
    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.has_added_block = False

    def get_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list:

    def accept_accuracy(self, worker, effort_rewards):
        worker_idx = worker.return_idx()
        if not worker_idx in self.black_list:
            worker_accuracy = worker.return_accuracy_this_round()
            # record in block
            self.accuracies_this_round[worker_idx] = worker_accuracy
            self.validation_rewards_this_round += 1
            self.receive_rewards(effort_rewards)
            # record in its own cache
            # if worker_idx in self.worker_accuracy_records.keys():
            #     self.worker_accuracy_records[worker_idx].append(worker_accuracy)
            #     self.receive_rewards(effort_rewards)
            # else:
            #     self.worker_accuracy_records[worker_idx] = [worker_accuracy]
            #     self.receive_rewards(effort_rewards)
    
    # def record_worker_performance_in_block(self, validator_candidate_block, comm_round, rewards):
    #     kick_out_device_set = set()
    #     validation_effort_rewards = 0
    #     from more_itertools import split_when
    #     for worker_idx, worker_accuracy in self.worker_accuracy_records.items():
    #         decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
    #         for decreasing_accuracie in decreasing_accuracies:
    #             if decreasing_accuracie >= self.kick_out_rounds:
    #                 kick_out_device_list.add(worker_idx)
    #                 break
    #         self.receive_rewards(rewards)
    #         validation_effort_rewards += rewards
    #     validator_candidate_block.add_validator_transaction({'validator_idx': self.return_idx(), 'round_number': comm_round, 'accuracies_this_round': self.accuracies_this_round, 'kicked_out_devices': kick_out_device_set, 'validation_effort_rewards': validation_effort_rewards})

    def return_validations_and_signature(self, comm_round):
        #kick_out_device_set = set()
        #validation_effort_rewards = 0
        # from more_itertools import split_when
        # for worker_idx, worker_accuracy in self.worker_accuracy_records.items():
        #     decreasing_accuracies = list(split_when(worker_accuracy, lambda x, y: y > x))
        #     for decreasing_accuracie in decreasing_accuracies:
        #         if decreasing_accuracie >= self.kick_out_rounds:
        #             kick_out_device_list.add(worker_idx)
        #             break
        #     self.receive_rewards(rewards)
        #     validation_effort_rewards += rewards
        validation_transaction_dict = {'validator_device_idx': self.return_idx(), 'round_number': comm_round, 'accuracies_this_round': copy.deepcopy(self.accuracies_this_round), 'validation_effort_rewards': self.validation_rewards_this_round, "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    ''' miner and validator '''
    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    ''' worker and validator '''
    # def associate_with_miner(self):
    #     online_miners_in_peer_list = set()
    #     for peer in self.peer_list:
    #         if peer.is_online():
    #             if peer.return_role() == "miner":
    #                 online_miners_in_peer_list.add(peer)
    #     if not online_miners_in_peer_list:
    #         return False
    #     self.worker_associated_miner = random.sample(online_miners_in_peer_list, 1)[0]
    #     return self.worker_associated_miner

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.worker_associated_miner = random.sample(miners_in_peer_list, 1)[0]
        return self.worker_associated_miner

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, batch_size, loss_func, opti, num_devices, network_stability, net, dev, kick_out_rounds):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = {}
        self.kick_out_rounds = kick_out_rounds
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
            a_device = Device(device_idx, TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label)), TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), self.batch_size, self.loss_func, self.opti, self.default_network_stability, self.net, self.dev, self.kick_out_rounds)
            # device index starts from 1
            self.devices_set[device_idx] = a_device