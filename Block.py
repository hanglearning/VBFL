# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
import copy
import json
from hashlib import sha256

class Block:
    def __init__(self, idx, previous_hash, transactions=None, nonce=0, block_hash=None, mined_by=None, signature=None, mining_rewards=None, is_validator_block=False, validated_by=None, validating_rewards=None, validator_transactions=None):
        self._idx = idx
        self._previous_hash = previous_hash
        self._transactions = transactions or []
        self._nonce = nonce
        # validator specific
        self._is_validator_block = is_validator_block
        self._validated_by = validated_by
        self._validating_rewards = validating_rewards
        # miner and validator
        self._mined_by = mined_by
        self._mining_rewards = mining_rewards
        # the hash of the current block, calculated by compute_hash
        self._block_hash = block_hash
        self._signature = signature
        # self._validator_transactions = validator_transactions or []

    # compute_hash() also used to return value for block verification
    # if False by default, used for pow and verification, in which block_hash has to be None, because at this moment -
    # pow - block hash is None, so does not affect much
    # verification - the block already has its hash
    # if hash_whole_block == True -> used in set_previous_hash, where we need to hash the whole previous block
    def compute_hash(self, hash_whole_block=False):
        block_content = copy.deepcopy(self.__dict__)
        if not hash_whole_block:
            block_content['_block_hash'] = None
            block_content['_signature'] = None
            block_content['_mining_rewards'] = None
        # need sort keys to preserve order of key value pairs
        return sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()

    def set_hash(self, the_hash):
        self._block_hash = the_hash

    def nonce_increment(self):
        self._nonce += 1

    # returnters of the private attribute
    def return_block_hash(self):
        return self._block_hash
    
    def return_previous_hash(self):
        return self._previous_hash

    def return_block_idx(self):
        return self._idx

    
    def return_pow_proof(self):
        return self._block_hash

    ''' Miner Specific '''
    def set_previous_hash(self, hash_to_set):
        self._previous_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        # after verified in cross_verification()
        self._transactions.append(transaction)

    def set_nonce(self, nonce):
        # used if propagated block not verified
        self._nonce = nonce

    def return_current_nonce(self):
        return self._nonce

    def set_mined_by(self, mined_by):
        self._mined_by = mined_by
    
    def return_mined_by(self):
        return self._mined_by

    def add_signature(self, signature):
        # signed by mined_by node
        self._signature = signature

    def return_signature(self):
        return self._signature

    ''' Validator Specific '''
    def is_validator_block(self):
        return self._is_validator_block

    def add_validator_transaction(self, validator_transaction):
        self._transactions.append(validator_transaction)

    ''' Miner and Validator '''
    def set_mining_rewards(self, mining_rewards):
        self._mining_rewards = mining_rewards

    def return_mining_rewards(self):
        self._mining_rewards = mining_rewards
    
    def return_transactions(self):
        # return the updates or validator's transactions from this block
        return self._transactions

    