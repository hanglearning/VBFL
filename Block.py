# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
import copy
import json
from hashlib import sha256

class Block:
	def __init__(self, idx, previous_block_hash=None, transactions=None, nonce=0, miner_rsa_pub_key=None, mined_by=None, mining_rewards=None, pow_proof=None, signature=None):
		self._idx = idx
		self._previous_block_hash = previous_block_hash
		self._transactions = transactions
		self._nonce = nonce
		# miner specific
		self._miner_rsa_pub_key = miner_rsa_pub_key
		self._mined_by = mined_by
		self._mining_rewards = mining_rewards
		# validator specific
		# self._is_validator_block = is_validator_block
		# the hash of the current block, calculated by compute_hash
		self._pow_proof = pow_proof
		self._signature = signature

	# compute_hash() also used to return value for block verification
	# if False by default, used for pow and verification, in which pow_proof has to be None, because at this moment -
	# pow - block hash is None, so does not affect much
	# verification - the block already has its hash
	# if hash_entire_block == True -> used in set_previous_block_hash, where we need to hash the whole previous block
	def compute_hash(self, hash_entire_block=False):
		block_content = copy.deepcopy(self.__dict__)
		if not hash_entire_block:
			block_content['_pow_proof'] = None
			block_content['_signature'] = None
			block_content['_mining_rewards'] = None
		# need sort keys to preserve order of key value pairs
		return sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()

	def remove_signature_for_verification(self):
		self._signature = None

	def set_pow_proof(self, the_hash):
		self._pow_proof = the_hash

	def nonce_increment(self):
		self._nonce += 1

	# returners of the private attributes
	
	def return_previous_block_hash(self):
		return self._previous_block_hash

	def return_block_idx(self):
		return self._idx
	
	def return_pow_proof(self):
		return self._pow_proof
	
	def return_miner_rsa_pub_key(self):
		return self._miner_rsa_pub_key

	''' Miner Specific '''
	def set_previous_block_hash(self, hash_to_set):
		self._previous_block_hash = hash_to_set

	def add_verified_transaction(self, transaction):
		# after verified in cross_verification()
		# transactions can be both worker' or validators' transactions
		self._transactions.append(transaction)

	def set_nonce(self, nonce):
		self._nonce = nonce

	def set_mined_by(self, mined_by):
		self._mined_by = mined_by
	
	def return_mined_by(self):
		return self._mined_by

	def set_signature(self, signature):
		# signed by mined_by node
		self._signature = signature

	def return_signature(self):
		return self._signature

	def set_mining_rewards(self, mining_rewards):
		self._mining_rewards = mining_rewards

	def return_mining_rewards(self):
		return self._mining_rewards
	
	def return_transactions(self):
		return self._transactions

	# ''' Validator Specific '''
	# def is_validator_block(self):
	#	 return self._is_validator_block

	# def add_validator_transaction(self, validator_transaction):
	#	 self._transactions.append(validator_transaction)


	