# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/

class Block:
    def __init__(self, idx, transactions=None, previous_hash=None, nonce=0, block_hash=None, mined_by=None, signature=None, mining_rewards=None):
        self._idx = idx
        self._transactions = transactions or []
        self._previous_hash = previous_hash
        self._nonce = nonce
        # the hash of the current block, calculated by compute_hash
        self._block_hash = block_hash
        self._mined_by = mined_by
        self._signature = signature
        self._mining_rewards = mining_rewards

    # compute_hash() also used to return value for block verification  
    def compute_hash(self, hash_previous_block=False):
        if hash_previous_block:
            block_content = self.__dict__
        else:
            block_content = copy.deepcopy(self.__dict__)
            block_content['_block_hash'] = None
            block_content['_signature'] = None
            block_content['_mining_rewards'] = None
        block_content = json.dumps(block_content, sort_keys=True)
        return sha256(block_content.encode()).hexdigest()

    def set_hash(self):
        self._block_hash = self.compute_hash()

    def nonce_increment(self):
        self._nonce += 1

    # getters of the private attribute
    def get_block_hash(self):
        return self._block_hash
    
    def get_previous_hash(self):
        return self._previous_hash

    def get_block_idx(self):
        return self._idx

    def get_transactions(self):
        # get the updates from this block
        return self._transactions
    
    ''' Miner Specific '''
    def set_previous_hash(self, hash_to_set):
        self._previous_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        # after verified in cross_verification()
        self._transactions.append(transaction)

    def set_nonce(self, nonce):
        # used if propagated block not verified
        self._nonce = nonce

    def get_current_nonce(self):
        return self._nonce

    def add_signature(self, mined_by, signature):
        # mined_by is also signed_by
        self._mined_by = mined_by
        self._signature = signature

    def set_mining_rewards(self, mining_rewards):
        self._mining_rewards = mining_rewards