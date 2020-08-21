from block import Block

class Blockchain:

    # for PoW
    pow_difficulty = 2

    def __init__(self):
        self.chain = []

    def get_chain_length(self):
        return len(self.chain)

    def get_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None
    
    def replace_chain(self, chain):
        self.chain = chain

    def append_block(self, block):
        self._chain.append(block)