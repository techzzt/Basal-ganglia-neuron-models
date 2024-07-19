from brian2 import *

class SynapseModel:
    def __init__(self, source, target, model, pre, post=None, connect_rule=None, p_connect=0.1):
        self.source = source
        self.target = target
        self.model = model
        self.pre = pre
        self.post = post
        self.connect_rule = connect_rule if connect_rule else 'i != j'
        self.p_connect = p_connect
        self.build_synapse()

    def build_synapse(self):
        self.synapses = Synapses(self.source, self.target, model=self.model, on_pre=self.pre, on_post=self.post)
        self.synapses.connect(self.connect_rule, p=self.p_connect)

def connect(source, target, model, p_connect=0.1, pre=''):
    synapse_model = SynapseModel(source, target, model, pre, p_connect=p_connect)
    return synapse_model.synapses
