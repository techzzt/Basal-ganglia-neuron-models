import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, pA
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor

class Run:
    def __init__(self, neuron_model):
        self.neuron_model = neuron_model
        self.network = Network(neuron_model.neurons)
        self.monitors_set_up = False

    def setup_monitors(self):
        if not self.monitors_set_up:
            print("Setting up monitors")
            self.dv_monitor = StateMonitor(self.neuron_model.neurons, 'v', record=True)
            self.spike_monitor = SpikeMonitor(self.neuron_model.neurons)
            self.rate_monitor = PopulationRateMonitor(self.neuron_model.neurons)
            self.current_monitor = StateMonitor(self.neuron_model.neurons, 'I', record=True)

            self.network.add(self.dv_monitor, self.spike_monitor, self.rate_monitor, self.current_monitor)
            self.monitors_set_up = True
        else:
            print("Monitors already set up")

    def run(self, duration):
        print(f"Running simulation for {duration} ms")
        self.setup_monitors()
        self.network.run(duration)
        # Reset the monitors for the next phase
        self.monitors_set_up = False

