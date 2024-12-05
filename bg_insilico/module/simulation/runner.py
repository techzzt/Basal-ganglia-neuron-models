from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.synapse_models import create_synapses
from module.utils.data_handler import plot_raster

def run_simulation_with_inh_ext_input(neuron_configs, synapse_params, synapse_class, cortex_inputs, simulation_duration=1000*ms):
    try:
        start_scope()
        neuron_groups = create_neurons(neuron_configs)
        print(f"Neuron Groups: {neuron_groups.keys()}") 

        synapse_connections = create_synapses(neuron_groups, synapse_params, synapse_class)

        net = Network()
        net.add(neuron_groups.values())
        net.add(synapse_connections)
        
        spike_monitors = {}
        voltage_monitors = {}
        for name, group in neuron_groups.items():
            
            spike_mon = SpikeMonitor(group)
            spike_monitors[name] = spike_mon
            net.add(spike_mon)
            
            if 'v' in group.variables:
                voltage_mon = StateMonitor(group, 'v', record=True)
                voltage_monitors[name] = voltage_mon
                net.add(voltage_mon)
        
        for t in range(0, int(simulation_duration/ms), 100):  
            print(f"Remaining time: {simulation_duration/ms - t} ms")
            net.run(100 * ms)    
        
        net.run(1000 * ms)
        plot_raster(spike_monitors)  
        
        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections
        }
        
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise