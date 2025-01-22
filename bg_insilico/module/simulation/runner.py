from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.data_handler import plot_raster
from brian2 import profiling_summary
import json 

def run_simulation_with_inh_ext_input(neuron_configs, connections, synapse_class, simulation_params, plot_order=None):
    
    def get_neuron_params(configs, name):
        for config in configs:
            if config['name'] == name:
                # params_file에서 params 값을 읽어오기
                with open(config['params_file'], 'r') as f:
                    params = json.load(f)
                return params

        return None
    
    try:
        start_scope()
        
        neuron_groups = create_neurons(neuron_configs)
        print(f"Neuron Groups: {neuron_groups.keys()}") 
        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)

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
                print("voltage_mon:", voltage_mon.v[:]) 

        
        defaultclock.dt = simulation_params['dt'] * ms
        duration = simulation_params['duration'] * ms

        for t in range(0, int(duration/ms), 100):  
            print(f"Remaining time: {duration/ms - t} ms")
            net.run(100 * ms, profile=True)    

            for name, group in neuron_groups.items():
                if name == "STN":  

                    params = get_neuron_params(neuron_configs, name)
                    if params:
                        vr = params['params']['vr']['value'] * eval(params['params']['vr']['unit'])
                        u = group.u
                        v = group.v
                
                        condition = u < 0 * nA  
                        adjusted_u = (u - 15 * pA) / (1 * pA) * mV  

                        if len(v[condition]) > 0: 
                            v[condition] = vr + maximum(adjusted_u, 20 * mV) 
                        if len(v[~condition]) > 0: 
                            v[~condition] = vr

        if plot_order:
            plot_raster(spike_monitors, plot_order)
        else:
            plot_raster(spike_monitors)
        
        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections
        }
        print("summary:", profiling_summary(net))
        return results
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise