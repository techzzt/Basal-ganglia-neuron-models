import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz, volt, ohm
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor, start_scope
import importlib
import matplotlib.pyplot as plt 
from result import Visualization

def load_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params['params'], params['model']

def convert_units(params):
    converted_params = {}
    for param, info in params.items():
        value = info['value']
        unit = info['unit']
        if unit:
            # Convert units according to the specifications
            if unit == 'nS':
                value *= nS
            elif unit == 'S':
                value *= siemens
            elif unit == 'mV':
                value *= mV
            elif unit == 'ms':
                value *= ms
            elif unit == 'pF':
                value *= pF
            elif unit == 'pA':
                value *= pA
            elif unit == 'Hz':
                value *= Hz
            elif unit == '1/second':
                value *= Hz
            elif unit == 'volt/second':
                value *= volt / second
            elif unit == 'Ohm':
                value *= ohm
            else:
                print(f"Unknown unit for {param}: {unit}")

        converted_params[param] = value
    return converted_params

def run_simulation(N, params, model_name):
    # start_scope()

    results = []

    # Load neuron model dynamically
    module_path = f'models/{model_name}.py'
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    neuron_model_class = model_module.NeuronModel

    # Convert parameters
    converted_params = convert_units(params)
    print("Converted parameters:", converted_params)

    # Handle input current
    I_value = params['I']['value']
    I_unit = params['I']['unit']
    
    if I_unit == 'pA':
        initial_I = 0 * pA  
        increase_I = I_value * pA  
    elif I_unit == 'volt/second':
        initial_I = 0 * volt/second  
        increase_I = I_value * (volt / second) * 1e12  
    else:
        print(f"Unknown unit for I: {I_unit}")
        initial_I = 0 * pA  
        increase_I = initial_I  

    converted_params['I'] = initial_I  

    # Initialize neuron model
    neuron_model = neuron_model_class(N, converted_params)
    network = Network(neuron_model.neurons)
    
# Set up monitors before initial run
    dv_monitor = StateMonitor(neuron_model.neurons, 'v', record=True)
    spike_monitor = SpikeMonitor(neuron_model.neurons)
    rate_monitor = PopulationRateMonitor(neuron_model.neurons)
    current_monitor = StateMonitor(neuron_model.neurons, 'I', record=True)

    network.add(dv_monitor, spike_monitor, rate_monitor, current_monitor)
 
    neuron_model.neurons.v[0] = -80 * mV

    neuron_model.neurons.I = 0 * pA  
    Initialize_time = 200 * ms
    network.run(duration = Initialize_time)

    # Collect results
    times = dv_monitor.t
    v_reset = converted_params['vr']
    membrane_potential = dv_monitor.v[0]
    matching_indices = np.where(membrane_potential / mV >= v_reset / mV)[0]

    if len(matching_indices) > 0:
        earliest_time_stabilized = times[matching_indices[0]] * 1000
    else:
        earliest_time_stabilized = None

    # Run simulation with input current
    if earliest_time_stabilized is not None:
        wait_time_after_stabilization = 100 * ms
        time_after_increase = 200 * ms
        time_after_decrease = 200 * ms
        total_simulation_time = Initialize_time + wait_time_after_stabilization + time_after_increase + time_after_decrease
        
        neuron_model.neurons.I = 0 * pA  
        network.run(duration=wait_time_after_stabilization)
        neuron_model.neurons.I = increase_I  
        network.run(duration=time_after_increase)
        neuron_model.neurons.I = 0 * pA
        network.run(duration=time_after_decrease)

    else:
        print("v did not reach v_reset, stopping simulation")
        total_simulation_time = Initialize_time

    results = {
        'times': dv_monitor.t / ms,
        'membrane_potential': dv_monitor.v[0] / mV,
        'current': current_monitor.I[0] / pA
    }

    return results


def plot_results(results):
    plt.figure(figsize = (15, 8))
    times = results['times']
    membrane_potential = results['membrane_potential']
    current = results['current']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(times, membrane_potential)
    axes[0].set_title('Membrane Potential')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_ylim([-90, 50])
    axes[0].set_xlim(left=0)

    axes[1].plot(times, current)
    axes[1].set_title('Current')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (pA)')
    axes[1].set_xlim(left=0)

    plt.tight_layout()
    plt.show()
