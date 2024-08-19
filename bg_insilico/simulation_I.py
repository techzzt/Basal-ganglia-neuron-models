import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz, volt, ohm
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor
import importlib
import matplotlib.pyplot as plt
from result_I import Run

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
            if unit == 'nS':
                value *= nS
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


def run_simulation(N, params, model_name, I_values):
    module_path = f'models/{model_name}.py'
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    neuron_model_class = model_module.NeuronModel

    converted_params = convert_units(params)
    
    all_results = []
    all_currents = []
    total_time = []
    injection_times = []

    for I in I_values:
        if 'I' in params:
            I_info = params['I']
            I_unit = I_info['unit']
            
            if I_unit == 'pA':
                converted_I = I * pA
            elif I_unit == 'volt/second':
                converted_I = I * (volt / second) * 1e12
            else:
                print(f"Unknown unit for I: {I_unit}")
                converted_I = 0 * pA
        else:
            converted_I = 0 * pA

        converted_params['I'] = converted_I

        neuron_model = neuron_model_class(N, converted_params)
        sim = Run(neuron_model)

        Initialize_time = 1000 * ms
        neuron_model.neurons.I = 0 * pA

        wait_time_after_stabilization = 300 * ms
        sim.run(duration=wait_time_after_stabilization)

        membrane_potential = sim.dv_monitor.v[0]
        current = sim.current_monitor.I[0]

        neuron_model.neurons.I = converted_I
        time_after_increase = 200 * ms
        sim.run(duration=time_after_increase)
        
        membrane_potential_after_increase = sim.dv_monitor.v[0]
        current_after_increase = sim.current_monitor.I[0]

        neuron_model.neurons.I = 0 * pA
        time_after_decrease = Initialize_time - wait_time_after_stabilization - time_after_increase
        sim.run(duration=time_after_decrease)
        membrane_potential_after_decrease = sim.dv_monitor.v[0]
        current_after_decrease = sim.current_monitor.I[0]

        membrane_potential = np.concatenate((membrane_potential, membrane_potential_after_increase, membrane_potential_after_decrease))
        current = np.concatenate((current, current_after_increase, current_after_decrease))

        all_results.append(membrane_potential)
        all_currents.append(current)
        total_time.append(sim.dv_monitor.t / ms)
        injection_times.append({
            'start': wait_time_after_stabilization / ms,
            'duration': time_after_increase / ms
        })

    return all_results, all_currents, total_time, injection_times

def plot_results(all_results, all_currents, I_values, total_time, injection_times):
    num_plots = len(all_results)
    plt.figure(figsize=(15, 4 * num_plots))

    # Membrane Potential Plot
    for i, (membrane_potential, current, injection_time) in enumerate(zip(all_results, all_currents, injection_times)):
        plt.subplot(num_plots, 1, i + 1)
        
        # Find the minimum length among the time vector, membrane potential, and current
        min_length = min(len(total_time[i]), len(membrane_potential), len(current))
        
        # Use only the first 'min_length' elements for plotting
        time_vector = total_time[i][:min_length]
        membrane_potential = membrane_potential[:min_length]
        current = current[:min_length]
        
        plt.plot(time_vector, membrane_potential / mV, label=f'I = {I_values[i]} pA')
        # plt.plot(time_vector, current / pA, label=f'Current I = {I_values[i]} pA', linestyle='--', color='orange')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV) / Current (pA)')
        # plt.title(f'Membrane Potential & Current for I = {I_values[i]} pA')
        plt.legend()

        start_time = injection_time['start']
        duration = injection_time['duration']
        plt.axvline(x=start_time, color='r', linestyle='--', label='Current Injection Start')
        plt.axvline(x=start_time + duration, color='g', linestyle='--', label='Current Injection End')

    plt.tight_layout()
    plt.show()

    # Plot all currents together in a single plot
    plt.figure(figsize=(15, 6))
    for i, current in enumerate(all_currents):
        min_length = min(len(total_time[i]), len(current))
        time_vector = total_time[i][:min_length]
        current = current[:min_length]
        
        plt.plot(time_vector, current / pA, label=f'I = {I_values[i]} pA', linestyle='--')

    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.title('All Current Patterns Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


