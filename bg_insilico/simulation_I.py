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

        neuron_model = neuron_model_class(N, converted_params)
        sim = Run(neuron_model)

        total_simtime = 1000 * ms

        # Phase 1: Initial stabilization with 0 pA current
        neuron_model.neurons.I = 0 * pA
        stabilization_time = 200 * ms
        sim.run(duration=stabilization_time)
        membrane_potential_phase1 = sim.dv_monitor.v[0]
        current_phase1 = sim.current_monitor.I[0]

        # Phase 2: Inject specific current I
        neuron_model.neurons.I = converted_I
        injection_time = 300 * ms
        sim.run(duration=injection_time)
        membrane_potential_phase2 = sim.dv_monitor.v[0]
        current_phase2 = sim.current_monitor.I[0]

        # Phase 3: Return to 0 pA current
        neuron_model.neurons.I = 0 * pA
        return_to_baseline_time = total_simtime - stabilization_time - injection_time
        sim.run(duration=return_to_baseline_time)
        membrane_potential_phase3 = sim.dv_monitor.v[0]
        current_phase3 = sim.current_monitor.I[0]
        
        membrane_potential = np.concatenate((membrane_potential_phase1, membrane_potential_phase2, membrane_potential_phase3))
        current = np.concatenate((current_phase1, current_phase2, current_phase3))

        all_results.append(membrane_potential)
        all_currents.append(current)
        total_time.append(sim.dv_monitor.t / ms)
        injection_times.append({
            'start': stabilization_time / ms,
            'duration': injection_time / ms
        })

    return all_results, all_currents, total_time, injection_times

def plot_results(all_results, all_currents, I_values, total_time, injection_times):
    num_plots = len(all_results)
    plt.figure(figsize=(15, 4 * num_plots))

    # Membrane Potential Plot
    for i, (membrane_potential, current, injection_time) in enumerate(zip(all_results, all_currents, injection_times)):
        plt.subplot(num_plots, 1, i + 1)

        # Construct a time vector manually to ensure it starts from 0 ms and aligns with the concatenated data
        total_duration = len(membrane_potential)  # Assuming sampling is consistent
        dt = total_time[i][1] - total_time[i][0]  # Time step size
        time_vector = np.arange(0, total_duration * dt, dt)

        # Use only the first 'min_length' elements for plotting (if there's any discrepancy)
        min_length = min(len(time_vector), len(membrane_potential), len(current))
        time_vector = time_vector[:min_length]
        membrane_potential = membrane_potential[:min_length]
        current = current[:min_length]

        plt.plot(time_vector, membrane_potential / mV, label=f'I = {I_values[i]} pA')
        # plt.plot(time_vector, current / pA, label=f'Current I = {I_values[i]} pA', linestyle='--', color='orange')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()

        # Mark the start and end of the current injection
        start_time = injection_time['start']
        duration = injection_time['duration']
        plt.axvline(x=start_time, color='r', linestyle='--', label=f'Current Injection Start (I = {I_values[i]} pA)')
        plt.axvline(x=start_time + duration, color='g', linestyle='--', label='Current Injection End')

    plt.tight_layout()
    plt.show()

    # Plot all currents together in a single plot
    plt.figure(figsize=(15, 6))
    for i, current in enumerate(all_currents):
        total_duration = len(current)  # Assuming sampling is consistent
        dt = total_time[i][1] - total_time[i][0]  # Time step size
        time_vector = np.arange(0, total_duration * dt, dt)

        min_length = min(len(time_vector), len(current))
        time_vector = time_vector[:min_length]
        current = current[:min_length]
        
        plt.plot(time_vector, current / pA, label=f'I = {I_values[i]} pA', linestyle='--')

    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.title('All Current Patterns Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()



