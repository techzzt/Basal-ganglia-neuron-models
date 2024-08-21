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

def run_simulation(N, params, model_name, I_values, durations):
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
        for duration in durations:
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
            neuron_model.neurons.I = 0 * pA
            wait_time_after_stabilization = 50 * ms
            sim.run(duration=wait_time_after_stabilization)
            membrane_potential = sim.dv_monitor.v[0]
            current = sim.current_monitor.I[0]

            neuron_model.neurons.I = converted_I
            time_after_increase = duration * ms
            sim.run(duration=time_after_increase)
            membrane_potential_after_increase = sim.dv_monitor.v[0]
            current_after_increase = sim.current_monitor.I[0]

            Initialize_time = 1000 * ms
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
                'duration': duration
            })

    return all_results, all_currents, total_time, injection_times


def plot_results(all_results, all_currents, I_values, total_time, injection_times, durations):
    num_I_values = len(I_values)
    num_durations = len(durations)

    # Create a figure with a grid of subplots: num_I_values rows for membrane potential + 1 row per duration for currents
    fig, axes = plt.subplots(num_I_values + 1, num_durations, figsize=(5 * num_durations, 4 * (num_I_values + 1)))
    
    # Plot membrane potentials
    for i, I in enumerate(I_values):
        for j, duration in enumerate(durations):
            # Calculate the index for all_results, all_currents, etc.
            index = i * num_durations + j
            
            # Access the appropriate subplot for membrane potential
            ax_membrane = axes[i, j] if num_I_values > 1 else axes[j]

            membrane_potential = all_results[index]
            injection_time = injection_times[index]
            time_vector = total_time[index]
            
            # Ensure the time_vector starts at 0
            if time_vector[0] != 0:
                time_vector = np.concatenate(([0], time_vector))
                membrane_potential = np.concatenate(([membrane_potential[0]], membrane_potential))
            
            min_length = min(len(time_vector), len(membrane_potential))
            time_vector = time_vector[:min_length]
            membrane_potential = membrane_potential[:min_length]

            # Plot membrane potential
            ax_membrane.plot(time_vector, membrane_potential / mV)
            ax_membrane.set_xlabel('Time (ms)')
            ax_membrane.set_ylabel('Membrane Potential (mV)')
            ax_membrane.set_title(f'I = {I} pA, Duration = {duration} ms')

            start_time = injection_time['start']
            injection_duration = injection_time['duration']
            ax_membrane.axvline(x=start_time, color='r', linestyle='--', label='Injection Start')
            ax_membrane.axvline(x=start_time + injection_duration, color='g', linestyle='--', label='Injection End')

            if j == 0:
                ax_membrane.set_ylabel(f'I = {I} pA\nMembrane Potential (mV)')
            if i == num_I_values - 1:
                ax_membrane.set_xlabel(f'Duration = {duration} ms\nTime (ms)')

    # Plot currents below each column for each duration
    for j, duration in enumerate(durations):
        ax_current = axes[num_I_values, j]
        for i, I in enumerate(I_values):
            index = i * num_durations + j
            current = all_currents[index]
            time_vector = total_time[index]

            min_length = min(len(time_vector), len(current))
            time_vector = time_vector[:min_length]
            current = current[:min_length]

            # Plot the current for this duration across all I_values
            ax_current.plot(time_vector, current / pA, label=f'I = {I} pA', linestyle='--')

        ax_current.set_xlabel('Time (ms)')
        ax_current.set_ylabel('Current (pA)')
        ax_current.set_title(f'Current for Duration = {duration} ms')
        ax_current.legend()

    plt.tight_layout()
    plt.show()






