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
    
    # Set up monitors before initial run
    dv_monitor = StateMonitor(neuron_model.neurons, 'v', record=True)
    spike_monitor = SpikeMonitor(neuron_model.neurons)
    current_monitor = StateMonitor(neuron_model.neurons, 'I', record=True)
    network = Network(neuron_model.neurons, dv_monitor, spike_monitor, current_monitor)
 
    neuron_model.neurons.v[0] = -80 * mV

    neuron_model.neurons.I = 0 * pA  
    Initialize_time = 200 * ms
    network.run(duration = Initialize_time)

    neuron_model.neurons.I = increase_I 
    time_after_increase = 300 * ms 
    network.run(duration=time_after_increase)
    
    neuron_model.neurons.I = 0 * pA
    remaining_time = 1000 * ms - (Initialize_time + time_after_increase)
    network.run(duration = remaining_time)
    
    # Collect results
    results = {
        'times': dv_monitor.t / ms,
        'membrane_potential': dv_monitor.v[0] / mV,
        'current': current_monitor.I[0] / pA
    }

    return results

def run_simulation_noise(N, params, model_name):
    results = []

    # Load neuron model dynamically
    module_path = f'models/{model_name}.py'
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    neuron_model_class = model_module.NeuronModel

    # Convert parameters
    converted_params = convert_units(params)

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
    
    # Set up monitors before initial run
    dv_monitor = StateMonitor(neuron_model.neurons, 'v', record=True)
    spike_monitor = SpikeMonitor(neuron_model.neurons)
    current_monitor = StateMonitor(neuron_model.neurons, 'I', record=True)
    
    # Create a noise monitor
    noise_monitor = StateMonitor(neuron_model.neurons, 'xi', record=True)
    
    network = Network(neuron_model.neurons, dv_monitor, spike_monitor, current_monitor, noise_monitor)
 
    neuron_model.neurons.v[0] = -80 * mV

    neuron_model.neurons.I = 0 * pA  
    Initialize_time = 200 * ms
    network.run(duration=Initialize_time)

    neuron_model.neurons.I = increase_I 
    time_after_increase = 300 * ms 
    network.run(duration=time_after_increase)
    
    neuron_model.neurons.I = 0 * pA
    remaining_time = 1000 * ms - (Initialize_time + time_after_increase)
    network.run(duration=remaining_time)
    
    # Collect results
    results = {
        'times': dv_monitor.t / ms,
        'membrane_potential': dv_monitor.v[0] / mV,
        'current': current_monitor.I[0] / pA
    }
    
    # Extract noise data
    noise_data = noise_monitor.xi[0] / (volt / second)  # Adjust unit as necessary

    return results, noise_data


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
    axes[0].set_ylim([-120, 0])
    axes[0].set_xlim(left=0)

    axes[1].plot(times, current)
    axes[1].set_title('Current')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (pA)')
    axes[1].set_xlim(left=0)

    plt.tight_layout()
    plt.show()

def plot_results_noise(results, noise_data):
    plt.figure(figsize=(15, 12))
    times = results['times']
    membrane_potential = results['membrane_potential']
    current = results['current']

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot membrane potential
    axes[0].plot(times, membrane_potential)
    axes[0].set_title('Membrane Potential')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_ylim([-90, 50])
    axes[0].set_xlim(left=0)

    # Plot current
    axes[1].plot(times, current)
    axes[1].set_title('Current')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (pA)')
    axes[1].set_xlim(left=0)

    # Plot noise
    axes[2].plot(times, noise_data)
    axes[2].set_title('Noise')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Noise')
    axes[2].set_xlim(left=0)

    plt.tight_layout()
    plt.show()


def plot_results_noise_FI(results, noise_data):
    times = results['times']
    membrane_potential = results['membrane_potential']
    current = results['current']
    
    plt.figure(figsize=(20, 10))
    
    # Plot the membrane potential and current
    plt.subplot(2, 2, 1)
    plt.plot(times, membrane_potential)
    plt.title('Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.ylim([-90, 50])
    plt.xlim(left=0)
    
    plt.subplot(2, 2, 2)
    plt.plot(times, current)
    plt.title('Current')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.xlim(left=0)
    
    # Plot noise data
    plt.subplot(2, 2, 3)
    plt.plot(times, noise_data)
    plt.title('Noise Data')
    plt.xlabel('Time (ms)')
    plt.ylabel('Noise (volt)')
    plt.xlim(left=0)

    # F-I curve and spike time irregularity plots
    plt.subplot(2, 2, 4)
    I_mean = np.linspace(0, 100, num=5)  # Example range, adjust as needed
    I_std = np.linspace(0, 50, num=5)    # Example range, adjust as needed
    
    # Replace these with actual spike count and CV-ISI data
    spk_count = np.random.rand(len(I_mean), len(I_std)) * 50
    cv_isi = np.random.rand(len(I_mean), len(I_std))

    plt.subplot(2, 2, 4)

    # Plot the F-I curve
    plt.figure(figsize=(20, 10))  # New figure for F-I curve to avoid overlapping
    plt.subplot(2, 2, 1)
    for ii in range(len(I_std)):
        plt.plot(I_mean, spk_count[:, ii], label='std = ' + str(I_std[ii]))
    plt.ylabel('Spike count')
    plt.xlabel('Mean of injected current')
    plt.legend(loc='upper right')  # Adjust legend position if needed

    # Plot firing rate vs coefficient of variation of ISI
    plt.subplot(2, 2, 2)
    plt.plot(spk_count.flatten(), cv_isi.flatten(), '.')
    plt.xlabel('Spike count')
    plt.ylabel('Spike time irregularity (CV-ISI)')
    
    # Plot Firing rate as a function of both mean and std
    plt.subplot(2, 2, 3)
    plt.pcolor(I_mean, I_std, spk_count.T)
    plt.ylabel('std. of injected current')
    plt.xlabel('mean of injected current')
    plt.colorbar()
    plt.clim(0, 50)
    plt.title('Spike count')

    # Plot Spike time irregularity (CV-ISI) as a function of both mean and std
    plt.subplot(2, 2, 4)
    plt.pcolor(I_mean, I_std, cv_isi.T)
    plt.ylabel('std. of injected current')
    plt.xlabel('mean of injected current')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title('Spike time irregularity (CV-ISI)')

    plt.tight_layout()
    plt.show()