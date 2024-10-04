import json
from brian2 import *
from brian2 import profiling_summary

from Neuronmodels.GPe_STN import GPeSTNSynapse
from Neuronmodels.GPe import GPe
from Neuronmodels.STN import STN
import matplotlib.pyplot as plt
import importlib
import numpy as np
import plotly.graph_objects as go

def load_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    params = data['params']
    model_name = data['model']
    
    # Extract 'N' from params
    N = params.pop('N')['value'] 
    return N, params, model_name

def convert_units(params):
    converted_params = {}
    for param, info in params.items():
        value = info['value']
        unit = info['unit']
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
        elif unit == 'Ohm':
            value *= ohm
        converted_params[param] = value
    return converted_params

### Simulation part 
def run_simulation(N_1, N_2, params_file_1, params_file_2, synapse_params, model_class_1, model_class_2, synapse_class):
    
    # Load parameters for the two neuron groups (without N)
    _, params_1, model_name_1 = load_params(params_file_1)
    _, params_2, model_name_2 = load_params(params_file_2)

    # Convert units for the neuron models
    params_1_converted = convert_units(params_1)
    params_2_converted = convert_units(params_2)

    # Dynamically load neuron models using the provided class names
    model_module_1 = importlib.import_module(f'Neuronmodels.{model_class_1}')
    model_module_2 = importlib.import_module(f'Neuronmodels.{model_class_2}')
    
    # Initialize the neuron models with the specified N
    neuron_model_1 = getattr(model_module_1, model_class_1)(N=N_1, params=params_1_converted)
    neuron_model_2 = getattr(model_module_2, model_class_2)(N=N_2, params=params_2_converted)

    neurons_1 = neuron_model_1.create_neurons()
    neurons_2 = neuron_model_2.create_neurons()

    synapse_module = importlib.import_module(f'Neuronmodels.{synapse_class}')

    # Set up the synapses between the two neuron groups
    synapse = synapse_module.GPeSTNSynapse(neurons_1, neurons_2, synapse_params)
    syn_generic = synapse.create_synapse()

    # Set up the monitors
    dv_monitor = StateMonitor(neurons_2, 'v', record=True)
    state_monitor = StateMonitor(neurons_2, ['v', 'u'], record=True)
    spike_monitor = SpikeMonitor(neurons_2)

    # Create a network and run the simulation
    net = Network(neurons_1, neurons_2, syn_generic, dv_monitor, spike_monitor)
    net.run(1000*ms)

    # Process the results
    v = state_monitor.v
    u = state_monitor.u
    vr = params_2_converted['vr']
    vr = vr.item() 

    for i in range(len(v)):
        for j in range(len(v[0])):
            if u[i][j] < 0:
                v[i][j] = clip(v[i][j], vr - 15 * mV, 20 * mV)
            else:
                v[i][j] = vr

    # Return results for plotting
    return {
        'times': dv_monitor.t / ms,
        'membrane_potential': dv_monitor.v[0] / mV
    }


def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['times'], results['membrane_potential'])
    plt.title('STN Membrane Potential with GPe Input')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.show()


def run_simulation_with_input(N_GPe, N_STN, gpe_params_file, STN_params_file, synapse_params, model_class_gpe, model_class_STN, synapse_class):
    # Load GPe and STN parameters from the JSON files (without N)
    _, gpe_params, gpe_model_name = load_params(gpe_params_file)
    _, STN_params, STN_model_name = load_params(STN_params_file)

    # Convert units for the neuron models
    gpe_params_converted = convert_units(gpe_params)
    STN_params_converted = convert_units(STN_params)

    # Dynamically load neuron models using the provided class names
    model_module_gpe = importlib.import_module(f'Neuronmodels.{model_class_gpe}')
    model_module_STN = importlib.import_module(f'Neuronmodels.{model_class_STN}')
    
    # Initialize the neuron models
    gpe_model = getattr(model_module_gpe, model_class_gpe)(N=N_GPe, params=gpe_params_converted)
    STN_model = getattr(model_module_STN, model_class_STN)(N=N_STN, params=STN_params_converted)

    GPe = gpe_model.create_neurons()
    STN = STN_model.create_neurons()

    # Set up the synapses between GPe and STN
    synapse = importlib.import_module(f'Neuronmodels.{synapse_class}')
    synapse_instance = synapse.GPeSTNSynapse(GPe, STN, synapse_params)
    syn_GPe_STN = synapse_instance.create_synapse()

    # Set up monitors for both neuron groups
    dv_monitor_gpe = StateMonitor(GPe, 'v', record=True)
    dv_monitor_STN = StateMonitor(STN, ['v', 'u'], record=True)
    spike_monitor_gpe = SpikeMonitor(GPe)
    spike_monitor_STN = SpikeMonitor(STN)

    # Create a network
    net = Network(GPe, STN, syn_GPe_STN, dv_monitor_gpe, dv_monitor_STN, spike_monitor_gpe, spike_monitor_STN)

    # Initial run without input
    GPe.I = 0 * pA
    net.run(200*ms)

    GPe.I = 12 * pA
    net.run(300*ms)

    GPe.I = 0 * pA
    net.run(200*ms)  

    # Process the results
    v = dv_monitor_STN.v
    u = dv_monitor_STN.u
    
    vr = STN_params_converted['vr']
    vr = vr.item()  
    for i in range(len(v)):
        for j in range(len(v[0])):
            if u[i][j] < 0:
                v[i][j] = clip(v[i][j], vr - 15 * mV, 20 * mV)
            else:
                v[i][j] = vr
    
    # Return results for plotting and analysis
    return {
        'gpe_times': dv_monitor_gpe.t / ms,
        'gpe_membrane_potential': dv_monitor_gpe.v[0] / mV,
        'STN_times': dv_monitor_STN.t / ms,
        'STN_membrane_potential': dv_monitor_STN.v[0] / mV,
        'gpe_spikes': spike_monitor_gpe.count,
        'STN_spikes': spike_monitor_STN.count,
        'spike_monitor_gpe': spike_monitor_gpe,  # Include spike monitors here
        'spike_monitor_STN': spike_monitor_STN,  # Include spike monitors here
        'synapse': syn_GPe_STN  
    }

def run_simulation_with_inh_ext_input(
    N_GPe, N_STN, N_Striatum, gpe_params_file, STN_params_file, striatum_params_file, synapse_params, 
    model_class_gpe, model_class_STN, model_class_striatum, synapse_class, input_condition='slow_wave'):

    _, gpe_params, gpe_model_name = load_params(gpe_params_file)
    _, STN_params, STN_model_name = load_params(STN_params_file)
    _, striatum_params, striatum_model_name = load_params(striatum_params_file)

    gpe_params_converted = convert_units(gpe_params)
    STN_params_converted = convert_units(STN_params)
    striatum_params_converted = convert_units(striatum_params)

    model_module_gpe = importlib.import_module(f'Neuronmodels.{model_class_gpe}')
    model_module_STN = importlib.import_module(f'Neuronmodels.{model_class_STN}')
    model_module_striatum = importlib.import_module(f'Neuronmodels.{model_class_striatum}')

    # Initialize the neuron models
    gpe_model = getattr(model_module_gpe, model_class_gpe)(N=N_GPe, params=gpe_params_converted)
    STN_model = getattr(model_module_STN, model_class_STN)(N=N_STN, params=STN_params_converted)
    striatum_model = getattr(model_module_striatum, model_class_striatum)(N=N_Striatum, params=striatum_params_converted)

    # Create neurons for GPe, STN, and Striatum
    GPe = gpe_model.create_neurons()
    STN = STN_model.create_neurons()
    Striatum = striatum_model.create_neurons()

    # Create Cortex neuron group
    N_Cortex = N_STN  
    Cortex = PoissonGroup(N_Cortex, rates= 10 * Hz)  # Cortex input as Poisson spikes
    
    # Define input rate pattern (0 Hz -> 200 Hz -> 0 Hz)
    @network_operation(dt=1*ms)
    def input_rate(t):
        if 200*ms <= t <= 500*ms:
            return 200  # High activity between 200 and 500 ms
        else:
            return 0    # No activity otherwise

    # Set up synapses (inhibitory and excitatory)
    synapse = importlib.import_module(f'Neuronmodels.{synapse_class}')
    synapse_instance = synapse.GPeSTNSynapse(GPe, STN, Striatum, Cortex, synapse_params)
    
    # Synapses from Cortex to Striatum and STN, as well as from GPe to STN
    syn_GPe_STN, syn_Cortex_Striatum, syn_Cortex_STN = synapse_instance.create_synapse()

    # Set up monitors for GPe and STN neuron groups
    dv_monitor_gpe = StateMonitor(GPe, 'v', record=True)
    dv_monitor_STN = StateMonitor(STN, ['v', 'u'], record=True)
    dv_monitor_striatum = StateMonitor(Striatum, 'v', record=True)  # Monitor Striatum membrane potential
    spike_monitor_gpe = SpikeMonitor(GPe)
    spike_monitor_STN = SpikeMonitor(STN)
    spike_monitor_cortex = SpikeMonitor(Cortex)  # Monitor Cortex spikes

    # Create a network and add components
    net = Network(GPe, STN, Striatum, Cortex, syn_GPe_STN, 
                  syn_Cortex_Striatum, syn_Cortex_STN, dv_monitor_gpe, 
                  dv_monitor_STN, dv_monitor_striatum, spike_monitor_gpe, 
                  spike_monitor_STN, spike_monitor_cortex, input_rate)

    net.run(700*ms)

    # Post-process STN membrane potential based on u and reset conditions
    v = dv_monitor_STN.v
    u = dv_monitor_STN.u
    vr = STN_params_converted['vr'].item()
    for i in range(len(v)):
        for j in range(len(v[0])):
            if u[i][j] < 0:
                v[i][j] = clip(v[i][j], vr - 15 * mV, 20 * mV)
            else:
                v[i][j] = vr

    # Retrieve synapse weights
    weights = np.array(syn_GPe_STN.w)

    # Return results for plotting and analysis
    return {
        'gpe_times': dv_monitor_gpe.t / ms,
        'gpe_membrane_potential': dv_monitor_gpe.v[0] / mV,
        'STN_times': dv_monitor_STN.t / ms,
        'STN_membrane_potential': dv_monitor_STN.v[0] / mV,
        'striatum_times': dv_monitor_striatum.t / ms,
        'striatum_membrane_potential': dv_monitor_striatum.v[0] / mV,
        'gpe_spikes': spike_monitor_gpe.count,
        'STN_spikes': spike_monitor_STN.count,
        # 'striatum_spikes': dv_monitor_striatum.count,  # Add Striatum spike counts
        'cortex_spikes': spike_monitor_cortex.count,
        'spike_monitor_gpe': spike_monitor_gpe,  
        'spike_monitor_STN': spike_monitor_STN,
        'spike_monitor_striatum': dv_monitor_striatum,
        'spike_monitor_cortex': spike_monitor_cortex,
        'synapse_GPe_STN': syn_GPe_STN,
        'weights': weights    
    }


### Visualization part 
### Visualization post spike pattern with input 
def plot_results_pre_post(results):
    plt.figure(figsize=(10, 6))

    # Plot GPe membrane potential
    plt.subplot(2, 1, 1)
    plt.plot(results['gpe_times'], results['gpe_membrane_potential'])
    plt.title('GPe Membrane Potential with Input')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')

    # Plot STN membrane potential
    plt.subplot(2, 1, 2)
    plt.plot(results['STN_times'], results['STN_membrane_potential'])
    plt.title('STN Membrane Potential (Spontaneous Response to GPe)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')

    plt.tight_layout()
    plt.show()

### Visualization neuron connection with point  
def plot_connectivity(synapse, N_pre, N_post, title='Synaptic Connectivity'):
    '''
    Plots the connectivity matrix of the synapses.
    
    Parameters:
    synapse (Brian2 Synapses object): The synapse object for which to plot the connectivity.
    N_pre (int): Number of presynaptic neurons.
    N_post (int): Number of postsynaptic neurons.
    title (str): Title for the plot.
    '''
    plt.figure(figsize=(8, 8))
    
    # Get the pre and post-synaptic indices from the Synapses object
    pre_indices = synapse.i
    post_indices = synapse.j
    
    # Create a scatter plot of the connectivity
    plt.scatter(pre_indices, post_indices, s=10, c='blue', alpha=0.5)
    
    plt.xlim([0, N_pre])
    plt.ylim([0, N_post])
    plt.xlabel('Presynaptic neuron index (GPe)')
    plt.ylabel('Postsynaptic neuron index (STN)')
    plt.title(title)
    plt.grid(True)
    
    plt.show()

### Visualization weight matrix within synapse connection 
def plot_results_with_weight_matrix(results, N_GPe, N_STN):
    plt.figure(figsize=(10, 8))
    
    # Extract weights from the results
    synapses = results['synapse'] 
    weights = synapses.w 
    connected_pre = synapses.i  
    connected_post = synapses.j  

    # Create an empty weight matrix
    weight_matrix = np.zeros((N_GPe, N_STN))

    # Populate the weight matrix with the corresponding weights
    for pre, post, weight in zip(connected_pre, connected_post, weights):
        weight_matrix[pre, post] = weight  #

    # Plot synapse weights
    plt.imshow(weight_matrix, aspect='auto', cmap='viridis')
    plt.title('Synapse Weights (GPe to STN)')
    plt.xlabel('Post-synaptic Neurons (STN)')
    plt.ylabel('Pre-synaptic Neurons (GPe)')
    plt.colorbar(label='Weight (nS)')

    plt.tight_layout()
    plt.show()

### Visualization pre, post spike pattern 
def plot_results_with_spikes(results, spike_monitor_gpe, spike_monitor_STN):
    plt.figure(figsize=(10, 8))

    # Plot GPe membrane potential with spikes
    plt.subplot(2, 1, 1)
    plt.plot(results['gpe_times'], results['gpe_membrane_potential'], label='Membrane potential')

    # Plot GPe spikes (spike times for each neuron)
    for neuron_id in np.unique(spike_monitor_gpe.i):
        spike_times = spike_monitor_gpe.t[spike_monitor_gpe.i == neuron_id] / ms  # Get spike times for this neuron
        spike_values = np.interp(spike_times, results['gpe_times'], results['gpe_membrane_potential'])  # Interpolate membrane potential at spike times
        plt.scatter(spike_times, spike_values, color='red', label='Spikes' if neuron_id == 0 else "", zorder=3)
    
    plt.title('GPe Membrane Potential with Input')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.legend()

    # Plot STN membrane potential with spikes
    plt.subplot(2, 1, 2)
    plt.plot(results['STN_times'], results['STN_membrane_potential'], label='Membrane potential')

    # Plot STN spikes (spike times for each neuron)
    for neuron_id in np.unique(spike_monitor_STN.i):
        spike_times = spike_monitor_STN.t[spike_monitor_STN.i == neuron_id] / ms  # Get spike times for this neuron
        spike_values = np.interp(spike_times, results['STN_times'], results['STN_membrane_potential'])  # Interpolate membrane potential at spike times
        plt.scatter(spike_times, spike_values, color='red', label='Spikes' if neuron_id == 0 else "", zorder=3)
    
    plt.title('STN Membrane Potential (Spontaneous Response to GPe)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.legend()

    plt.tight_layout()
    plt.show()

### Visualization with statemonitor result 
def plot_raster(results):
    plt.figure(figsize=(12, 6))

    # GPe Neuron
    plt.subplot(2, 1, 1)
    plt.scatter(results['spike_monitor_gpe'].t/ms, results['spike_monitor_gpe'].i, s=2, color='blue')
    plt.title('GPe Population Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 700)

    # STN Neuron
    plt.subplot(2, 1, 2)
    plt.scatter(results['spike_monitor_STN'].t/ms, results['spike_monitor_STN'].i, s=2, color='green')
    plt.title('STN Population Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 700)

    plt.tight_layout()
    plt.show()

def plot_neuron_connections_3D(neuron_positions, connections):
    # Create figure
    fig = go.Figure()

    # Add neuron nodes
    for idx, pos in enumerate(neuron_positions):
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name=f'Neuron {idx}'
        ))

    # Add synapse connections as lines
    for connection in connections:
        source, target = connection
        x_values = [neuron_positions[source][0], neuron_positions[target][0]]
        y_values = [neuron_positions[source][1], neuron_positions[target][1]]
        z_values = [neuron_positions[source][2], neuron_positions[target][2]]
        fig.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            mode='lines',
            line=dict(width=2, color='green'),
            name=f'Connection {source} -> {target}'
        ))

    # Set axis titles and layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title="3D Neuron Connections"
    )
    
    # Show the plot
    fig.show()
