import json
from brian2 import *
from brian2 import profiling_summary

from Neuronmodels import STN, GPeTA, GPeT1, FSN, MSND1, MSND2, SNr, Synapse

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
def run_simulation_with_inh_ext_input(
    N_FSN, N_GPe, N_STN, N_MSN, N_SNr, fsn_params_file, gpet1_params_file, gpeta_params_file, STN_params_file, msnd1_params_file, msnd2_params_file, snr_params_file, synapse_params, 
    model_class_fsn, model_class_gpet1, model_class_gpeta, model_class_STN, model_class_msnd1, model_class_msnd2, model_class_snr, synapse_class, input_condition='slow_wave'):

    _, fsn_params, fsn_model_name = load_params(fsn_params_file)
    _, gpet1_params, gpet1_model_name = load_params(gpet1_params_file)
    _, gpeta_params, gpeta_model_name = load_params(gpeta_params_file)
    _, STN_params, STN_model_name = load_params(STN_params_file)
    _, msnd1_params, msnd1_model_name = load_params(msnd1_params_file)
    _, msnd2_params, msnd2_model_name = load_params(msnd2_params_file)
    _, snr_params, snr_model_name = load_params(snr_params_file)

    fsn_params_converted = convert_units(fsn_params)
    gpet1_params_converted = convert_units(gpet1_params)
    gpeta_params_converted = convert_units(gpeta_params)
    STN_params_converted = convert_units(STN_params)
    msnd1_params_converted = convert_units(msnd1_params)
    msnd2_params_converted = convert_units(msnd2_params)
    snr_params_converted = convert_units(snr_params)

    model_module_fsn = importlib.import_module(f'Neuronmodels.{model_class_fsn}')
    model_module_gpet1 = importlib.import_module(f'Neuronmodels.{model_class_gpet1}')
    model_module_gpeta = importlib.import_module(f'Neuronmodels.{model_class_gpeta}')
    model_module_STN = importlib.import_module(f'Neuronmodels.{model_class_STN}')
    model_module_msnd1 = importlib.import_module(f'Neuronmodels.{model_class_msnd1}')
    model_module_msnd2 = importlib.import_module(f'Neuronmodels.{model_class_msnd2}')
    model_module_snr = importlib.import_module(f'Neuronmodels.{model_class_snr}')

    # Initialize the neuron models
    fsn_model = getattr(model_module_fsn, model_class_fsn)(N=N_FSN, params=fsn_params_converted, neuron_type="E")
    gpet1_model = getattr(model_module_gpet1, model_class_gpet1)(N=N_GPe, params=gpet1_params_converted, neuron_type="E")
    gpeta_model = getattr(model_module_gpeta, model_class_gpeta)(N=N_GPe, params=gpeta_params_converted, neuron_type="E")
    STN_model = getattr(model_module_STN, model_class_STN)(N=N_STN, params=STN_params_converted, neuron_type="E")
    msnd1_model = getattr(model_module_msnd1, model_class_msnd1)(N=N_MSN, params=msnd1_params_converted, neuron_type="E")
    msnd2_model = getattr(model_module_msnd2, model_class_msnd2)(N=N_MSN, params=msnd2_params_converted, neuron_type="E")
    snr_model = getattr(model_module_snr, model_class_snr)(N=N_SNr, params=snr_params_converted, neuron_type="E")

    # Create neurons for GPe, STN, and MSND2
    FSN = fsn_model.create_neurons()
    GPeT1 = gpet1_model.create_neurons()
    GPeTA = gpeta_model.create_neurons()
    STN = STN_model.create_neurons()
    MSND1 = msnd1_model.create_neurons()
    MSND2 = msnd2_model.create_neurons()
    SNr = snr_model.create_neurons()

    # Create Cortex neuron group as a PoissonGroup
    N_Cortex = N_STN  # Set the number of Cortex neurons equal to the number of STN neurons
    sigma = 3 * Hz 
    
    # rates = TimedArray([0 * Hz, 200 * Hz, 0 * Hz], dt=300*ms)  # 0 Hz from 0-200 ms, 200 Hz from 200-500 ms, 0 Hz after
    Cortex = PoissonGroup(N_Cortex, rates='50*Hz + (t >= 200*ms) * (t < 400*ms) * 200*Hz + sigma * randn()')

    # Set up synapses (inhibitory and excitatory) using an imported synapse model
    synapse_module = importlib.import_module(f'Neuronmodels.{synapse_class}')
    synapse_instance = synapse_module.Synapse(FSN, GPeT1, GPeTA, STN, MSND1, MSND2, SNr, Cortex, synapse_params)

    # Create synapses from GPe to STN, Cortex to MSND2, and Cortex to STN
    syn_Cortex_FSN, syn_Cortex_MSND1, syn_Cortex_MSND2, syn_Cortex_STN, syn_FSN_FSN, syn_FSN_MSND1, syn_FSN_MSND2, syn_MSND1_SNr, syn_MSND1_MSND1, syn_MSND1_MSND2, syn_MSND2_MSND2, syn_MSND2_MSND1, syn_MSND2_GPeT1, syn_STN_GPeT1, syn_STN_GPeTA, syn_STN_SNr, syn_GPeT1_FSN, syn_GPeT1_STN, syn_GPeT1_SNr, syn_GPeT1_GPeT1, syn_GPeT1_GPeTA, syn_GPeTA_GPeT1, syn_GPeTA_GPeTA, syn_GPeTA_FSN, syn_GPeTA_MSND1, syn_GPeTA_MSND2 = synapse_instance.create_synapse()
    
    # Set up monitors to track membrane potentials and spikes in each neuron group
    dv_monitor_fsn = StateMonitor(FSN, 'v', record=True)
    dv_monitor_gpet1 = StateMonitor(GPeT1, 'v', record=True)
    dv_monitor_gpeta = StateMonitor(GPeTA, 'v', record=True)
    dv_monitor_STN = StateMonitor(STN, ['v', 'u'], record=True)
    dv_monitor_msnd1= StateMonitor(MSND1, 'v', record=True)
    dv_monitor_msnd2= StateMonitor(MSND2, 'v', record=True)
    dv_monitor_snr= StateMonitor(SNr, 'v', record=True)
    spike_monitor_fsn = SpikeMonitor(FSN)
    spike_monitor_gpet1 = SpikeMonitor(GPeT1)
    spike_monitor_gpeta = SpikeMonitor(GPeTA)
    spike_monitor_STN = SpikeMonitor(STN)
    spike_monitor_cortex = SpikeMonitor(Cortex)
    spike_monitor_msnd1 = SpikeMonitor(MSND1)
    spike_monitor_msnd2 = SpikeMonitor(MSND2)
    spike_monitor_snr= SpikeMonitor(SNr)

    
    # Create a network and add components to it
    net = Network(FSN, GPeT1, GPeTA, STN, MSND1, MSND2, SNr, Cortex, syn_Cortex_FSN, syn_Cortex_MSND1, syn_Cortex_MSND2, syn_Cortex_STN, syn_FSN_FSN, syn_FSN_MSND1, syn_FSN_MSND2, syn_MSND1_SNr, syn_MSND1_MSND1, syn_MSND1_MSND2, syn_MSND2_MSND2, syn_MSND2_MSND1, syn_MSND2_GPeT1, syn_STN_GPeT1, syn_STN_GPeTA, syn_STN_SNr, syn_GPeT1_FSN, syn_GPeT1_STN, syn_GPeT1_SNr, syn_GPeT1_GPeT1, syn_GPeT1_GPeTA, syn_GPeTA_GPeT1, syn_GPeTA_GPeTA, syn_GPeTA_FSN, syn_GPeTA_MSND1, syn_GPeTA_MSND2, 
                  dv_monitor_fsn, dv_monitor_gpet1, dv_monitor_gpeta, dv_monitor_STN, dv_monitor_msnd1, dv_monitor_msnd2, dv_monitor_snr, spike_monitor_fsn, spike_monitor_gpet1, spike_monitor_gpeta, 
                  spike_monitor_STN, spike_monitor_cortex, spike_monitor_msnd1, spike_monitor_msnd2, spike_monitor_snr)

    # Run the network simulation
    simulation_duration = 1000 * ms
    net.run(1000*ms)
    
    fsn_firing_rate = spike_monitor_fsn.count / (simulation_duration / second)
    gpet1_firing_rate = spike_monitor_gpet1.count / (simulation_duration / second)
    gpeta_firing_rate = spike_monitor_gpeta.count / (simulation_duration / second)
    STN_firing_rate = spike_monitor_STN.count / (simulation_duration / second)
    cortex_firing_rate = spike_monitor_cortex.count / (simulation_duration / second)
    msnd1_firing_rate = spike_monitor_msnd1.count / (simulation_duration / second)
    msnd2_firing_rate = spike_monitor_msnd2.count / (simulation_duration / second)
    snr_firing_rate = spike_monitor_snr.count / (simulation_duration / second)

    # Return results for analysis
    return {
        'fsn_times': dv_monitor_fsn.t / ms,
        'fsn_membrane_potential': dv_monitor_fsn.v[0] / mV,
        'gpet1_times': dv_monitor_gpet1.t / ms,
        'gpeta_times': dv_monitor_gpeta.t / ms,
        'gpet1_membrane_potential': dv_monitor_gpet1.v[0] / mV,
        'gpet1_times': dv_monitor_gpet1.t / ms,
        'gpeta_membrane_potential': dv_monitor_gpeta.v[0] / mV,
        'gpeta_times': dv_monitor_gpeta.t / ms,
        'STN_times': dv_monitor_STN.t / ms,
        'STN_membrane_potential': dv_monitor_STN.v[0] / mV,
        'msnd1_times': dv_monitor_msnd1.t / ms,
        'msnd2_times': dv_monitor_msnd2.t / ms,
        'snr_times': dv_monitor_snr.t / ms,
        'msnd1_membrane_potential': dv_monitor_msnd1.v[0] / mV,
        'msnd2_membrane_potential': dv_monitor_msnd2.v[0] / mV,
        'fsn_spikes': spike_monitor_fsn.count,
        'gpet1_spikes': spike_monitor_gpet1.count,
        'gpeta_spikes': spike_monitor_gpeta.count,
        'STN_spikes': spike_monitor_STN.count,
        'MSND1_spikes': spike_monitor_msnd1.count,
        'MSND2_spikes': spike_monitor_msnd2.count,        
        'SNr_spikes': spike_monitor_snr.count,
        'snr_membrane_potential': dv_monitor_snr.v[0] / mV,
        'cortex_spikes': spike_monitor_cortex.count,
        'firing_rates': {
            'fsn': fsn_firing_rate,
            'gpet1': gpet1_firing_rate,
            'gpeta': gpeta_firing_rate,
            'STN': STN_firing_rate,
            'cortex': cortex_firing_rate,
            'msnd1': msnd1_firing_rate,
            'msnd2': msnd2_firing_rate,
            'snr': snr_firing_rate
        },
        'spike_monitor_fsn': spike_monitor_fsn,
        'spike_monitor_gpet1': spike_monitor_gpet1,
        'spike_monitor_gpeta': spike_monitor_gpeta,
        'spike_monitor_STN': spike_monitor_STN,
        'spike_monitor_msnd1': spike_monitor_msnd1, 
        'spike_monitor_msnd2': spike_monitor_msnd2, 
        'spike_monitor_snr': spike_monitor_snr, 
        'spike_monitor_cortex': spike_monitor_cortex 
    }


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

### Visualization with statemonitor result 
def plot_raster(results):
    plt.figure(figsize=(8, 12))

    # 1. Cortex Neuron
    plt.subplot(8, 1, 1)
    plt.scatter(results['spike_monitor_cortex'].t/ms, results['spike_monitor_cortex'].i, s=2, color='red')
    plt.title('Cortex Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)

    # 2. MSND1 Neuron
    plt.subplot(8, 1, 2)
    plt.scatter(results['spike_monitor_msnd1'].t/ms, results['spike_monitor_msnd1'].i, s=2, color='orange')
    plt.title('MSND1 Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)
    
    # 3. MSND2 Neuron
    plt.subplot(8, 1, 3)
    plt.scatter(results['spike_monitor_msnd2'].t/ms, results['spike_monitor_msnd2'].i, s=2, color='orange')
    plt.title('MSND2 Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)

    # 4. FSM Neuron
    plt.subplot(8, 1, 4)
    plt.scatter(results['spike_monitor_fsn'].t/ms, results['spike_monitor_fsn'].i, s=2, color='orange')
    plt.title('FSN Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)

    # 5. GPe Neuron
    plt.subplot(8, 1, 5)
    plt.scatter(results['spike_monitor_gpet1'].t/ms, results['spike_monitor_gpet1'].i, s=2, color='blue')
    plt.title('GPeT1 Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)
    
    # 6. GPe Neuron
    plt.subplot(8, 1, 6)
    plt.scatter(results['spike_monitor_gpeta'].t/ms, results['spike_monitor_gpeta'].i, s=2, color='blue')
    plt.title('GPeTA Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)
   
    # 7. STN Neuron
    plt.subplot(8, 1, 7)
    plt.scatter(results['spike_monitor_STN'].t/ms, results['spike_monitor_STN'].i, s=2, color='green')
    plt.title('STN Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)

    # 8. SNr Neuron
    plt.subplot(8, 1, 8)
    plt.scatter(results['spike_monitor_snr'].t/ms, results['spike_monitor_snr'].i, s=2, color='green')
    plt.title('SNr Population Raster Plot')
    #plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.xlim(0, 1000)

    plt.tight_layout()
    plt.show()

def plot_membrane_potentials(results):
    plt.figure(figsize=(15, 12))
    
    # 1. Cortex와 FSN
    plt.subplot(5, 1, 1)
    plt.plot(results['fsn_times'], results['fsn_membrane_potential'], label='FSN', color='orange')
    plt.title('FSN Membrane Potential')
    plt.ylabel('Membrane\nPotential (mV)')
    plt.legend()
    plt.grid(True)

    # 2. GPe (T1, TA)
    plt.subplot(5, 1, 2)
    plt.plot(results['gpet1_times'], results['gpet1_membrane_potential'], label='GPe-T1', color='blue')
    plt.plot(results['gpeta_times'], results['gpeta_membrane_potential'], label='GPe-TA', color='skyblue')
    plt.title('GPe Membrane Potentials')
    plt.ylabel('Membrane\nPotential (mV)')
    plt.legend()
    plt.grid(True)

    # 3. STN
    plt.subplot(5, 1, 3)
    plt.plot(results['STN_times'], results['STN_membrane_potential'], label='STN', color='green')
    plt.title('STN Membrane Potentials')
    plt.ylabel('Membrane\nPotential (mV)')
    plt.legend()
    plt.grid(True)

    # 4. MSN
    plt.subplot(5, 1, 4)
    plt.plot(results['msnd1_times'], results['msnd1_membrane_potential'], label='MSN-D1', color='red')
    plt.plot(results['msnd2_times'], results['msnd2_membrane_potential'], label='MSN-D2', color='purple')
    plt.title('MSN Membrane Potentials')
    plt.ylabel('Membrane\nPotential (mV)')
    plt.legend()
    plt.grid(True)

    # 5. SNr
    plt.subplot(5, 1, 5)
    plt.plot(results['snr_times'], results['snr_membrane_potential'], label='SNr', color='darkgreen')
    plt.title('SNr Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane\nPotential (mV)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()