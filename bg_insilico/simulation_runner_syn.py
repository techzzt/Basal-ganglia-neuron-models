import json
from brian2 import *
from Neuronmodels.GPe_STN import GPeModel, STNModel, GPeSTNSynapse
import matplotlib.pyplot as plt

def load_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    params = data['params']
    model_name = data['model']
    
    # Extract 'N' from params
    N = params.pop('N')['value']  # Extract the value of N and remove it from the params
    return N, params, model_name

def convert_units(params):
    converted_params = {}
    for param, info in params.items():
        value = info['value']
        unit = info['unit']
        # Convert units according to the specifications
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

def run_simulation(gpe_params_file, stn_params_file, synapse_params):
    # Load GPe and STN parameters from the JSON files
    N_GPe, gpe_params, gpe_model_name = load_params(gpe_params_file)
    N_STN, stn_params, stn_model_name = load_params(stn_params_file)

    # Convert units for the neuron models
    gpe_params_converted = convert_units(gpe_params)
    stn_params_converted = convert_units(stn_params)
    
    # Initialize the neuron models
    gpe_model = GPeModel(N=N_GPe, params=gpe_params_converted)
    stn_model = STNModel(N=N_STN, params=stn_params_converted)

    GPe = gpe_model.create_neurons()
    STN = stn_model.create_neurons()

    # Set up the synapses between GPe and STN
    synapse = GPeSTNSynapse(GPe, STN, synapse_params)
    syn_GPe_STN = synapse.create_synapse()

    # Set up the monitors
    dv_monitor = StateMonitor(STN, 'v', record=True)
    spike_monitor = SpikeMonitor(STN)

    # Create a network and run the simulation
    net = Network(GPe, STN, syn_GPe_STN, dv_monitor, spike_monitor)
    net.run(1000*ms)

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
