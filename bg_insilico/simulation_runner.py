import numpy as np
from brian2 import ms, mV, nA, pA, StateMonitor, SpikeMonitor, PopulationRateMonitor
from SPN import NeuronModel
from simulation import Simulation

def run_simulation(N, params, v_reset):
    # Initialize neuron model and simulation
    neuron_model = NeuronModel(N, params)
    sim = Simulation(neuron_model)
    
    # Initial run of the simulation
    Initialize_time = 1000 * ms
    sim.run(duration=Initialize_time)
    
    # Initial run of the simulation
    times = sim.dv_monitor.t
    membrane_potential = sim.dv_monitor.v[0]
    matching_indices = np.where(membrane_potential / mV >= v_reset / mV)[0]

    if len(matching_indices) > 0:
        earliest_time_stabilized = times[matching_indices[0]] * 1000
    else:
        earliest_time_stabilized = None

    print("Earliest time when v stabilizes at v_reset (in ms):", earliest_time_stabilized)
    
    # Create new monitors for the second phase
    sim.dv_monitor_new = StateMonitor(neuron_model.neurons, 'v', record=True)
    sim.spike_monitor_new = SpikeMonitor(neuron_model.neurons)
    sim.rate_monitor_new = PopulationRateMonitor(neuron_model.neurons)
    sim.network.add(sim.dv_monitor_new, sim.spike_monitor_new, sim.rate_monitor_new)
    
    # Run the second phase of the simulation
    if earliest_time_stabilized is not None:
        wait_time_after_stabilization = 1000 * ms
        sim.network.run(wait_time_after_stabilization)
        neuron_model.neurons.I = 1 * nA
        time_after_increase = 1000 * ms
        sim.network.run(time_after_increase)

        neuron_model.neurons.I = 0 * pA
        time_after_decrease = 1000 * ms
        sim.network.run(time_after_decrease)

        simulation_time = 3000 * ms
        remaining_time = simulation_time - earliest_time_stabilized - wait_time_after_stabilization - time_after_increase - time_after_decrease
        sim.network.run(remaining_time)
    else:
        print("v does not reach v_reset, stopping simulation")

    sim.plot_results(earliest_time_stabilized=earliest_time_stabilized)
