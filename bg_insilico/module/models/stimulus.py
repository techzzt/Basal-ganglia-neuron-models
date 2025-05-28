from brian2 import *
import numpy as np

def create_poisson_inputs(neuron_groups, ext_inputs, chunk_size=1000):
    if not ext_inputs:
        return []
    poisson_groups = []
    try:
        for target, input_config in ext_inputs.items():
            if target not in neuron_groups:
                print(f"Warning: Target group '{target}' not found in neuron groups")
                continue
            target_group = neuron_groups[target]
            N = len(target_group)
            rate = input_config.get('rate', 0)
            if isinstance(rate, str):
                time_array = TimedArray(eval(rate), dt=defaultclock.dt)
                group = PoissonGroup(N, rates='time_array(t)')
            else:
                group = PoissonGroup(N, rates=rate * Hz)
            poisson_groups.append(group)
    except Exception as e:
        print(f"Error creating Poisson inputs: {str(e)}")
        raise
    return poisson_groups