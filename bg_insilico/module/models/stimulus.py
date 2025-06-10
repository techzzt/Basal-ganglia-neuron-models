from brian2 import *
import numpy as np

from brian2 import *
import numpy as np

def create_poisson_inputs(neuron_groups, ext_inputs, chunk_size=1000):
    if not ext_inputs:
        return []
    poisson_groups = []
    try:
        start_scope()
        for target, input_config in ext_inputs.items():
            if target not in neuron_groups:
                print(f"Warning: Target group '{target}' not found in neuron groups")
                continue
            target_group = neuron_groups[target]

            cortex_group_name = f"Cortex_{target}"
            if cortex_group_name in neuron_groups:
                cortex_group = neuron_groups[cortex_group_name]
                N = len(cortex_group)
                print(f"Found {N} Cortex neurons for {target}")
            else:
                N = len(target_group)
                print(f"Using {N} target neurons for {target} (Ext input)")

            rate = input_config.get('rate', 0)
            original_n = input_config.get('original_n', N)

            if isinstance(rate, str):
                try:
                    total_rate = eval(rate)
                    if hasattr(total_rate, 'dim') and total_rate.dim == Hz.dim:
                        rate_per_neuron = total_rate / original_n
                        group = PoissonGroup(N, rates=rate_per_neuron)
                        print(f"Created Poisson input for {target}: {N} neurons @ {rate_per_neuron} each (original_n={original_n}) = {total_rate} total")
                    else:
                        time_array = TimedArray(eval(rate), dt=defaultclock.dt)
                        group = PoissonGroup(N, rates='time_array(t)', namespace={'time_array': time_array})
                        print(f"Created time-varying Poisson input for {target}: {rate}")
                except:
                    group = PoissonGroup(N, rates=eval(rate))
                    print(f"Created fallback Poisson input for {target}: {rate}")
            else:
                rate_per_neuron = rate * Hz / original_n
                group = PoissonGroup(N, rates=rate_per_neuron)
                print(f"Created numeric Poisson input for {target}: {N} neurons @ {rate_per_neuron} each (original_n={original_n})")

            group.namespace.update({'Hz': Hz, 'ms': ms, 'second': second, 'np': np})
            poisson_groups.append(group)

    except Exception as e:
        print(f"Error creating Poisson inputs: {str(e)}")
        raise
    return poisson_groups
