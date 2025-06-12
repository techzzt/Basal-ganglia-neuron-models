from brian2 import *
import numpy as np

def create_poisson_inputs(neuron_groups, external_inputs, scaled_neuron_counts, amplitude_oscillations=None):
    """Create external inputs (with amplitude oscillation support)"""
    poisson_groups = {}
    poisson_synapses = []
    
    # Default amplitude oscillation values
    default_amplitudes = {
        'MSND1': 0.11,
        'MSND2': 0.11, 
        'FSN': 0.11,
        'STN': 0.11  
    }
    
    if amplitude_oscillations is None:
        amplitude_oscillations = default_amplitudes
    
    for target, rate_expr in external_inputs.items():
        if target not in neuron_groups:
            # print(f"Warning: Target group '{target}' not found in neuron groups")
            continue
        
        try:
            if target in scaled_neuron_counts and target != 'Cortex':
                N = scaled_neuron_counts[target]
                original_n = neuron_groups[target].N
                # print(f"Found {N} Cortex neurons for {target}")
            else:
                N = neuron_groups[target].N
                original_n = N
                # print(f"Using {N} target neurons for {target} (Ext input)")

            # Apply amplitude oscillation
            amplitude = amplitude_oscillations.get(target, 1.0)
            
            # Extract oscillation frequency from rate_expr
            if isinstance(rate_expr, str) and '*Hz' in rate_expr:
                # Extract frequency value (e.g., "646*Hz" -> 646)
                osc_freq = float(rate_expr.replace('*Hz', '').strip())
            else:
                osc_freq = float(rate_expr) if isinstance(rate_expr, (int, float)) else 20.0
            
            # Process rate expression
            if isinstance(rate_expr, str) and '*Hz' in rate_expr:
                if rate_expr.startswith('TimedArray'):
                    # TimedArray-based time-varying rate
                    total_rate = eval(rate_expr.split(' * ')[0].replace('TimedArray(', '').replace(')', '').replace('[', '').replace(']', '').split(',')[0])
                    rate_per_neuron = total_rate / original_n
                    oscillating_rate = f'{rate_per_neuron/Hz} * (1 + {amplitude} * sin(2*pi*{osc_freq}*t/second)) * Hz'
                    group = PoissonGroup(N, rates=oscillating_rate)
                    # print(f"Created oscillating Poisson input for {target}: {N} neurons with amplitude {amplitude}")
                else:
                    # Simple rate expression like "646*Hz"
                    base_rate = eval(rate_expr)
                    rate_per_neuron = base_rate / original_n
                    oscillating_rate = f'{rate_per_neuron/Hz} * (1 + {amplitude} * sin(2*pi*{osc_freq}*t/second)) * Hz'
                    group = PoissonGroup(N, rates=oscillating_rate)
                    # print(f"Created oscillating Poisson input for {target}: {N} neurons with amplitude {amplitude}")
            elif isinstance(rate_expr, str):
                base_rate = eval(rate_expr)
                oscillating_rate = f'{base_rate/Hz} * (1 + {amplitude} * sin(2*pi*{osc_freq}*t/second)) * Hz'
                group = PoissonGroup(N, rates=oscillating_rate)
                # print(f"Created oscillating fallback Poisson input for {target} with amplitude {amplitude}")
            else:
                rate_per_neuron = rate_expr / original_n if original_n > 0 else 0
                oscillating_rate = f'{rate_per_neuron/Hz} * (1 + {amplitude} * sin(2*pi*{osc_freq}*t/second)) * Hz'
                group = PoissonGroup(N, rates=oscillating_rate)
                # print(f"Created oscillating numeric Poisson input for {target} with amplitude {amplitude}")
            
            poisson_groups[target] = group

        except Exception as e:
            print(f"Error creating Poisson inputs: {str(e)}")
    
    return poisson_groups, poisson_synapses
