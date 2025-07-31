from brian2 import *
import numpy as np

def create_poisson_inputs(neuron_groups, external_inputs, scaled_neuron_counts, neuron_configs=None, amplitude_oscillations=None, stimulus_config=None, simulation_params=None):

    poisson_groups = {}
    
    print("\n" + "="*50)
    print("POISSON INPUT RATES")
    print("="*50)
    
    # Get simulation duration from params
    total_duration = 10000 
    if simulation_params:
        total_duration = simulation_params.get('duration', 10000)
    
    # Stimulus parameters
    if stimulus_config and stimulus_config.get('enabled', False):
        stim_start = stimulus_config.get('start_time', 10000)
        stim_duration = stimulus_config.get('duration', 1000) 
        stim_rates = stimulus_config.get('rates', {})
        dt_array = stimulus_config.get('dt_array', 1)  
        use_stimulus = True
        print(f"Stimulus enabled: {stim_start}-{stim_start + stim_duration}ms")
    else:
        use_stimulus = False
        print("Stimulus disabled")
    
    for target, rate_expr in external_inputs.items():
        if target not in neuron_groups:
            continue
        
        try:
            if target in scaled_neuron_counts and target != 'Cortex':
                N = scaled_neuron_counts[target]
                original_n = neuron_groups[target].N
            else:
                N = neuron_groups[target].N
                original_n = N

            if isinstance(rate_expr, str) and '*Hz' in rate_expr:
                base_rate = eval(rate_expr)
            elif isinstance(rate_expr, str):
                base_rate = eval(rate_expr) * Hz
            else:
                base_rate = rate_expr * Hz
            
            rate_per_neuron_base = base_rate / original_n
            
            print(f"\n[{target}] Input rate: {rate_per_neuron_base/Hz:.3f} Hz/neuron")
            
            if use_stimulus and target in stim_rates and stim_duration > 0:
                baseline_rate_total = base_rate
                baseline_rate_per_neuron = baseline_rate_total / original_n
                
                stim_rate_total = stim_rates[target] * Hz
                stim_rate_per_neuron = stim_rate_total / original_n
                
                print(f"  Stimulus period: {stim_start}-{stim_start + stim_duration}ms")
                print(f"    Baseline: {baseline_rate_per_neuron/Hz:.3f} Hz/neuron")
                print(f"    Stimulus: {stim_rate_per_neuron/Hz:.3f} Hz/neuron (+{((stim_rate_per_neuron - baseline_rate_per_neuron) / baseline_rate_per_neuron * 100):+.1f}%)")
                
                array_duration = total_duration + 1000  
                time_points = np.arange(0, array_duration, dt_array)
                
                # Create rate array - START WITH BASELINE
                rates = np.full(len(time_points), baseline_rate_per_neuron / Hz)
                
                # Replace with stimulus rate during stimulus period (sharp transition)
                stim_start_idx = int(stim_start / dt_array)
                stim_end_idx = int((stim_start + stim_duration) / dt_array)
                
                if stim_start_idx < len(rates):
                    end_idx = min(stim_end_idx, len(rates))
                    rates[stim_start_idx:end_idx] = stim_rate_per_neuron / Hz
                    print(f"    Applied stimulus from index {stim_start_idx} to {end_idx}")
                else:
                    print(f"    WARNING: Stimulus start index {stim_start_idx} >= array length {len(rates)}")
                
                timed_rates = TimedArray(rates * Hz, dt=dt_array*ms)
                
                group = PoissonGroup(N, rates='timed_rates(t)', namespace={'timed_rates': timed_rates})
            
            elif use_stimulus and target in stim_rates and stim_duration == 0:
                # Stimulus enabled but duration is 0 - use constant stimulus rate
                stim_rate_total = stim_rates[target] * Hz
                stim_rate_per_neuron = stim_rate_total / original_n
                
                print(f"  Constant stimulus rate: {stim_rate_per_neuron/Hz:.3f} Hz/neuron")
                
                group = PoissonGroup(N, rates=stim_rate_per_neuron)
            
            else:
                # Simple constant rate

                group = PoissonGroup(N, rates=rate_per_neuron_base)
            
            external_name = None
            for neuron_config in neuron_configs:
                if neuron_config.get('neuron_type') == 'poisson':
                    if 'target_rates' in neuron_config:
                        config_target, _ = list(neuron_config['target_rates'].items())[0]
                        if config_target == target:
                            external_name = neuron_config['name']
                            break
            
            if external_name:
                poisson_groups[external_name] = group
            else:
                poisson_groups[target] = group

        except Exception as e:
            print(f"Error creating Poisson inputs for {target}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal Poisson groups: {len(poisson_groups)}")
    print("="*50)
    
    return poisson_groups, []
