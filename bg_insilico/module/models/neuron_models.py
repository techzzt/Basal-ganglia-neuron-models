from brian2 import *
import importlib
import json

def load_params_from_file(params_file):
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
            return data['params']
    except Exception as e:
        print(f"Load Error: {str(e)}")
        raise

def get_neuron_count(neuron_configs, target_name):
    for config in neuron_configs:
        if config["name"] == target_name:
            return config["N"]
    return None  

def create_neurons(neuron_configs, connections=None):  
    try:
        neuron_groups = {}
        
        for config in neuron_configs:
            name = config['name']
            N = config['N']
            
            if name == 'Cortex':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)
                        
                        if target_N is None:
                            print(f"Warning: Could not find neuron count for target '{target}'")
                            continue

                        rate_equation = rate_info['equation']
                        evaluated_rate = eval(rate_equation)  
                        rate_per_neuron = evaluated_rate / target_N  
                        # print("rate_per_neuron", rate_per_neuron)
                        # print(f"Creating PoissonGroup: {target} (N={target_N}, rate={rate_equation})")
                        neuron_group = PoissonGroup(target_N, rates=rate_per_neuron)
                        group_name = f'Cortex_{target}'
                        neuron_groups[group_name] = neuron_group

                continue
            
            if 'model_class' in config and 'params_file' in config:
                params = load_params_from_file(config['params_file'])
                
                module_name = f"Neuronmodels.{config['model_class']}"
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, config['model_class'])
                
                model_instance = model_class(N, params, connections)  
                neuron_group = model_instance.create_neurons()
            
                neuron_groups[name] = neuron_group
                print(f"Created {name} neurons using {config['model_class']}")
            
        return neuron_groups
        
    except Exception as e:
        print(f"Error creating neuron groups: {str(e)}")
        print(f"Failed configuration: {config}")
        raise
