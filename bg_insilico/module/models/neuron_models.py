from brian2 import *
import importlib
import json

def load_params_from_file(params_file):
    """JSON 파일에서 파라미터 로드"""
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
            # params 키 아래의 파라미터들을 반환
            return data['params']
    except Exception as e:
        print(f"파라미터 파일 로드 중 오류 발생: {str(e)}")
        print(f"파일 경로: {params_file}")
        raise

def create_neurons(neuron_configs):

    try:
        neuron_groups = {}
        
        for config in neuron_configs:
            name = config['name']
            N = config['N']
            
            # Cortex
            if name == 'Cortex':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        rate_equation = rate_info['equation']
                        group_name = f'Cortex_{target}'
                        neuron_groups[group_name] = PoissonGroup(N, rates=rate_equation)
                continue
            
            if 'model_class' in config and 'params_file' in config:
                # load parameter
                params = load_params_from_file(config['params_file'])

                module_name = f"Neuronmodels.{config['model_class']}"
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, config['model_class'])
                
                model_instance = model_class(N, params)
                group = model_instance.create_neurons()
                neuron_groups[name] = group
                
        return neuron_groups
        
    except Exception as e:
        print(f"뉴런 그룹 생성 중 오류 발생: {str(e)}")
        print(f"문제가 발생한 설정: {config}")
        raise

def generate_rate_equation(params):
    return (f"0*Hz + (t >= {params['start_time']}*ms) * "
            f"(t < {params['end_time']}*ms) * {params['peak_rate']}*Hz + "
            f"{params['noise_amplitude']}*Hz * randn()")