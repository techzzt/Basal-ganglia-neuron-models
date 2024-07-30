import json
from brian2 import mV, ms, pA, pF


def load_params(file_path):
    with open(file_path, 'r') as f:
        params_json = json.load(f)
    
    # Convert parameters to appropriate units
    params = {
        'N': params_json['N'],
        'v': params_json['v'] * mV,
        'u': params_json['u'] * mV / ms,
        'a': params_json['a'] / ms,
        'b': params_json['b'] / ms,
        'c': params_json['c'] * mV,
        'C': params_json['C'] * pF,
        'd': params_json['d'] * mV / ms,
        'k': params_json['k'],
        'vr': params_json['vr'] * mV,
        'vt': params_json['vt'] * mV,
        'vpeak': params_json['vpeak'] * mV,
        'I': params_json['I'] * pA,
        'Dop1': params_json['Dop1'],
        'Dop2': params_json['Dop2'],
        'KAPA': params_json['KAPA'],
        'ALPHA': params_json['ALPHA']
    }
    return params