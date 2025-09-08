# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import mV, pA
import numpy as np

# Get parameter with unit
def _get_param_quantity(params, key, default_value=0.0, default_unit=None):
    if key not in params:
        if default_unit is None:
            return default_value
        return default_value * eval(default_unit)
    value = params[key].get('value', default_value)
    unit = params[key].get('unit', default_unit)
    if unit is None:
        return value
    return value * eval(unit)

# Compute QIF fixed points
def compute_qif_fixed_points(params, I_override=None):
    vr = _get_param_quantity(params, 'vr')
    vt = _get_param_quantity(params, 'vt')
    b = params.get('b', {}).get('value', 0.0)
    k = params.get('k', {}).get('value', 1.0)
    I = I_override if I_override is not None else _get_param_quantity(params, 'I_ext', 0.0, 'pA')

    vr_mv = float(vr / mV)
    vt_mv = float(vt / mV)
    I_pA = float(I / pA)

    A = vr_mv + vt_mv + b / k
    B2 = A * A - 4.0 * (vr_mv * vt_mv + (b * vr_mv + I_pA) / k)
    if B2 < 0:
        B2 = 0.0
    B = float(np.sqrt(B2))
    v_down = (A - B) / 2.0
    v_up = (A + B) / 2.0
    return v_down * mV, v_up * mV

# Compute QIF down state
def compute_qif_v_down(params, I_override=None, use_literature_values=True):
    if use_literature_values:
        return -75.0 * mV
    else:
        v_down, v_up = compute_qif_fixed_points(params, I_override)
        return v_down

# Get AdEx resting voltage
def compute_adex_v_rest(params):
    return _get_param_quantity(params, 'vr')