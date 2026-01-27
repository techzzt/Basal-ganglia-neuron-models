# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import *

def phi(alpha_dop: float, alpha0: float = 0.8) -> float:
    return float(alpha_dop) - float(alpha0)

def compute_EL_with_dopamine(EL_high: float, alpha_dop: float, shift_low_vs_high_mV: float = 10.0) -> float:
    return float(EL_high) - float(shift_low_vs_high_mV) * (1.0 - float(alpha_dop))

def _extract_base_value(params_dict: dict, key: str, default_val: float) -> float:
    try:
        if key in params_dict:
            v = params_dict[key]
            if isinstance(v, dict):
                return float(v.get('value', default_val))
            return float(v)
    except Exception:
        pass
    return float(default_val)

def apply_intrinsic_dopamine(neuron_group, base_params: dict, dop_cfg: dict, group_name: str):

    if not dop_cfg or not dop_cfg.get('enabled', False):
        return

    alpha = float(dop_cfg.get('alpha_dop', 0.8))
    rules_all = dop_cfg.get('rules', {}) if isinstance(dop_cfg, dict) else {}
    rules = rules_all.get(group_name, rules_all.get('*', {})) if isinstance(rules_all, dict) else {}

    delta_EL = float(rules.get('EL_delta_mV', dop_cfg.get('EL_delta_mV', 0.0)))
    if abs(delta_EL) > 0:
        base_EL = _extract_base_value(base_params, 'E_L', -80.0)
        new_EL_mV = compute_EL_with_dopamine(base_EL, alpha_dop=alpha, shift_low_vs_high_mV=delta_EL)
        try:
            neuron_group.E_L = new_EL_mV * mV
        except Exception:
            pass

    delta_vr = float(rules.get('vr_delta_mV', 0.0))
    if abs(delta_vr) > 0:
        base_vr = _extract_base_value(base_params, 'vr', -70.0)
        new_vr_mV = base_vr - delta_vr * (1.0 - alpha)
        try:
            neuron_group.vr = new_vr_mV * mV
        except Exception:
            pass

    d_scale_low = rules.get('d_scale_low', None)
    d_delta_pA = rules.get('d_delta_pA', None)

    if d_scale_low is not None or d_delta_pA is not None:
        base_d = _extract_base_value(base_params, 'd', 0.0)
        new_d = base_d

        if d_scale_low is not None:
            low_scale = float(d_scale_low)
            scale = 1.0 + (low_scale - 1.0) * (1.0 - alpha)
            new_d = base_d * scale
        if d_delta_pA is not None:
            delta_A = float(d_delta_pA) * 1e-12
            new_d = new_d + delta_A * (1.0 - alpha)
        
        try:
            neuron_group.d = new_d * amp
        except Exception:
            pass

def scale_beta_with_dopamine(beta_base: float, dop_cfg: dict, k_beta: float = 1.0) -> float:

    if not dop_cfg or not dop_cfg.get('enabled', False):
        return float(beta_base)
    ph = phi(dop_cfg.get('alpha_dop', 0.8), dop_cfg.get('alpha0', 0.8))
    return float(beta_base) * (1.0 + float(k_beta) * float(ph))

def apply_synaptic_dopamine(synapse_dict: dict, connections_config: dict, dop_cfg: dict):

    if not dop_cfg or not dop_cfg.get('enabled', False):
        return
    
    alpha = float(dop_cfg.get('alpha_dop', 0.8))
    alpha0 = float(dop_cfg.get('alpha0', 0.8))
    ph = phi(alpha, alpha0)
    
    synapse_rules = dop_cfg.get('synapses', {}) if isinstance(dop_cfg, dict) else {}
    
    for (pre, post, receptor, conn_name), syn in synapse_dict.items():
        candidates = []
        if isinstance(conn_name, str):
            candidates.append(conn_name + f"_{receptor}")
            candidates.append(conn_name)
        prepost = f"{pre}_to_{post}"
        candidates.append(prepost + f"_{receptor}")
        candidates.append(prepost)

        rule = {}
        for k in candidates:
            cfg = synapse_rules.get(k, {})
            if not cfg:
                continue

            if receptor in cfg:
                rule = cfg.get(receptor, {})
                break

            if 'receptors' in cfg and isinstance(cfg['receptors'], dict) and receptor in cfg['receptors']:
                rule = cfg['receptors'][receptor]
                break

            if ('beta' in cfg) or ('formula' in cfg):
                rule = cfg
                break

        if not rule:
            continue

        formula_type = rule.get('formula', 'plus_phi')
        beta_val = float(rule.get('beta', 0.0))
        if abs(beta_val) < 1e-6:
            continue
        scale = (1.0 - beta_val * ph) if formula_type == 'minus_phi' else (1.0 + beta_val * ph)

        try:
            if hasattr(syn, 'w'):
                syn.w = syn.w * scale
        except Exception:
            pass

def compute_connection_probability_scale(conn_name: str, pre: str, post: str, dop_cfg: dict) -> float:

    if not dop_cfg or not dop_cfg.get('enabled', False):
        return 1.0
    
    alpha = float(dop_cfg.get('alpha_dop', 0.8))
    alpha0 = float(dop_cfg.get('alpha0', 0.8))
    ph = phi(alpha, alpha0)
    
    synapse_rules = dop_cfg.get('synapses', {}) if isinstance(dop_cfg, dict) else {}
    key = f"{pre}_to_{post}"
    rule = synapse_rules.get(key, {})
    
    if not rule:
        return 1.0
    
    beta_N = float(rule.get('beta_N', 0.0))
    if abs(beta_N) < 1e-6:
        return 1.0
    
    scale = 1.0 + beta_N * ph
    return max(0.0, scale) 


