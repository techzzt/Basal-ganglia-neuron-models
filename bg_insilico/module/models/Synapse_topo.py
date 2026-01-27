#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import numpy as np
import os
import pickle
from brian2 import *
from module.models.Synapse import (
    get_synapse_class,
    generate_synapse_inputs,
    generate_neuron_specific_synapse_inputs
)
from module.utils.topographic import (
    TopographicMapper,
    identify_connection_type,
    should_apply_topographic
)
from module.utils.dopamine import compute_connection_probability_scale

def create_synapses_with_topography(neuron_groups, connections, synapse_class_name, 
                                   topo_mapper=None, enable_topographic=True,
                                   dop_cfg: dict = None,
                                   distance_cache: dict = None):

    try:
        synapse_connections = []
        synapse_class = get_synapse_class(synapse_class_name) if isinstance(synapse_class_name, str) else synapse_class_name
        synapse_instance = synapse_class(neurons=neuron_groups, connections=connections)
        created_synapses_map = {}
        
        # Initialize topographic mapper if needed
        if enable_topographic and topo_mapper is None:
            topo_mapper = TopographicMapper(overlap_ratio=0.3, distant_ratio=0.1)
        
        if enable_topographic and topo_mapper is not None:
            for group_name, group in neuron_groups.items():
                if should_apply_topographic(group_name):
                    if group_name not in topo_mapper.region_map:
                        topo_mapper.divide_population(group_name, int(group.N))
                        print(f"Registered {group_name} ({group.N} neurons) for topographic organization")
        
        if enable_topographic and topo_mapper is not None:
            topo_mapper.print_topology_summary()
        
        cache_cfg = distance_cache or {}
        cache_enabled = bool(cache_cfg.get('enabled', False))
        cache_path = cache_cfg.get('path', None)
        _disk_cache = None
        _cache_dirty = False

        def _load_cache():
            nonlocal _disk_cache
            if _disk_cache is not None:
                return _disk_cache
            if not cache_enabled or not cache_path or not os.path.exists(cache_path):
                _disk_cache = {}
                return _disk_cache
            try:
                with open(cache_path, 'rb') as f:
                    _disk_cache = pickle.load(f)
            except Exception:
                _disk_cache = {}
            return _disk_cache

        def _save_cache():
            nonlocal _disk_cache
            if not cache_enabled or not cache_path:
                return
            if _disk_cache is None:
                return
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            except Exception:
                pass
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(_disk_cache, f)
            except Exception:
                pass

        def _get_distance_indices_with_cache(pre_name, post_name, total_connections, distance_decay, seed):
            nonlocal _cache_dirty
            if not cache_enabled:
                return topo_mapper.create_distance_based_connections(pre_name, post_name, total_connections, distance_decay=distance_decay, seed=seed)

            cache = _load_cache()
            N_pre = int(neuron_groups[pre_name].N)
            N_post = int(neuron_groups[post_name].N)
            ov = getattr(topo_mapper, 'overlap_ratio', None)
            dr = getattr(topo_mapper, 'distant_ratio', None)
            key = ('distance', pre_name, post_name, N_pre, N_post, int(total_connections), float(round(distance_decay, 6)), int(seed if seed is not None else -1), float(ov) if ov is not None else None, float(dr) if dr is not None else None)

            if key in cache:
                try:
                    i_arr, j_arr = cache[key]
                    return np.asarray(i_arr, dtype=int), np.asarray(j_arr, dtype=int)
                except Exception:
                    pass

            i_idx, j_idx = topo_mapper.create_distance_based_connections(pre_name, post_name, total_connections, distance_decay=distance_decay, seed=seed)
            try:
                cache[key] = (np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int))
                _cache_dirty = True
            except Exception:
                pass
            return i_idx, j_idx

        # Create synapses for each connection
        for conn_name, conn_config in connections.items():
            pre, post = conn_config['pre'], conn_config['post']
            pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)
            
            if not pre_group or not post_group:
                continue
            
            if 'receptor_type' in conn_config:
                receptor_types = conn_config['receptor_type']
                receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
            elif 'receptor_params' in conn_config:
                receptor_types = list(conn_config['receptor_params'].keys())
            else:
                continue
            
            weight_from_config = conn_config.get('weight', 1.0)
            
            for receptor_type in receptor_types:
                syn_key = (pre, post, receptor_type, conn_name)
                
                if syn_key in created_synapses_map:
                    continue
                
                if receptor_type not in synapse_instance.equations:
                    continue
                
                model_eqns = synapse_instance.equations[receptor_type]
                current_params = conn_config.get('receptor_params', {}).get(receptor_type, {})
                g0_value_for_on_pre = current_params.get('g0', {}).get('value', 0.0)
                tau_value_for_on_pre = current_params.get('tau_syn', {}).get('value', None)
                
                on_pre_code = synapse_instance._get_on_pre(
                    receptor_type, 
                    g0_value_for_on_pre, 
                    tau_value_for_on_pre,
                    weight=weight_from_config,
                    conn_name=conn_name
                )
                
                try:
                    is_external_pre = pre.startswith('Cortex_') or pre.startswith('Ext_')

                    if is_external_pre:
                        N_pre = int(pre_group.N)
                        N_post = int(post_group.N)
                        if N_pre == N_post:
                            i_indices = np.arange(N_pre)
                            j_indices = np.arange(N_post)
                        elif N_pre % N_post == 0 and N_post > 0:
                            i_indices = np.arange(N_pre)
                            j_indices = (np.arange(N_pre) % N_post)
                        else:
                            min_n = min(N_pre, N_post)
                            i_indices = np.arange(min_n)
                            j_indices = np.arange(min_n)

                        region_overrides = conn_config.get('region_specific_receptor_params', {})
                        post_overrides = region_overrides.get('post', {}) if isinstance(region_overrides, dict) else {}
                        motor_override = post_overrides.get('motor', {}).get(receptor_type, {}) if isinstance(post_overrides, dict) else {}

                        if motor_override and enable_topographic and topo_mapper is not None and post in topo_mapper.region_map:
                            motor_indices = set(topo_mapper.get_region_indices(post, 0))
                            mask_motor = np.isin(j_indices, list(motor_indices))
                            i_m, j_m = i_indices[mask_motor], j_indices[mask_motor]
                            i_o, j_o = i_indices[~mask_motor], j_indices[~mask_motor]

                            try:
                                motor_idx_arr = np.array(list(motor_indices), dtype=int)
                                if receptor_type == 'AMPA':
                                    tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                    if tau_val is not None and hasattr(post_group, 'tau_AMPA'):
                                        post_group.tau_AMPA[motor_idx_arr] = tau_val * ms
                                    E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                    if E_rev_val is not None and hasattr(post_group, 'E_AMPA'):
                                        post_group.E_AMPA[motor_idx_arr] = E_rev_val * mV
                                    beta_val = motor_override.get('beta', {}).get('value', None)
                                    if beta_val is not None and hasattr(post_group, 'ampa_beta'):
                                        post_group.ampa_beta[motor_idx_arr] = beta_val
                                
                                elif receptor_type == 'GABA':
                                    tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                    if tau_val is not None and hasattr(post_group, 'tau_GABA'):
                                        post_group.tau_GABA[motor_idx_arr] = tau_val * ms
                                    E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                    if E_rev_val is not None and hasattr(post_group, 'E_GABA'):
                                        post_group.E_GABA[motor_idx_arr] = E_rev_val * mV
                                    beta_val = motor_override.get('beta', {}).get('value', None)
                                    if beta_val is not None and hasattr(post_group, 'gaba_beta'):
                                        post_group.gaba_beta[motor_idx_arr] = beta_val
                                
                                elif receptor_type == 'NMDA':
                                    tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                    if tau_val is not None and hasattr(post_group, 'tau_NMDA'):
                                        post_group.tau_NMDA[motor_idx_arr] = tau_val * ms
                                    E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                    if E_rev_val is not None and hasattr(post_group, 'E_NMDA'):
                                        post_group.E_NMDA[motor_idx_arr] = E_rev_val * mV
                                    beta_val = motor_override.get('beta', {}).get('value', None)
                                    if beta_val is not None and hasattr(post_group, 'nmda_beta'):
                                        post_group.nmda_beta[motor_idx_arr] = beta_val
                            
                            except Exception:
                                pass

                            syn_default = Synapses(pre_group, post_group, model=model_eqns,
                                                   on_pre=synapse_instance._get_on_pre(
                                                       receptor_type,
                                                       current_params.get('g0', {}).get('value', 0.0),
                                                       current_params.get('tau_syn', {}).get('value', None),
                                                       weight=weight_from_config,
                                                       conn_name=conn_name))
                            created_synapses_map[(pre, post, receptor_type, f"{conn_name}:default")] = syn_default
                            synapse_connections.append(syn_default)

                            if len(i_o) > 0:
                                syn_default.connect(i=i_o, j=j_o)

                            # Motor-override synapse
                            syn_motor = Synapses(pre_group, post_group, model=model_eqns,
                                                 on_pre=synapse_instance._get_on_pre(
                                                     receptor_type,
                                                     motor_override.get('g0', {}).get('value', current_params.get('g0', {}).get('value', 0.0)),
                                                     motor_override.get('tau_syn', {}).get('value', current_params.get('tau_syn', {}).get('value', None)),
                                                     weight=weight_from_config,
                                                     conn_name=conn_name))

                            created_synapses_map[(pre, post, receptor_type, f"{conn_name}:motor")] = syn_motor
                            synapse_connections.append(syn_motor)
 
                            if len(i_m) > 0:
                                syn_motor.connect(i=i_m, j=j_m)

                            syn_default.w = weight_from_config
                            syn_motor.w = weight_from_config

                            # Delay override for motor subset
                            try:
                                delay_default = current_params.get('delay', {}).get('value', None)
                                if delay_default is not None:
                                    syn_default.delay = delay_default * ms
  
                            except Exception:
                                pass
  
                            try:
                                delay_motor = motor_override.get('delay', {}).get('value', None)
                                if delay_motor is None:
                                    delay_motor = current_params.get('delay', {}).get('value', None)
                                if delay_motor is not None:
                                    syn_motor.delay = delay_motor * ms
  
                            except Exception:
                                pass

                        else:
                            syn = Synapses(pre_group, post_group, model=model_eqns, on_pre=on_pre_code)
                            created_synapses_map[syn_key] = syn
                            synapse_connections.append(syn)
                            syn.connect(i=i_indices, j=j_indices)
                            syn.w = weight_from_config
                            try:
                                delay_val_ms = current_params.get('delay', {}).get('value', None)
                                if delay_val_ms is not None:
                                    syn.delay = delay_val_ms * ms
                            except Exception:
                                pass

                    elif enable_topographic and topo_mapper is not None:
                        conn_type = identify_connection_type(pre, post)
                        
                        if conn_type is not None and should_apply_topographic(pre) and should_apply_topographic(post):
                            is_feedback = (conn_type == 'feedback')
                            p_original = conn_config.get('p', 1.0)
                            try:
                                p_scale = compute_connection_probability_scale(conn_name, pre, post, dop_cfg or {})
                            except Exception:
                                p_scale = 1.0
                            p_topo = max(0.0, min(1.0, float(p_original) * float(p_scale)))
                            
                            i_indices, j_indices = topo_mapper.create_topographic_connection_indices(
                                pre, post, p_topo, is_feedback=is_feedback, seed=42
                            )
                            
                            if len(i_indices) > 0:
                                region_overrides = conn_config.get('region_specific_receptor_params', {})
                                post_overrides = region_overrides.get('post', {}) if isinstance(region_overrides, dict) else {}
                                motor_override = post_overrides.get('motor', {}).get(receptor_type, {}) if isinstance(post_overrides, dict) else {}

                                if motor_override and post in topo_mapper.region_map:
                                    motor_indices = set(topo_mapper.get_region_indices(post, 0))
                                    mask_motor = np.isin(j_indices, list(motor_indices))
                                    i_m, j_m = i_indices[mask_motor], j_indices[mask_motor]
                                    i_o, j_o = i_indices[~mask_motor], j_indices[~mask_motor]

                                    try:
                                        motor_idx_arr = np.array(list(motor_indices), dtype=int)
                                        if receptor_type == 'AMPA':
                                            tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                            if tau_val is not None and hasattr(post_group, 'tau_AMPA'):
                                                post_group.tau_AMPA[motor_idx_arr] = tau_val * ms
                                            E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                            if E_rev_val is not None and hasattr(post_group, 'E_AMPA'):
                                                post_group.E_AMPA[motor_idx_arr] = E_rev_val * mV
                                            beta_val = motor_override.get('beta', {}).get('value', None)
                                            if beta_val is not None and hasattr(post_group, 'ampa_beta'):
                                                post_group.ampa_beta[motor_idx_arr] = beta_val

                                        elif receptor_type == 'GABA':
                                            tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                            if tau_val is not None and hasattr(post_group, 'tau_GABA'):
                                                post_group.tau_GABA[motor_idx_arr] = tau_val * ms
                                            E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                            if E_rev_val is not None and hasattr(post_group, 'E_GABA'):
                                                post_group.E_GABA[motor_idx_arr] = E_rev_val * mV
                                            beta_val = motor_override.get('beta', {}).get('value', None)
                                            if beta_val is not None and hasattr(post_group, 'gaba_beta'):
                                                post_group.gaba_beta[motor_idx_arr] = beta_val

                                        elif receptor_type == 'NMDA':
                                            tau_val = motor_override.get('tau_syn', {}).get('value', None)
                                            if tau_val is not None and hasattr(post_group, 'tau_NMDA'):
                                                post_group.tau_NMDA[motor_idx_arr] = tau_val * ms
                                            E_rev_val = motor_override.get('E_rev', {}).get('value', None)
                                            if E_rev_val is not None and hasattr(post_group, 'E_NMDA'):
                                                post_group.E_NMDA[motor_idx_arr] = E_rev_val * mV
                                            beta_val = motor_override.get('beta', {}).get('value', None)
                                            if beta_val is not None and hasattr(post_group, 'nmda_beta'):
                                                post_group.nmda_beta[motor_idx_arr] = beta_val

                                    except Exception:
                                        pass

                                    syn_default = Synapses(pre_group, post_group, model=model_eqns,
                                                           on_pre=synapse_instance._get_on_pre(
                                                               receptor_type,
                                                               current_params.get('g0', {}).get('value', 0.0),
                                                               current_params.get('tau_syn', {}).get('value', None),
                                                               weight=weight_from_config,
                                                               conn_name=conn_name))

                                    created_synapses_map[(pre, post, receptor_type, f"{conn_name}:default")] = syn_default
                                    synapse_connections.append(syn_default)

                                    if len(i_o) > 0:
                                        syn_default.connect(i=i_o, j=j_o)

                                    syn_motor = Synapses(pre_group, post_group, model=model_eqns,
                                                         on_pre=synapse_instance._get_on_pre(
                                                             receptor_type,
                                                             motor_override.get('g0', {}).get('value', current_params.get('g0', {}).get('value', 0.0)),
                                                             motor_override.get('tau_syn', {}).get('value', current_params.get('tau_syn', {}).get('value', None)),
                                                             weight=weight_from_config,
                                                             conn_name=conn_name))
                                    created_synapses_map[(pre, post, receptor_type, f"{conn_name}:motor")] = syn_motor
                                    synapse_connections.append(syn_motor)
                                    if len(i_m) > 0:
                                        syn_motor.connect(i=i_m, j=j_m)

                                    syn_default.w = weight_from_config
                                    syn_motor.w = weight_from_config

                                    try:
                                        delay_default = current_params.get('delay', {}).get('value', None)
                                        if delay_default is not None:
                                            syn_default.delay = delay_default * ms
                                    except Exception:
                                        pass

                                    try:
                                        delay_motor = motor_override.get('delay', {}).get('value', None)
                                        if delay_motor is None:
                                            delay_motor = current_params.get('delay', {}).get('value', None)
                                        if delay_motor is not None:
                                            syn_motor.delay = delay_motor * ms
                                    except Exception:
                                        pass

                                    print(f"  {conn_name} ({pre}->{post}, {receptor_type}): "
                                          f"{len(i_indices)} topographic connections with motor override")
                                else:
                                    syn = Synapses(pre_group, post_group, model=model_eqns, on_pre=on_pre_code)
                                    created_synapses_map[syn_key] = syn
                                    synapse_connections.append(syn)
                                    syn.connect(i=i_indices, j=j_indices)
                                    print(f"  {conn_name} ({pre}->{post}, {receptor_type}): "
                                          f"{len(i_indices)} topographic connections "
                                          f"({'feedback' if is_feedback else 'forward'})")
                                    syn.w = weight_from_config

                                    try:
                                        delay_val_ms = current_params.get('delay', {}).get('value', None)
                                        if delay_val_ms is not None:
                                            syn.delay = delay_val_ms * ms

                                    except Exception:
                                        pass
                            else:
                                print(f"  WARNING: {conn_name} has no topographic connections — skipping (no connect)")
                        else:

                            if should_apply_topographic(pre) and should_apply_topographic(post):
                                p = conn_config.get('p', 1.0)

                                try:
                                    p_scale = compute_connection_probability_scale(conn_name, pre, post, dop_cfg or {})
                                except Exception:
                                    p_scale = 1.0

                                p = max(0.0, min(1.0, float(p) * float(p_scale)))
                                N_pre = pre_group.N
                                fan_in = p * N_pre
                                N_beta = current_params.get('N_beta', {}).get('value', 1.0)
                                effective_p = (fan_in * N_beta) / N_pre
                                effective_p = min(effective_p, 1.0)
                                
                                total_connections = int(N_pre * post_group.N * effective_p)
                                
                                try:
                                    i_indices, j_indices = _get_distance_indices_with_cache(
                                        pre, post, total_connections, distance_decay=0.1, seed=42
                                    )

                                    region_overrides = conn_config.get('region_specific_receptor_params', {})
                                    post_overrides = region_overrides.get('post', {}) if isinstance(region_overrides, dict) else {}
                                    motor_override = post_overrides.get('motor', {}).get(receptor_type, {}) if isinstance(post_overrides, dict) else {}

                                    if motor_override and post in topo_mapper.region_map:
                                        motor_indices = set(topo_mapper.get_region_indices(post, 0))
                                        mask_motor = np.isin(j_indices, list(motor_indices))
                                        i_m, j_m = i_indices[mask_motor], j_indices[mask_motor]
                                        i_o, j_o = i_indices[~mask_motor], j_indices[~mask_motor]

                                        syn_default = Synapses(pre_group, post_group, model=model_eqns,
                                                               on_pre=synapse_instance._get_on_pre(
                                                                   receptor_type,
                                                                   current_params.get('g0', {}).get('value', 0.0),
                                                                   current_params.get('tau_syn', {}).get('value', None),
                                                                   weight=weight_from_config,
                                                                   conn_name=conn_name))
                                        created_synapses_map[(pre, post, receptor_type, f"{conn_name}:default")] = syn_default
                                        synapse_connections.append(syn_default)
 
                                        if len(i_o) > 0:
                                            syn_default.connect(i=i_o, j=j_o)

                                        syn_motor = Synapses(pre_group, post_group, model=model_eqns,
                                                             on_pre=synapse_instance._get_on_pre(
                                                                 receptor_type,
                                                                 motor_override.get('g0', {}).get('value', current_params.get('g0', {}).get('value', 0.0)),
                                                                 motor_override.get('tau_syn', {}).get('value', current_params.get('tau_syn', {}).get('value', None)),
                                                                 weight=weight_from_config,
                                                                 conn_name=conn_name))
                                        created_synapses_map[(pre, post, receptor_type, f"{conn_name}:motor")] = syn_motor
                                        synapse_connections.append(syn_motor)

                                        if len(i_m) > 0:
                                            syn_motor.connect(i=i_m, j=j_m)

                                        syn_default.w = weight_from_config
                                        syn_motor.w = weight_from_config

                                        try:
                                            delay_default = current_params.get('delay', {}).get('value', None)
                                            if delay_default is not None:
                                                syn_default.delay = delay_default * ms
                                        except Exception:
                                            pass

                                        try:
                                            delay_motor = motor_override.get('delay', {}).get('value', None)
                                            if delay_motor is None:
                                                delay_motor = current_params.get('delay', {}).get('value', None)
                                            if delay_motor is not None:
                                                syn_motor.delay = delay_motor * ms
                                        except Exception:
                                            pass

                                    else:
                                        syn = Synapses(pre_group, post_group, model=model_eqns, on_pre=on_pre_code)
                                        created_synapses_map[syn_key] = syn
                                        synapse_connections.append(syn)
                                        syn.connect(i=i_indices, j=j_indices)
                                        print(f"  {conn_name} ({pre}->{post}, {receptor_type}): "
                                              f"{len(i_indices)} distance-based connections")
                                        syn.w = weight_from_config

                                        try:
                                            delay_val_ms = current_params.get('delay', {}).get('value', None)
                                            if delay_val_ms is not None:
                                                syn.delay = delay_val_ms * ms
                                        except Exception:
                                            pass

                                except Exception as e:
                                    print(f"  WARNING: Distance-based connection failed: {e} — skipping (no connect)")

                            else:
                                p = conn_config.get('p', 1.0)

                                try:
                                    p_scale = compute_connection_probability_scale(conn_name, pre, post, dop_cfg or {})
                                except Exception:
                                    p_scale = 1.0

                                p = max(0.0, min(1.0, float(p) * float(p_scale)))
                                N_pre = pre_group.N
                                fan_in = p * N_pre
                                N_beta = current_params.get('N_beta', {}).get('value', 1.0)
                                effective_p = (fan_in * N_beta) / N_pre
                                effective_p = min(effective_p, 1.0)
                                syn = Synapses(pre_group, post_group, model=model_eqns, on_pre=on_pre_code)
                                created_synapses_map[syn_key] = syn
                                synapse_connections.append(syn)
                                syn.connect(p=effective_p)
                                print(f"  {conn_name} ({pre}->{post}, {receptor_type}): "
                                      f"standard connectivity (p={effective_p:.3f})")
                                syn.w = weight_from_config

                                try:
                                    delay_val_ms = current_params.get('delay', {}).get('value', None)
                                    if delay_val_ms is not None:
                                        syn.delay = delay_val_ms * ms
                                except Exception:
                                    pass
                    else:
                        p = conn_config.get('p', 1.0)

                        try:
                            p_scale = compute_connection_probability_scale(conn_name, pre, post, dop_cfg or {})
                        except Exception:
                            p_scale = 1.0

                        p = max(0.0, min(1.0, float(p) * float(p_scale)))
                        N_pre = pre_group.N
                        fan_in = p * N_pre
                        N_beta = current_params.get('N_beta', {}).get('value', 1.0)
                        effective_p = (fan_in * N_beta) / N_pre
                        effective_p = min(effective_p, 1.0)
                        syn = Synapses(pre_group, post_group, model=model_eqns, on_pre=on_pre_code)
                        created_synapses_map[syn_key] = syn
                        synapse_connections.append(syn)
                        syn.connect(p=effective_p)
                        syn.w = weight_from_config

                        try:
                            delay_val_ms = current_params.get('delay', {}).get('value', None)
                            if delay_val_ms is not None:
                                syn.delay = delay_val_ms * ms
                        except Exception:
                            pass
                
                except Exception as e:
                    print(f"ERROR creating {receptor_type} synapse for {conn_name}: {str(e)}")
                    raise
        
        try:
            if _cache_dirty:
                _save_cache()
        except Exception:
            pass

        return synapse_connections, created_synapses_map, topo_mapper
    
    except Exception as e:
        print(f"Error creating synapses with topography: {str(e)}")
        raise

