#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *
import numpy as np

def _build_adex_equations():
    eqs = (
        'dv/dt = (g_L*(E_L - v) + g_L*Delta_T*exp((v - vt)/Delta_T) - w + I_ext)/C : volt\n'
        'dw/dt = (a*(v - E_L) - w)/tau_w : amp\n'
        'I_ext : amp\n'
        'a : siemens\n'
        'tau_w : second\n'
        'Delta_T : volt\n'
        'E_L : volt\n'
        'g_L : siemens\n'
        'C : farad\n'
        'vt : volt\n'
        'vr : volt\n'
        'd : amp\n'
    )
    return eqs

def _burst_fraction(spike_times, isi_thr_ms=10.0):
    if len(spike_times) < 3:
        return 0.0
    isi = np.diff(spike_times)
    if len(isi) == 0:
        return 0.0
    return float(np.mean(isi < isi_thr_ms))

def scan_stn_burst(params_base: dict, duration_ms: float = 2000.0,
                   grid: dict = None, isi_thr_ms: float = 10.0, seed_val: int = 42):
    if grid is None:
        grid = {
            'a': [1e-9, 3e-9, 5e-9],
            'tau_w': [150e-3, 200e-3, 300e-3],
            'Delta_T': [6e-3, 8e-3, 12e-3],
            'E_L': [-78e-3, -75e-3, -72e-3],
            'I_ext': [80e-12, 100e-12, 120e-12],
        }

    defaultclock.dt = 0.1*ms
    rng = np.random.RandomState(seed_val)

    g_L = params_base.get('g_L', 10e-9)
    C = params_base.get('C', 60e-12)
    vt = params_base.get('vt', -64e-3)
    vr = params_base.get('vr', -70e-3)
    d = params_base.get('d', 0.05e-12)

    combos = []
    for a in grid['a']:
        for tau_w in grid['tau_w']:
            for Delta_T in grid['Delta_T']:
                for E_L in grid['E_L']:
                    for I_ext in grid['I_ext']:
                        combos.append({'a': a, 'tau_w': tau_w, 'Delta_T': Delta_T,
                                       'E_L': E_L, 'I_ext': I_ext})

    results = []
    for ps in combos:
        start_scope()
        eqs = _build_adex_equations()
        G = NeuronGroup(1, eqs,
                        threshold='v > vt', reset='v = vr; w += d', method='euler')
        G.v = params_base.get('v', -75e-3) * volt
        G.w = 0*amp
        G.I_ext = ps['I_ext'] * amp
        G.a = ps['a'] * siemens
        G.tau_w = ps['tau_w'] * second
        G.Delta_T = ps['Delta_T'] * volt
        G.E_L = ps['E_L'] * volt
        G.g_L = g_L * siemens
        G.C = C * farad
        G.vt = vt * volt
        G.vr = vr * volt
        G.d = d * amp

        M = SpikeMonitor(G)
        run(duration_ms*ms)
        st_ms = np.array(M.t/ms, dtype=float)
        bf = _burst_fraction(st_ms, isi_thr_ms=isi_thr_ms)
        rate_hz = len(st_ms) / (duration_ms/1000.0)
        results.append((ps, bf, rate_hz))

    results.sort(key=lambda x: (-x[1], -x[2]))
    return results

def suggest_stn_burst_params(results, target_burst=0.5):
    for ps, bf, rate in results:
        if bf >= target_burst:
            return ps, bf, rate
    return results[0] if results else (None, 0.0, 0.0)


