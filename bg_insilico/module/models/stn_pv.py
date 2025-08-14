import numpy as np
from brian2 import *
import json, os

def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_single_stn_test(param_path, duration_ms=1000):
    params = load_params(param_path)["params"]
    defaultclock.dt = 0.1*ms

    eqs = '''
    dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - z + I_ext) / C : volt
    dz/dt = (a * (v - E_L) - z) / tau_w : amp
    g_L    : siemens
    E_L    : volt
    Delta_T: volt
    vt     : volt
    vr     : volt
    tau_w  : second
    th     : volt
    a      : siemens
    d      : amp
    C      : farad
    I_ext  : amp
    '''

    reset = '''
    w_adaptation = z/pA
    w_minus_15 = w_adaptation - 15
    use_paper_reset = int(w_adaptation < 0)
    reset_offset_val = use_paper_reset * (w_minus_15 * int(w_minus_15 >= 20) + 20 * int(w_minus_15 < 20))
    v = vr + reset_offset_val * mV
    z += d
    '''

    G = NeuronGroup(1, eqs, threshold='v > th', reset=reset, method='euler')
    G.g_L = params['g_L']['value'] * eval(params['g_L']['unit'])
    G.E_L = params['E_L']['value'] * eval(params['E_L']['unit'])
    G.Delta_T = params['Delta_T']['value'] * eval(params['Delta_T']['unit'])
    G.vt = params['vt']['value'] * eval(params['vt']['unit'])
    G.vr = params['vr']['value'] * eval(params['vr']['unit'])
    G.tau_w = params['tau_w']['value'] * eval(params['tau_w']['unit'])
    G.th = params['th']['value'] * eval(params['th']['unit'])
    G.a = params['a']['value'] * eval(params['a']['unit'])
    G.d = params['d']['value'] * eval(params['d']['unit'])
    G.C = params['C']['value'] * eval(params['C']['unit'])
    G.I_ext = params['I_ext']['value'] * eval(params['I_ext']['unit'])
    G.v = params.get('v', {"value": -75, "unit": "mV"})['value'] * mV

    Mv = StateMonitor(G, 'v', record=True)
    Ms = SpikeMonitor(G)

    run(duration_ms*ms)

    t = Mv.t/ms
    v = Mv.v[0]/mV
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(t, v, 'b-')
    plt.title(f'STN single neuron test: {os.path.basename(param_path)}')
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    if Ms.num_spikes > 0:
        plt.scatter(Ms.t/ms, np.full_like(Ms.t/ms, np.max(v)), s=15, c='r', edgecolors='k')
    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    base = os.path.join(os.path.dirname(__file__), 'params_ref')
    pv_plus = os.path.join(base, 'STN_PV_positive.json')
    pv_minus = os.path.join(base, 'STN_PV_negative.json')
    print('Testing PV+...')
    run_single_stn_test(pv_plus, duration_ms=1000)
    print('Testing PV-...')
    run_single_stn_test(pv_minus, duration_ms=1000)
import matplotlib.pyplot as plt
import random

def create_non_overlapping_poisson_input_visualization_and_zoom():
    duration = 10000  # ms
    dt = 0.1  # ms
    n_neurons = 20
    base_rate = 2.0
    stimulus_rate = 5.0
    stim_start = 6000
    stim_end = 7000

    time_vector = np.arange(0, duration, dt)
    neuron_spikes = {i: [] for i in range(n_neurons)}

    for t in time_vector:
        # 자극 구간인지 확인
        rate = stimulus_rate if stim_start <= t <= stim_end else base_rate
        # 각 time마다 1개의 spike만 허용 (어떤 neuron에서 터질지 랜덤)
        prob = rate * dt / 1000.0 * n_neurons
        if random.random() < prob:
            neuron_idx = random.randint(0, n_neurons - 1)
            neuron_spikes[neuron_idx].append(t)
    
    # === 전체 래스터 플롯 ===
    fig, ax = plt.subplots(figsize=(16, 8))
    for neuron_idx in range(n_neurons):
        spikes = neuron_spikes[neuron_idx]
        if spikes:
            ax.scatter(spikes, [neuron_idx] * len(spikes), s=20, color='gold', alpha=0.9)
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neuron Index', fontsize=12, fontweight='bold')
    ax.set_title('Non-overlapping Poisson Input Raster Plot (All Neurons)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.axvspan(stim_start, stim_end, alpha=0.1, color='red', label='Stimulus Period')
    ax.legend()
    plt.tight_layout()
    plt.savefig('non_overlapping_poisson_input.png', dpi=300, bbox_inches='tight')
    print("전체 래스터 plot 저장됨: 'non_overlapping_poisson_input.png'")

    # === 자극 근처 구간 zoom-in ===
    zoom_margin = 200  # ms
    zoom_start = stim_start - zoom_margin
    zoom_end = stim_end + zoom_margin

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for neuron_idx in range(n_neurons):
        spikes = [t for t in neuron_spikes[neuron_idx] if zoom_start <= t <= zoom_end]
        if spikes:
            ax2.scatter(spikes, [neuron_idx] * len(spikes), s=30, color='gold', alpha=0.9)
    ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Neuron Index', fontsize=12, fontweight='bold')
    ax2.set_title(f'Zoomed Raster: Stimulus ({stim_start}~{stim_end} ms, ±{zoom_margin} ms)', fontsize=14, fontweight='bold')
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_ylim(-0.5, n_neurons - 0.5)
    ax2.axvspan(stim_start, stim_end, alpha=0.1, color='red', label='Stimulus Period')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('zoomed_non_overlapping_poisson_input.png', dpi=300, bbox_inches='tight')
    print("Zoom-in 래스터 plot 저장됨: 'zoomed_non_overlapping_poisson_input.png'")

    plt.show()

if __name__ == "__main__":
    create_non_overlapping_poisson_input_visualization_and_zoom()
