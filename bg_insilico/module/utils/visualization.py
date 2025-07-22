import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
import platform
system = platform.system()

backend = os.environ.get('MPLBACKEND', None)
if backend:
    matplotlib.use(backend)
    print(f"Using backend: {backend}")
else:
    try:
        matplotlib.use('TkAgg')
        print("Using TkAgg backend for interactive display")
    except:
        try:
            matplotlib.use('Qt5Agg')
            print("Using Qt5Agg backend for interactive display")
        except:
            matplotlib.use('Agg')
            print("Using Agg backend (non-interactive)")

plt.ion()  

def get_monitor_spikes(monitor):
    try:
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            return monitor.t, monitor.i
        elif hasattr(monitor, '_spike_times') and hasattr(monitor, '_spike_indices'):
            return monitor._spike_times, monitor._spike_indices
        else:
            return np.array([]) * ms, np.array([])
    except:
        return np.array([]) * ms, np.array([])

def get_all_post_neurons(connections_config):
    post_neurons = set()
    for conn_name, conn_info in connections_config.items():
        post_neurons.add(conn_info['post'])
    return sorted(list(post_neurons))

def plot_raster(spike_monitors, sample_size=30, plot_order=None, start_time=0*ms, end_time=1000*ms, display_names=None, save_plot=True):
    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        firing_rates = {}
        
        print(f"\nRaster plot Range: {start_time/ms:.0f}ms - {end_time/ms:.0f}ms")
          
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            display_name = display_names.get(name, name) if display_names else name
            axes[i].scatter(display_t / ms, display_i, s=0.1)
            axes[i].set_title(f'{display_name} Raster Plot (subset of {len(chosen_neurons)} neurons)')
            axes[i].set_ylabel('Neuron index')

            axes[i].set_ylim(min(chosen_neurons) - 1, max(chosen_neurons) + 1)
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        if save_plot:
            filename = 'raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Raster plot saved to '{filename}'")
        
        try:
            print("Raster plot displayed. Plot saved to file for permanent viewing.")
            print("Close the plot window to continue...")
            plt.show(block=True)  
        except Exception as e:
            print(f"Error displaying raster plot: {e}")
        finally:
            pass  

        return firing_rates

    except Exception as e:
        print(f"Raster plot Error: {str(e)}")

def plot_firing_rate_fft(
    spike_monitor, 
    neuron_indices=None, 
    start_time=0*ms, 
    end_time=10000*ms, 
    bin_size=10*ms, 
    show_mean=True, 
    max_freq=100, 
    title='Firing Rate FFT Spectrum'
):
    """
    spike_monitor: Brian2 SpikeMonitor 객체
    neuron_indices: 분석할 뉴런 인덱스 리스트 (None이면 전체)
    start_time, end_time: 분석 구간
    bin_size: firing rate 계산 bin 크기
    show_mean: 여러 뉴런의 평균 스펙트럼도 표시할지
    max_freq: x축 최대 주파수(Hz)
    """
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i
    N = spike_monitor.source.N

    if neuron_indices is None:
        neuron_indices = range(N)
    
    time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
    time_centers = time_bins[:-1] + bin_size/ms/2

    all_spectra = []
    all_freqs = None

    plt.figure(figsize=(10, 6))
    for neuron_idx in neuron_indices:
        neuron_spikes = spike_times[spike_indices == neuron_idx]
        counts, _ = np.histogram(neuron_spikes, bins=time_bins)
        firing_rate = counts / (bin_size/ms/1000.0)  # Hz

        # FFT
        fft_result = np.fft.fft(firing_rate)
        freqs = np.fft.fftfreq(len(firing_rate), d=(bin_size/ms/1000.0))
        pos_mask = freqs >= 0

        # 저장
        if all_freqs is None:
            all_freqs = freqs[pos_mask]
        spectrum = np.abs(fft_result)[pos_mask] / len(firing_rate)
        all_spectra.append(spectrum)

        # 뉴런별 스펙트럼 plot (투명도 낮게)
        plt.plot(freqs[pos_mask], spectrum, alpha=0.2, label=f'Neuron {neuron_idx}' if len(neuron_indices) <= 10 else None)

    all_spectra = np.array(all_spectra)
    if show_mean and len(all_spectra) > 0:
        mean_spectrum = np.mean(all_spectra, axis=0)
        plt.plot(all_freqs, mean_spectrum, 'k-', linewidth=2, label='Mean Spectrum')

        # 가장 큰 파워의 주파수 출력
        peak_idx = np.argmax(mean_spectrum[1:]) + 1  # DC(0Hz) 제외
        peak_freq = all_freqs[peak_idx]
        peak_power = mean_spectrum[peak_idx]
        print(f"가장 강한 주파수: {peak_freq:.2f} Hz (파워: {peak_power:.4f})")
        plt.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, label=f'Peak: {peak_freq:.2f} Hz')

    plt.xlim(0, max_freq)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_membrane_potential(voltage_monitors, plot_order=None):
    if plot_order:
        filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
    else:
        filtered_monitors = voltage_monitors

    n_plots = len(filtered_monitors)
    if n_plots == 0:
        print("No voltage monitors to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]  

    for i, (name, monitor) in enumerate(filtered_monitors.items()):
        if len(monitor.v) == 0:
            print(f"Warning: No voltage data recorded for {name}")
            continue
        axes[i].plot(monitor.t / ms, monitor.v[0] / mV)
        axes[i].set_title(f'{name} Neuron 0 Membrane Potential')
        axes[i].set_ylabel('V (mV)')
        axes[i].set_xlim(0, int(monitor.t[-1] / ms))
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    
    try:
        plt.show(block=False)  
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying membrane potential plot: {e}")
        plt.close()  


def plot_raster_all_neurons_stim_window(spike_monitors, stim_start=200*ms, end_time=1000*ms, plot_order=None):
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        
        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return

        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue

            time_mask = (spike_times >= stim_start) & (spike_times <= end_time)
            display_t = spike_times[time_mask]
            display_i = spike_indices[time_mask]

            axes[i].scatter(display_t / ms, display_i, s=0.5, color='darkblue')
            axes[i].set_title(f'{name} Raster (All neurons, {int(stim_start/ms)}–{int(end_time/ms)} ms)')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(int(stim_start/ms), int(end_time/ms))
            axes[i].set_ylim(-1, monitor.source.N)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        try:
            plt.show(block=False)  
            plt.pause(0.1) 
        except Exception as e:
            print(f"Error displaying raster all neuron stim window: {e}")
            plt.close()

    except Exception as e:
        print(f"Raster all neuron stim window error: {str(e)}")


def analyze_firing_rates_by_stimulus_periods(spike_monitors, stimulus_config, analysis_start_time=2000*ms, plot_order=None, display_names=None):
    
    if not stimulus_config.get('enabled', False):
        print("스티뮬러스가 비활성화되어 있습니다.")
        return

    stim_start = stimulus_config.get('start_time', 10000) * ms
    stim_duration = stimulus_config.get('duration', 1000) * ms
    stim_end = stim_start + stim_duration

    pre_stim_start = analysis_start_time
    pre_stim_end = stim_start
    post_stim_start = stim_end

    if plot_order:
        monitors_to_analyze = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
    else:
        monitors_to_analyze = spike_monitors
    
    print("="*60)
    print(f"Stimulus: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms")
    print(f"Pre-stimulus: {pre_stim_start/ms:.0f}-{pre_stim_end/ms:.0f}ms")
    print(f"During-stimulus: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms") 
    print(f"Post-stimulus: after {post_stim_start/ms:.0f}ms")
    
    for name, monitor in monitors_to_analyze.items():
        display_name = display_names.get(name, name) if display_names else name
        
        spike_times, spike_indices = get_monitor_spikes(monitor)
            
        total_neurons = monitor.source.N
        spike_times_ms = spike_times / ms
        
        # Pre-stimulus 구간 분석
        pre_mask = (spike_times >= pre_stim_start) & (spike_times < pre_stim_end)
        pre_spikes = np.sum(pre_mask)
        pre_duration_sec = (pre_stim_end - pre_stim_start) / second
        pre_rate = pre_spikes / (total_neurons * pre_duration_sec) if pre_duration_sec > 0 else 0
        
        # During-stimulus 구간 분석  
        stim_mask = (spike_times >= stim_start) & (spike_times < stim_end)
        stim_spikes = np.sum(stim_mask)
        stim_duration_sec = stim_duration / second
        stim_rate = stim_spikes / (total_neurons * stim_duration_sec) if stim_duration_sec > 0 else 0
        
        # Post-stimulus 구간 분석 (스티뮬러스 후 같은 길이만큼)
        post_end = post_stim_start + (pre_stim_end - pre_stim_start)  # pre와 같은 길이
        post_mask = (spike_times >= post_stim_start) & (spike_times < post_end)
        post_spikes = np.sum(post_mask)
        post_duration_sec = (post_end - post_stim_start) / second
        post_rate = post_spikes / (total_neurons * post_duration_sec) if post_duration_sec > 0 else 0
        stim_change = ((stim_rate - pre_rate) / pre_rate * 100) if pre_rate > 0 else 0
        post_change = ((post_rate - pre_rate) / pre_rate * 100) if pre_rate > 0 else 0
        
        print(f"\n[{display_name}] (총 {total_neurons}개 뉴런)")
        print(f"  Pre-stimulus  ({pre_stim_start/ms:.0f}-{pre_stim_end/ms:.0f}ms): {pre_rate:.3f} Hz")
        print(f"  During-stimulus ({stim_start/ms:.0f}-{stim_end/ms:.0f}ms): {stim_rate:.3f} Hz")
        print(f"  Post-stimulus ({post_stim_start/ms:.0f}-{post_end/ms:.0f}ms): {post_rate:.3f} Hz")
        print(f"  스티뮬러스 효과: {stim_change:+.1f}%")
        print(f"  회복 상태: {post_change:+.1f}%")
        
    print("\n" + "="*60)


def plot_continuous_firing_rate(spike_monitors, start_time=0*ms, end_time=10000*ms, bin_size=20*ms, 
                               plot_order=None, display_names=None, stimulus_config=None, 
                               smooth_sigma=3, save_plot=True, show_confidence=True, 
                               multi_sample=True, n_samples=10, neurons_per_sample=30,
                               layout_mode='single', plots_per_page=3):
    """    
    Parameters:
    - spike_monitors: 스파이크 모니터 딕셔너리
    - start_time, end_time: 분석 시간 범위
    - bin_size: 각 구간의 크기 (기본값 20ms)
    - plot_order: 플롯 순서
    - display_names: 표시 이름 매핑
    - stimulus_config: 스티뮬러스 설정 (배경 표시용)
    - smooth_sigma: Gaussian smoothing 시그마 값
    - save_plot: 플롯 저장 여부
    - show_confidence: 신뢰구간 표시 여부
    - layout_mode: 레이아웃 모드 ('single': 한 페이지에 모든 그래프, 'multi': 여러 페이지로 분할)
    - plots_per_page: 페이지당 그래프 수 (layout_mode='multi'일 때 사용)
    """
    
    def gaussian_smooth(data, sigma):
        if sigma <= 0:
            return data
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        
        padded_data = np.pad(data, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded_data, kernel, mode='valid')
        return smoothed
    
    def calculate_population_firing_rate(spike_times_ms, spike_indices, time_bins, total_neurons):
        neuron_rates = []
        
        for neuron_id in range(total_neurons):
            neuron_spikes = spike_times_ms[spike_indices == neuron_id]
            neuron_firing_rates = []
            
            for j in range(len(time_bins)-1):
                bin_start = time_bins[j]
                bin_end = time_bins[j+1]
                spikes_in_bin = np.sum((neuron_spikes >= bin_start) & (neuron_spikes < bin_end))
                bin_duration_sec = (bin_end - bin_start) / 1000.0
                rate = spikes_in_bin / bin_duration_sec if bin_duration_sec > 0 else 0
                neuron_firing_rates.append(rate)
            
            if len(neuron_firing_rates) > 0:
                neuron_rates.append(neuron_firing_rates)
        
        if len(neuron_rates) == 0:
            return np.zeros(len(time_bins)-1), np.zeros(len(time_bins)-1), np.zeros(len(time_bins)-1)
        
        neuron_rates = np.array(neuron_rates)
        mean_rates = np.mean(neuron_rates, axis=0)
        std_rates = np.std(neuron_rates, axis=0)
        sem_rates = std_rates / np.sqrt(len(neuron_rates))
        
        return mean_rates, std_rates, sem_rates
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return

        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        
        n_plots = len(spike_monitors)
        
        if layout_mode == 'single':
            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
            if n_plots == 1:
                axes = [axes]
            all_axes = [axes]
            all_figs = [fig]
        else:
            all_axes = []
            all_figs = []
            
            for page_start in range(0, n_plots, plots_per_page):
                page_end = min(page_start + plots_per_page, n_plots)
                page_plots = page_end - page_start
                
                fig, axes = plt.subplots(page_plots, 1, figsize=(10, 2.5 * page_plots), sharex=True)
                if page_plots == 1:
                    axes = [axes]
                all_axes.append(axes)
                all_figs.append(fig)

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        
        print(f"\nAdvanced firing rate analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms, bin size: {bin_size/ms:.0f}ms")
        print(f"Layout mode: {layout_mode}, plots per page: {plots_per_page}")
        
        plot_idx = 0
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            
            if show_confidence and total_neurons > 10: 
                mean_rates, std_rates, sem_rates = calculate_population_firing_rate(
                    spike_times_ms, spike_indices, time_bins, total_neurons)
                firing_rates_smooth = gaussian_smooth(mean_rates, smooth_sigma)
                confidence_upper = gaussian_smooth(mean_rates + sem_rates, smooth_sigma)
                confidence_lower = gaussian_smooth(mean_rates - sem_rates, smooth_sigma)
            else:
                firing_rates = []
                for j in range(len(time_bins)-1):
                    bin_start = time_bins[j]
                    bin_end = time_bins[j+1]
                    spikes_in_bin = np.sum((spike_times_ms >= bin_start) & (spike_times_ms < bin_end))
                    bin_duration_sec = (bin_end - bin_start) / 1000.0
                    rate = spikes_in_bin / (total_neurons * bin_duration_sec) if bin_duration_sec > 0 else 0
                    firing_rates.append(rate)
                
                firing_rates = np.array(firing_rates)
                firing_rates_smooth = gaussian_smooth(firing_rates, smooth_sigma)
                confidence_upper = confidence_lower = None
            
            display_name = display_names.get(name, name) if display_names else name
            color = colors[i % len(colors)]
            
            if layout_mode == 'single':
                current_axes = axes[plot_idx]
            else:
                page_idx = plot_idx // plots_per_page
                subplot_idx = plot_idx % plots_per_page
                current_axes = all_axes[page_idx][subplot_idx]
            
            current_axes.plot(time_centers, firing_rates_smooth, linewidth=2.5, color=color, 
                            label=display_name, alpha=0.9)
            
            if show_confidence and confidence_upper is not None and confidence_lower is not None:
                current_axes.fill_between(time_centers, confidence_lower, confidence_upper, 
                                       color=color, alpha=0.3, linewidth=0)
            
            if stimulus_config and stimulus_config.get('enabled', False):
                stim_start = stimulus_config.get('start_time', 0)
                stim_duration = stimulus_config.get('duration', 0)
                stim_end = stim_start + stim_duration
                
                y_max = np.max(firing_rates_smooth) * 1.1
                current_axes.hlines(y_max, stim_start, stim_end, colors='red', linewidth=4, alpha=0.8)
                current_axes.text((stim_start + stim_end)/2, y_max * 1.05, 'Stimulus', 
                               ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
                
                current_axes.axvline(stim_start, color='red', linestyle=':', alpha=0.7, linewidth=1)
                current_axes.axvline(stim_end, color='red', linestyle=':', alpha=0.7, linewidth=1)
            
            current_axes.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
            current_axes.set_title(f'{display_name}', fontsize=14, fontweight='bold', pad=15)
            
            current_axes.grid(True, alpha=0.3, linewidth=0.5)
            current_axes.spines['top'].set_visible(False)
            current_axes.spines['right'].set_visible(False)
            current_axes.spines['left'].set_linewidth(1.5)
            current_axes.spines['bottom'].set_linewidth(1.5)
            
            if np.max(firing_rates_smooth) > 0:
                current_axes.set_ylim(0, np.max(firing_rates_smooth) * 1.15)
            
            print(f"{display_name}: Peak rate = {np.max(firing_rates_smooth):.2f} Hz, Mean rate = {np.mean(firing_rates_smooth):.2f} Hz")
            
            plot_idx += 1

        if layout_mode == 'single':
            axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
            axes[-1].set_xlim(start_time/ms, end_time/ms)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
        
            try:
                print("Advanced firing rate plot displayed. Close the plot window to continue...")
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying firing rate plot: {e}")
        else:
            for page_idx, (fig, axes_page) in enumerate(zip(all_figs, all_axes)):
                axes_page[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
                axes_page[-1].set_xlim(start_time/ms, end_time/ms)
                
                for ax in axes_page:
                    ax.set_xlim(start_time/ms, end_time/ms)
                
                plt.figure(fig.number)
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3)
                
            try:
                print("Advanced firing rate plots displayed. Close the plot windows to continue...")
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying firing rate plots: {e}")

    except Exception as e:
        print(f"Firing rate plot Error: {str(e)}")


def plot_individual_neuron_firing_rates(spike_monitors, start_time=0*ms, end_time=10000*ms, bin_size=20*ms,
                                       plot_order=None, display_names=None, stimulus_config=None,
                                       smooth_sigma=3, save_plot=True, neurons_per_group=10):

    def gaussian_smooth(data, sigma):
        """Gaussian smoothing"""
        if sigma <= 0:
            return data
        # Create Gaussian kernel
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        
        padded_data = np.pad(data, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded_data, kernel, mode='valid')
        return smoothed
    
    def calculate_individual_neuron_firing_rate(spike_times_ms, spike_indices, neuron_id, time_bins):
        neuron_spikes = spike_times_ms[spike_indices == neuron_id]
        firing_rates = []
        
        for j in range(len(time_bins)-1):
            bin_start = time_bins[j]
            bin_end = time_bins[j+1]
            spikes_in_bin = np.sum((neuron_spikes >= bin_start) & (neuron_spikes < bin_end))
            bin_duration_sec = (bin_end - bin_start) / 1000.0
            rate = spikes_in_bin / bin_duration_sec if bin_duration_sec > 0 else 0
            firing_rates.append(rate)
        
        return np.array(firing_rates)
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return

        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        
        print(f"\nIndividual neuron firing rate analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms, bin size: {bin_size/ms:.0f}ms")
        
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            
            selected_neurons = min(neurons_per_group, total_neurons)
            if total_neurons > selected_neurons:
                neuron_indices = list(range(selected_neurons))
            else:
                neuron_indices = list(range(total_neurons))
            
            individual_rates = []
            for neuron_idx in neuron_indices:
                rate_data = calculate_individual_neuron_firing_rate(
                    spike_times_ms, spike_indices, neuron_idx, time_bins)
                if smooth_sigma > 0:
                    rate_data = gaussian_smooth(rate_data, smooth_sigma)
                individual_rates.append(rate_data)
            
            if individual_rates:
                mean_rate = np.mean(individual_rates, axis=0)
                if smooth_sigma > 0:
                    mean_rate = gaussian_smooth(mean_rate, smooth_sigma)
            else:
                mean_rate = np.zeros(len(time_centers))
            
            fig = plt.figure(figsize=(20, 12))
            
            for i, neuron_idx in enumerate(neuron_indices):
                ax = plt.subplot(2, 6, i + 1)  
                
                if i < len(individual_rates):
                    ax.plot(time_centers, individual_rates[i], 
                           color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'Neuron {neuron_idx}', fontsize=10, fontweight='bold')
                ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                if stimulus_config and stimulus_config.get('enabled', False):
                    stim_start = stimulus_config.get('start_time', 0)
                    stim_duration = stimulus_config.get('duration', 0)
                    stim_end = stim_start + stim_duration
                    ax.axvspan(stim_start, stim_end, alpha=0.2, color='red')
                
                if i >= 5:
                    ax.set_xlabel('Time (ms)', fontsize=9)
            
            ax_mean = plt.subplot(2, 6, (6, 12)) 
            
            for i, rate_data in enumerate(individual_rates):
                ax_mean.plot(time_centers, rate_data, 
                           color=colors[i % len(colors)], linewidth=1, alpha=0.3)
            
            ax_mean.plot(time_centers, mean_rate, 'black', linewidth=3, 
                        label='Mean', alpha=0.9)
            
            ax_mean.set_title(f'{display_name} - Mean Firing Rate', fontsize=14, fontweight='bold')
            ax_mean.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
            ax_mean.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
            ax_mean.grid(True, alpha=0.3)
            ax_mean.legend()
            
            plt.tight_layout()
            try:
                print(f"Individual firing rates for {display_name} displayed")
                if os.environ.get('MPLBACKEND') == 'Agg':
                    plt.close() 
                else:
                    plt.show(block=False)
                    plt.pause(0.1)
            except Exception as e:
                print(f"Error displaying individual firing rates: {e}")

    except Exception as e:
        print(f"Individual firing rate plot Error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_place_cell_theta_analysis(spike_monitors, start_time=0*ms, end_time=10000*ms,
                                  plot_order=None, display_names=None, stimulus_config=None,
                                  save_plot=True, place_field_center=0.0, spatial_range=3.0):

    def generate_theta_phase(time_ms, frequency=8.0):
        theta_phase = 2 * np.pi * frequency * time_ms / 1000.0
        return theta_phase % (2 * np.pi)
    
    def calculate_spatial_position(time_ms, velocity=0.1):
        spatial_pos = velocity * (time_ms - time_ms[0]) - spatial_range/2
        return spatial_pos
    
    def create_place_field_response(spatial_pos, center=0.0, width=1.0):
        return np.exp(-0.5 * ((spatial_pos - center) / width) ** 2)
    
    def calculate_phase_precession(spatial_pos, base_phase=0.0, precession_rate=0.5):
        phase_shift = precession_rate * spatial_pos
        return (base_phase + phase_shift) % (2 * np.pi)
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        
        print(f"\nPlace cell theta analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms")
        
        for group_idx, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            
            time_mask = (spike_times_ms >= start_time/ms) & (spike_times_ms <= end_time/ms)
            spike_times_window = spike_times_ms[time_mask]
            spike_indices_window = spike_indices[time_mask]
            
            if len(spike_times_window) == 0:
                print(f"No spikes in analysis window for {name}")
                continue
            
            time_vector = np.arange(start_time/ms, end_time/ms, 1.0)  
            theta_phase_vector = generate_theta_phase(time_vector)
            spatial_pos_vector = calculate_spatial_position(time_vector)
            
            place_response = create_place_field_response(spatial_pos_vector, place_field_center)
            
            spike_spatial_pos = []
            spike_theta_phase = []
            spike_place_response = []
            
            for spike_time in spike_times_window:
                time_idx = np.argmin(np.abs(time_vector - spike_time))
                if time_idx < len(spatial_pos_vector):
                    spatial_pos = spatial_pos_vector[time_idx]
                    theta_phase = theta_phase_vector[time_idx]
                    place_resp = place_response[time_idx]
                    
                    if abs(spatial_pos - place_field_center) <= spatial_range:
                        spike_spatial_pos.append(spatial_pos)
                        spike_theta_phase.append(theta_phase)
                        spike_phase_precession = calculate_phase_precession(spatial_pos)
                        spike_theta_phase.append(spike_phase_precession)
                        spike_place_response.append(place_resp)
            
            if len(spike_spatial_pos) == 0:
                print(f"No spikes in place field range for {name}")
                continue

            spatial_bins = 30
            phase_bins = 20
            
            spatial_edges = np.linspace(place_field_center - spatial_range/2, 
                                      place_field_center + spatial_range/2, spatial_bins + 1)
            phase_edges = np.linspace(0, 2*np.pi, phase_bins + 1)
            
            spike_count_matrix, _, _ = np.histogram2d(spike_spatial_pos, spike_theta_phase, 
                                                     bins=[spatial_edges, phase_edges])
            
            spatial_counts = np.sum(spike_count_matrix, axis=1)
            normalized_matrix = spike_count_matrix.copy()
            for i in range(spatial_bins):
                if spatial_counts[i] > 0:
                    normalized_matrix[i, :] /= spatial_counts[i]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{name} - Place Cell Theta Analysis', fontsize=16, fontweight='bold')
            
            im1 = ax1.imshow(spike_count_matrix.T, cmap='plasma', aspect='auto', 
                            extent=[spatial_edges[0], spatial_edges[-1], 0, 2*np.pi],
                            origin='lower', interpolation='bilinear')
            ax1.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Theta phase (rad)', fontsize=12, fontweight='bold')
            ax1.set_title('B: Average Spike Count', fontsize=14, fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('<spike count>', rotation=270, labelpad=15)
            
            # C: Mean Firing Rate
            mean_firing_rate = np.mean(spike_count_matrix, axis=1)
            spatial_centers = (spatial_edges[:-1] + spatial_edges[1:]) / 2
            ax2.plot(spatial_centers, mean_firing_rate, 'b-', linewidth=2, label='Mean Firing Rate')

            # 가장 큰 파워의 주파수 출력
            peak_idx = np.argmax(mean_firing_rate[1:]) + 1  # DC(0Hz) 제외
            peak_freq = spatial_centers[peak_idx]
            peak_power = mean_firing_rate[peak_idx]
            print(f"가장 강한 주파수: {peak_freq:.2f} Hz (파워: {peak_power:.4f})")
            ax2.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, label=f'Peak: {peak_freq:.2f} Hz')

            ax2.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
            ax2.set_title('C: Mean Firing Rate', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # D: Normalized Spike Count
            im2 = ax3.imshow(normalized_matrix.T, cmap='plasma', aspect='auto',
                            extent=[spatial_edges[0], spatial_edges[-1], 0, 2*np.pi],
                            origin='lower', interpolation='bilinear')
            ax3.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Theta phase (rad)', fontsize=12, fontweight='bold')
            ax3.set_title('D: Normalized Spike Count', fontsize=14, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax3)
            cbar2.set_label('<norm. spike count>', rotation=270, labelpad=15)
            
            # E: Spike-Phase Coupling
            coupling_strength = []
            for i in range(spatial_bins):
                if spatial_counts[i] > 0:
                    phase_dist = spike_theta_phase[np.digitize(spike_spatial_pos, spatial_edges) == i+1]
                    if len(phase_dist) > 1:
                        coupling = 1.0 / (np.std(phase_dist) + 1e-6)  
                    else:
                        coupling = 0.0
                else:
                    coupling = 0.0
                coupling_strength.append(coupling)
            
            ax4.plot(spatial_centers, coupling_strength, 'r-', linewidth=2, label='Raw')
            from scipy.ndimage import gaussian_filter1d
            smooth_coupling = gaussian_filter1d(coupling_strength, sigma=1.0)
            ax4.plot(spatial_centers, smooth_coupling, 'r--', linewidth=2, label='Smoothed')
            ax4.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Spike-Phase Coupling', fontsize=12, fontweight='bold')
            ax4.set_title('E: Spike-Phase Coupling', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            total_spikes = len(spike_spatial_pos)
            max_rate = np.max(mean_firing_rate)
            avg_coupling = np.mean(coupling_strength)
            
            stats_text = (f'Total spikes: {total_spikes}\n'
                         f'Max firing rate: {max_rate:.2f}\n'
                         f'Avg coupling: {avg_coupling:.3f}')
            
            fig.text(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            try:
                print(f"Place cell theta analysis for {name} displayed.")
                if os.environ.get('MPLBACKEND') == 'Agg':
                    plt.close() 
                else:
                    plt.show(block=False)  
                    plt.pause(0.1) 
                    print(f"  -> Displayed interactively")
            except Exception as e:
                print(f"Error displaying place cell theta analysis: {e}")

    except Exception as e:
        print(f"Place cell theta analysis error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_place_cell_theta_analysis_custom(
    spike_spatial_pos, spike_theta_phase, spatial_bins=40, phase_bins=40, spatial_range=3.0, save_path=None
):

    spatial_edges = np.linspace(-spatial_range, spatial_range, spatial_bins+1)
    phase_edges = np.linspace(0, 4*np.pi, phase_bins+1)
    
    # 2. B: 공간-위상 히트맵 (spike count)
    spike_count_matrix, _, _ = np.histogram2d(
        spike_spatial_pos, spike_theta_phase, bins=[spatial_edges, phase_edges]
    )
    spike_count_matrix = spike_count_matrix.T  # (phase, space)
    
    # 3. C: mean firing rate (공간별)
    mean_firing_rate = np.sum(spike_count_matrix, axis=0)
    spatial_centers = (spatial_edges[:-1] + spatial_edges[1:]) / 2
    
    # 4. D: normalized spike count (공간별 정규화)
    norm_matrix = spike_count_matrix / (np.max(spike_count_matrix, axis=0, keepdims=True) + 1e-6)
    
    # 5. E: spike-phase coupling (공간별 위상 결합 강도)
    coupling_strength = []
    for i in range(spatial_bins):
        mask = (spike_spatial_pos >= spatial_edges[i]) & (spike_spatial_pos < spatial_edges[i+1])
        phases = spike_theta_phase[mask]
        if len(phases) > 1:
            coupling = 1.0 / (np.std(phases) + 1e-6)
        else:
            coupling = 0.0
        coupling_strength.append(coupling)
    coupling_strength = np.array(coupling_strength)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1.1], height_ratios=[2,1], wspace=0.3, hspace=0.3)
    
    # B
    axB = fig.add_subplot(gs[0,0])
    imB = axB.imshow(
        spike_count_matrix, aspect='auto', origin='lower',
        extent=[spatial_edges[0], spatial_edges[-1], 0, 4*np.pi],
        cmap='jet'
    )
    axB.set_ylabel('Theta phase (rad)', fontsize=12, fontweight='bold')
    axB.set_title('B', loc='left', fontsize=18, fontweight='bold')
    plt.colorbar(imB, ax=axB, label='<Spike count>')
    axB.set_xticks([])
    axB.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    axB.set_yticklabels(['0', 'π', '2π', '3π', '4π'])
    
    # D
    axD = fig.add_subplot(gs[0,1])
    imD = axD.imshow(
        norm_matrix, aspect='auto', origin='lower',
        extent=[spatial_edges[0], spatial_edges[-1], 0, 4*np.pi],
        cmap='jet'
    )
    axD.set_title('D', loc='left', fontsize=18, fontweight='bold')
    plt.colorbar(imD, ax=axD, label='<norm. spike count>')
    axD.set_xticks([])
    axD.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    axD.set_yticklabels(['0', 'π', '2π', '3π', '4π'])
    
    # C
    axC = fig.add_subplot(gs[1,0])
    axC.plot(spatial_centers, mean_firing_rate, 'k-', linewidth=2)
    axC.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
    axC.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
    axC.set_title('C', loc='left', fontsize=18, fontweight='bold')
    axC.grid(True, alpha=0.3)
    
    # E
    axE = fig.add_subplot(gs[1,1])
    axE.plot(spatial_centers, coupling_strength, 'k-', linewidth=2)
    axE.set_xlabel('Relative distance to place field center', fontsize=12, fontweight='bold')
    axE.set_ylabel('Spike-Phase Coupling', fontsize=12, fontweight='bold')
    axE.set_title('E', loc='left', fontsize=18, fontweight='bold')
    axE.grid(True, alpha=0.3)
    
    fig.add_subplot(gs[0,2]).axis('off')
    fig.add_subplot(gs[1,2]).axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_spike_propagation_analysis(spike_monitors, connections_config, 
                                   target_neuron='FSN', analysis_window=(2000*ms, 3000*ms),
                                   propagation_delay=20*ms, save_plot=True):
    
    def find_target_spikes_with_preceding_inputs(target_spikes, input_spikes_dict, 
                                               propagation_delay=20*ms):
        propagation_events = []
        
        for target_time in target_spikes:
            event = {
                'target_time': target_time,
                'inputs': {}
            }
            
            for input_name, input_times in input_spikes_dict.items():

                preceding_spikes = input_times[
                    (input_times >= target_time - propagation_delay) & 
                    (input_times < target_time)
                ]
                
                if len(preceding_spikes) > 0:
                    event['inputs'][input_name] = preceding_spikes
            
            if len(event['inputs']) > 0:
                propagation_events.append(event)
        
        return propagation_events
    
    try:
        start_time, end_time = analysis_window
        
        incoming_connections = {}
        for conn_name, conn_info in connections_config.items():
            if conn_info['post'] == target_neuron:
                incoming_connections[conn_name] = conn_info
        
        if target_neuron not in spike_monitors:
            print(f"오류: {target_neuron} 모니터를 찾을 수 없습니다.")
            return
        
        target_times, target_indices = get_monitor_spikes(spike_monitors[target_neuron])
        target_times_in_window = target_times[
            (target_times >= start_time) & (target_times <= end_time)
        ]
        

        input_spikes_dict = {}
        input_neuron_names = []
        
        for conn_name, conn_info in incoming_connections.items():
            pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
            if pre_neuron in spike_monitors:
                input_neuron_names.append(pre_neuron)
                pre_times, _ = get_monitor_spikes(spike_monitors[pre_neuron])
                pre_times_in_window = pre_times[
                    (pre_times >= start_time - propagation_delay) & 
                    (pre_times <= end_time)
                ]
                input_spikes_dict[pre_neuron] = pre_times_in_window
        
        propagation_events = find_target_spikes_with_preceding_inputs(
            target_times_in_window, input_spikes_dict, propagation_delay)
        
        print(f"총 {len(target_times_in_window)}개의 {target_neuron} 스파이크 중 "
              f"{len(propagation_events)}개가 선행 입력과 연관됨")
        
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(3, 2, 1)
        
        y_offset = 0
        neuron_positions = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(input_neuron_names) + 1))
        
        for i, neuron_name in enumerate(input_neuron_names):
            if neuron_name in input_spikes_dict:
                spike_times = input_spikes_dict[neuron_name] / ms
                y_pos = np.full_like(spike_times, y_offset)
                ax1.scatter(spike_times, y_pos, s=15, c=[colors[i]], 
                           alpha=0.7, label=neuron_name)
                neuron_positions[neuron_name] = y_offset
                y_offset += 1
        
        target_spike_times = target_times_in_window / ms
        y_pos = np.full_like(target_spike_times, y_offset)
        ax1.scatter(target_spike_times, y_pos, s=30, c='red', 
                   marker='v', alpha=0.9, label=f'{target_neuron} (target)')
        neuron_positions[target_neuron] = y_offset
        
        for event in propagation_events[:20]:  
            target_time = event['target_time'] / ms
            target_y = neuron_positions[target_neuron]
            
            for input_name, input_times in event['inputs'].items():
                input_y = neuron_positions[input_name]
                for input_time in input_times / ms:
                    ax1.plot([input_time, target_time], [input_y, target_y], 
                           'k-', alpha=0.3, linewidth=0.5)
        
        ax1.set_xlim(start_time/ms, end_time/ms)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Neuron')
        ax1.set_title(f'{target_neuron} spike propagation pattern', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        
        all_delays = []
        input_contributions = {name: [] for name in input_neuron_names}
        
        for event in propagation_events:
            target_time = event['target_time']
            for input_name, input_times in event['inputs'].items():
                delays = (target_time - input_times) / ms
                all_delays.extend(delays)
                input_contributions[input_name].extend(delays)
        
        if len(all_delays) > 0:
            ax2.hist(all_delays, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(all_delays), color='red', linestyle='--', 
                       label=f'평균: {np.mean(all_delays):.1f}ms')
            ax2.set_xlabel('전파 지연시간 (ms)')
            ax2.set_ylabel('빈도')
            ax2.set_title('스파이크 전파 지연시간 분포', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 2, 3)
        
        contribution_counts = []
        contribution_names = []
        
        for input_name in input_neuron_names:
            count = len(input_contributions[input_name])
            contribution_counts.append(count)
            contribution_names.append(input_name)
        
        if len(contribution_counts) > 0:
            bars = ax3.bar(contribution_names, contribution_counts, 
                          color=colors[:len(contribution_counts)], alpha=0.8)
            ax3.set_xlabel('입력 뉴런')
            ax3.set_ylabel('기여한 스파이크 수')
            ax3.set_title(f'{target_neuron} 스파이크에 대한 입력별 기여도', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, contribution_counts):
                if count > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax4 = plt.subplot(3, 2, 4)
        
        if len(propagation_events) > 0:
            event_times = [event['target_time']/ms for event in propagation_events]
            
            time_bins = np.linspace(start_time/ms, end_time/ms, 50)
            hist, bin_edges = np.histogram(event_times, bins=time_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            ax4.plot(bin_centers, hist, linewidth=2, color='purple')
            ax4.fill_between(bin_centers, hist, alpha=0.3, color='purple')
            ax4.set_xlabel('시간 (ms)')
            ax4.set_ylabel('전파 이벤트 수')
            ax4.set_title('시간별 스파이크 전파 이벤트 밀도', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(3, 2, 5)
        
        connection_strengths = []
        contributions = []
        neuron_labels = []
        
        for conn_name, conn_info in incoming_connections.items():
            pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
            if pre_neuron in input_contributions:
                strength = conn_info['p'] * conn_info['weight']
                contribution = len(input_contributions[pre_neuron])
                
                connection_strengths.append(strength)
                contributions.append(contribution)
                neuron_labels.append(pre_neuron)
        
        if len(connection_strengths) > 0:
            scatter = ax5.scatter(connection_strengths, contributions, 
                                s=100, alpha=0.7, c=range(len(neuron_labels)), 
                                cmap='viridis')
            
            for i, label in enumerate(neuron_labels):
                ax5.annotate(label, (connection_strengths[i], contributions[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax5.set_xlabel('연결 강도 (p × w)')
            ax5.set_ylabel('스파이크 기여도')
            ax5.set_title('연결 강도 vs 실제 기여도', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            if len(connection_strengths) > 1:
                correlation = np.corrcoef(connection_strengths, contributions)[0, 1]
                ax5.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                        transform=ax5.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = f"{target_neuron} 스파이크 전파 통계\n\n"
        stats_text += f"분석 구간: {start_time/ms:.0f}-{end_time/ms:.0f}ms\n"
        stats_text += f"총 {target_neuron} 스파이크: {len(target_times_in_window)}\n"
        stats_text += f"전파 연관 스파이크: {len(propagation_events)}\n"
        stats_text += f"전파 연관 비율: {len(propagation_events)/len(target_times_in_window)*100:.1f}%\n\n"
        
        if len(all_delays) > 0:
            stats_text += f"평균 전파 지연시간: {np.mean(all_delays):.2f}ms\n"
            stats_text += f"표준편차: {np.std(all_delays):.2f}ms\n"
            stats_text += f"최소 지연시간: {np.min(all_delays):.2f}ms\n"
            stats_text += f"최대 지연시간: {np.max(all_delays):.2f}ms\n\n"
        
        stats_text += "입력별 기여도:\n"
        for input_name in input_neuron_names:
            count = len(input_contributions[input_name])
            percentage = count / len(propagation_events) * 100 if len(propagation_events) > 0 else 0
            stats_text += f"  {input_name}: {count} ({percentage:.1f}%)\n"
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"스파이크 전파 분석 표시 오류: {e}")
            
    except Exception as e:
        print(f"스파이크 전파 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_stimulus_zoom_raster(spike_monitors, stimulus_periods, sample_size=6, 
                             zoom_margin=50*ms, plot_order=None, display_names=None, save_plot=True):
    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        for period_idx, (stim_start, stim_end) in enumerate(stimulus_periods):
            zoom_start = max(0*ms, stim_start - zoom_margin)
            zoom_end = stim_end + zoom_margin
            n_plots = len(spike_monitors)
            fig, axes = plt.subplots(n_plots, 1, figsize=(18, 4 * n_plots), sharex=True)
            if n_plots == 1:
                axes = [axes]
            print(f"\nStimulus Period {period_idx + 1} Zoom: {zoom_start/ms:.0f}ms - {zoom_end/ms:.0f}ms")
            for i, (name, monitor) in enumerate(spike_monitors.items()):
                spike_times, spike_indices = get_monitor_spikes(monitor)
                if len(spike_indices) == 0:
                    print(f"No spikes recorded for {name}")
                    continue
                total_neurons = monitor.source.N
                chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)
                chosen_neurons = sorted(chosen_neurons)
                time_mask = (spike_times >= zoom_start) & (spike_times <= zoom_end)
                neuron_mask = np.isin(spike_indices, chosen_neurons)
                combined_mask = time_mask & neuron_mask
                display_t = spike_times[combined_mask]
                display_i = spike_indices[combined_mask]
                display_name = display_names.get(name, name) if display_names else name
                axes[i].scatter(display_t / ms, display_i, s=12.0, alpha=0.95, edgecolors='black', linewidth=0.8)
                axes[i].set_title(f'{display_name} - Stimulus Period {period_idx + 1} Zoom', fontsize=14, pad=15)
                
                axes[i].set_yticks([])
                axes[i].set_ylabel("")
                # axes[i].set_ylim(-0.5, len(chosen_neurons) - 0.5)  
                # for j in range(len(chosen_neurons)):
                #     axes[i].axhline(y=j-0.5, color='gray', alpha=0.3, linewidth=0.5)
                axes[i].set_xlim(int(zoom_start/ms), int(zoom_end/ms))
                axes[i].axvspan(stim_start/ms, stim_end/ms, alpha=0.25, color='red', label='Stimulus')
                axes[i].grid(True, alpha=0.15, axis='x')
                print(f"  {display_name}: {len(display_t)} spikes in zoom window ({sample_size} neurons)")
            axes[-1].set_xlabel('Time (ms)', fontsize=12)
            plt.tight_layout(pad=3.0)
            try:
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying stimulus zoom plot: {e}")
            finally:
                plt.close()
    except Exception as e:
        print(f"Stimulus zoom raster plot Error: {str(e)}")

def plot_improved_overall_raster(spike_monitors, sample_size=12, plot_order=None, 
                                start_time=0*ms, end_time=1000*ms, display_names=None, 
                                stimulus_periods=None, save_plot=True):

    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(18, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        print(f"\nImproved Raster plot Range: {start_time/ms:.0f}ms - {end_time/ms:.0f}ms")
          
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue
            
            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)
            chosen_neurons = sorted(chosen_neurons)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(name, name) if display_names else name
            
            axes[i].scatter(display_t / ms, remapped_i, s=5.0, alpha=0.9, edgecolors='black', linewidth=0.3)
            axes[i].set_title(f'{display_name} Raster Plot', fontsize=14, pad=15)
            axes[i].set_ylabel('Neuron Index', fontsize=12)
            
            axes[i].text(0.98, 0.95, f'n={len(chosen_neurons)}', transform=axes[i].transAxes,
                        horizontalalignment='right', verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

            if len(chosen_neurons) > 0:
                axes[i].set_ylim(-0.5, len(chosen_neurons) - 0.5)
                if len(chosen_neurons) <= 15: 
                    axes[i].set_yticks(range(len(chosen_neurons)))
                    axes[i].set_yticklabels([f'N{n}' for n in chosen_neurons])
                else: 
                    tick_indices = range(0, len(chosen_neurons), max(1, len(chosen_neurons)//5))
                    axes[i].set_yticks(tick_indices)
                    axes[i].set_yticklabels([f'N{chosen_neurons[j]}' for j in tick_indices])
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))
            
            for j in range(0, len(chosen_neurons), max(1, len(chosen_neurons)//10)):
                axes[i].axhline(y=j-0.5, color='gray', alpha=0.2, linewidth=0.3)
            
            if stimulus_periods:
                for period_idx, (stim_start, stim_end) in enumerate(stimulus_periods):
                    if stim_start >= start_time and stim_end <= end_time:
                        axes[i].axvspan(stim_start/ms, stim_end/ms, alpha=0.2, color='red')
                
                axes[i].grid(True, alpha=0.08, axis='x')
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")

        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        plt.tight_layout(pad=3.0)
        
        if save_plot:
            filename = 'improved_raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Improved raster plot saved to '{filename}'")
        
        try:
            if os.environ.get('MPLBACKEND') == 'Agg':
                plt.close()  
            else:
                plt.show(block=False) 
                plt.pause(0.1) 
        except Exception as e:
            print(f"Error displaying improved raster plot: {e}")
        finally:
            pass

    except Exception as e:
        print(f"Improved raster plot Error: {str(e)}")

def plot_circuit_flow_heatmap(spike_monitors, connections_config, 
                             start_time=0*ms, end_time=2000*ms, bin_size=10*ms,
                             plot_order=None, display_names=None, save_plot=True):
    
    def calculate_instantaneous_activity(spike_times_ms, total_neurons, time_bins):
        activity = []
        for i in range(len(time_bins)-1):
            bin_start = time_bins[i]
            bin_end = time_bins[i+1]
            spikes_in_bin = np.sum((spike_times_ms >= bin_start) & (spike_times_ms < bin_end))
            normalized_activity = spikes_in_bin / (total_neurons * (bin_end - bin_start) / 1000.0)
            activity.append(normalized_activity)
        return np.array(activity)
    
    def detect_propagation_waves(activity_matrix, time_centers, neuron_order):

        waves = []
        
        for t_idx in range(len(time_centers)):
            activities = activity_matrix[:, t_idx]
            

            active_groups = np.where(activities > np.percentile(activities, 75))[0]
            
            if len(active_groups) > 1:
                is_wave = True
                for i in range(len(active_groups)-1):
                    if active_groups[i] >= active_groups[i+1]: 
                        is_wave = False
                        break
                
                if is_wave:
                    waves.append({
                        'time': time_centers[t_idx],
                        'groups': active_groups,
                        'strength': np.mean(activities[active_groups])
                    })
        
        return waves
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        else:
            plot_order = list(spike_monitors.keys())
        
        if not spike_monitors:
            print("No valid neuron groups for circuit flow analysis.")
            return
        
        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        
        neuron_names = list(spike_monitors.keys())
        activity_matrix = np.zeros((len(neuron_names), len(time_centers)))
        
        max_activities = []
        
        for i, name in enumerate(neuron_names):
            spike_times, spike_indices = get_monitor_spikes(spike_monitors[name])
            if len(spike_times) > 0:
                spike_times_ms = spike_times / ms
                total_neurons = spike_monitors[name].source.N
                activity = calculate_instantaneous_activity(spike_times_ms, total_neurons, time_bins)
                activity_matrix[i, :] = activity
                max_activities.append(np.max(activity))
            else:
                max_activities.append(0)
        
        waves = detect_propagation_waves(activity_matrix, time_centers, neuron_names)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        im = ax.imshow(activity_matrix, cmap='plasma', aspect='auto', 
                      interpolation='bilinear', alpha=0.9)
        
        time_start = time_centers[0]
        time_end = time_centers[-1]
        
        tick_interval = 100
        tick_times = np.arange(
            np.ceil(time_start / tick_interval) * tick_interval, 
            time_end + tick_interval,
            tick_interval
        )
        
        time_tick_indices = []
        tick_labels = []
        for tick_time in tick_times:
            if tick_time <= time_end:
                closest_idx = np.argmin(np.abs(time_centers - tick_time))
                time_tick_indices.append(closest_idx)
                tick_labels.append(f'{tick_time:.0f}')
        
        ax.set_xticks(time_tick_indices)
        ax.set_xticklabels(tick_labels)
    
        ax.set_yticks(range(len(neuron_names)))
    
        display_labels = []
        for name in neuron_names:
            if display_names and name in display_names:
                display_labels.append(display_names[name])
            else:
                display_labels.append(name)
        ax.set_yticklabels(display_labels, fontsize=13, fontweight='bold')
        
        ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Neural Circuit', fontsize=16, fontweight='bold')
        ax.set_title('Circuit Flow Heat Map - Neural Activity Propagation', 
                    fontsize=18, fontweight='bold', pad=25)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Neural Activity (Hz)', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Time resolution: {bin_size/ms:.0f} ms")
        print(f"Detected propagation waves: {len(waves)}")
        
        if waves:
            print(f"Average wave strength: {np.mean([w['strength'] for w in waves]):.3f}")
            wave_intervals = np.diff([w['time'] for w in waves])
            if len(wave_intervals) > 0:
                print(f"Average wave interval: {np.mean(wave_intervals):.1f} ms")
        
        for i, name in enumerate(neuron_names):
            display_name = display_names.get(name, name) if display_names else name
            max_activity = max_activities[i]
            avg_activity = np.mean(activity_matrix[i, :])
            print(f"{display_name}: Max={max_activity:.2f} Hz, Avg={avg_activity:.2f} Hz")
        
        try:
            print("\nCircuit flow heat map displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying circuit flow heat map: {e}")
            
    except Exception as e:
        print(f"Circuit flow heat map error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_spike_burst_cascade(spike_monitors, connections_config, start_time=0*ms, end_time=2000*ms, 
                            burst_threshold=0.75, cascade_window=50*ms,
                            plot_order=None, display_names=None, save_plot=True):

    
    def detect_bursts(spike_times_ms, total_neurons, bin_size=20):
        if len(spike_times_ms) == 0:
            return [], []
        
        bins = np.arange(start_time/ms, end_time/ms + bin_size, bin_size)
        
        hist, bin_edges = np.histogram(spike_times_ms, bins=bins)
        firing_rates = hist / (total_neurons * bin_size / 1000.0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        non_zero_rates = firing_rates[firing_rates > 0]
        if len(non_zero_rates) > 0:
            threshold = np.percentile(non_zero_rates, burst_threshold * 100)
        else:
            threshold = 0
        
        burst_indices = np.where(firing_rates > threshold)[0]
        burst_times = bin_centers[burst_indices]
        burst_strengths = firing_rates[burst_indices]
        
        return burst_times, burst_strengths
    
    def extract_connection_map(connections_config):
        connection_map = {}
        for conn_name, conn_info in connections_config.items():
            pre = conn_info['pre']
            post = conn_info['post']
            
            if pre not in connection_map:
                connection_map[pre] = []
            connection_map[pre].append(post)
        
        return connection_map
    
    def find_connection_based_cascades(burst_data, connection_map, cascade_window_ms):
        cascades = []
        
        for start_group in burst_data.keys():
            if start_group not in burst_data or len(burst_data[start_group]['times']) == 0:
                continue
                
            for burst_time in burst_data[start_group]['times']:
                cascade = {
                    'start_time': burst_time,
                    'start_group': start_group,
                    'propagation_chain': [(start_group, burst_time)],
                    'connections_used': []
                }

                current_groups = [start_group]
                visited_groups = {start_group}
                
                while current_groups:
                    next_groups = []
                    
                    for current_group in current_groups:
                        if current_group in connection_map:
                            targets = connection_map[current_group]
                            
                            for target_group in targets:
                                if target_group in visited_groups or target_group not in burst_data:
                                    continue

                                target_bursts = burst_data[target_group]['times']
                                window_bursts = target_bursts[
                                    (target_bursts >= burst_time) & 
                                    (target_bursts <= burst_time + cascade_window_ms)
                                ]
                                
                                if len(window_bursts) > 0:
                                    closest_burst = window_bursts[0]
                                    
                                    cascade['propagation_chain'].append((target_group, closest_burst))
                                    cascade['connections_used'].append((current_group, target_group))
                                    
                                    visited_groups.add(target_group)
                                    next_groups.append(target_group)
                    
                    current_groups = next_groups
                
                if len(cascade['propagation_chain']) >= 2:
                    cascades.append(cascade)
        
        return cascades
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        else:
            plot_order = list(spike_monitors.keys())
        
        if not spike_monitors:
            print("No valid neuron groups for burst cascade analysis.")
            return
        
        connection_map = extract_connection_map(connections_config)
        print(f"Connection map: {connection_map}")
        
        burst_data = {}
        neuron_names = list(spike_monitors.keys())
        
        for name in neuron_names:
            spike_times, _ = get_monitor_spikes(spike_monitors[name])
            if len(spike_times) > 0:
                spike_times_ms = spike_times / ms
                total_neurons = spike_monitors[name].source.N
                
                time_mask = (spike_times_ms >= start_time/ms) & (spike_times_ms <= end_time/ms)
                spike_times_window = spike_times_ms[time_mask]
                
                if len(spike_times_window) > 0:
                    burst_times, burst_strengths = detect_bursts(spike_times_window, total_neurons)
                    burst_data[name] = {
                        'times': burst_times,
                        'strengths': burst_strengths
                    }
                else:
                    burst_data[name] = {'times': np.array([]), 'strengths': np.array([])}
            else:
                burst_data[name] = {'times': np.array([]), 'strengths': np.array([])}
        
        cascades = find_connection_based_cascades(burst_data, connection_map, cascade_window/ms)
        
        # 시각화
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Burst Cascade Visualization (상단 메인 플롯)
        ax1 = plt.subplot(2, 2, (1, 2))  
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(neuron_names)))
        
        # 각 뉴런 그룹의 burst 표시
        for i, name in enumerate(neuron_names):
            if name in burst_data and len(burst_data[name]['times']) > 0:
                burst_times = burst_data[name]['times']
                burst_strengths = burst_data[name]['strengths']
                
                sizes = (burst_strengths / np.max(burst_strengths) * 100) if np.max(burst_strengths) > 0 else [20]
                
                display_name = display_names.get(name, name) if display_names else name
                ax1.scatter(burst_times, [i] * len(burst_times), s=sizes, c=[colors[i]], 
                           alpha=0.8, edgecolors='black', linewidth=1, label=display_name)
        
        # 연결 관계 기반 Cascade 연결선 그리기 (모든 cascade 표시)
        arrow_colors = plt.cm.rainbow(np.linspace(0, 1, min(len(cascades), 50)))  # 최대 50개까지 다른 색상
        
        for idx, cascade in enumerate(cascades):
            if idx >= 50: 
                break
                
            chain = cascade['propagation_chain']
            connections = cascade['connections_used']
            color = arrow_colors[idx] if idx < len(arrow_colors) else 'red'
            
            for i, (connection_pre, connection_post) in enumerate(connections):
                if connection_pre in neuron_names and connection_post in neuron_names:
                    y1 = neuron_names.index(connection_pre)
                    y2 = neuron_names.index(connection_post)
                    
                    pre_time = None
                    post_time = None
                    
                    for group, timing in chain:
                        if group == connection_pre:
                            pre_time = timing
                        elif group == connection_post:
                            post_time = timing
                    
                    if pre_time is not None and post_time is not None:
                        ax1.annotate('', xy=(post_time, y2), xytext=(pre_time, y1),
                                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=2.5,
                                                 connectionstyle="arc3,rad=0.1"))
        
        ax1.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Neural Circuit', fontsize=14, fontweight='bold')
        ax1.set_title('Spike Burst Cascade Propagation', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlim(start_time/ms, end_time/ms)
        ax1.set_ylim(-0.5, len(neuron_names) - 0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Cascade 통계 (왼쪽 아래)
        ax2 = plt.subplot(2, 2, 3)
        
        if cascades:
            cascade_sizes = [len(c['propagation_chain']) for c in cascades]
            if len(cascade_sizes) > 0:
                max_size = max(cascade_sizes)
                ax2.hist(cascade_sizes, bins=range(2, max_size+2), alpha=0.7, 
                        color='skyblue', edgecolor='black')
                ax2.set_xlabel('Number of Participating Groups', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax2.set_title('Connection-Based Cascade Size', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No valid cascade sizes', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'No cascade events detected', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        
        # 3. 뉴런별 burst 통계 (오른쪽 아래)
        ax3 = plt.subplot(2, 2, 4)
        
        burst_counts = []
        group_names = []
        
        for name in neuron_names:
            if name in burst_data:
                count = len(burst_data[name]['times'])
                burst_counts.append(count)
                display_name = display_names.get(name, name) if display_names else name
                group_names.append(display_name)
            else:
                burst_counts.append(0)
                group_names.append(name)
        
        bars = ax3.bar(group_names, burst_counts, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Neural Groups', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Bursts', fontsize=12, fontweight='bold')
        ax3.set_title('Burst Count by Group', fontsize=13, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, burst_counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        for name in neuron_names:
            if name in burst_data:
                count = len(burst_data[name]['times'])
                display_name = display_names.get(name, name) if display_names else name
                print(f"{display_name}: {count} bursts detected")
        
        if cascades:
            avg_cascade_size = np.mean([len(c['propagation_chain']) for c in cascades])
            total_connections_used = sum(len(c['connections_used']) for c in cascades)

            longest_cascade = max(cascades, key=lambda x: len(x['propagation_chain']))
            print(f"Longest cascade: {len(longest_cascade['propagation_chain'])} groups")
            chain_names = [group for group, _ in longest_cascade['propagation_chain']]
        
        try:
            print("\nSpike burst cascade displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying spike burst cascade: {e}")
            
    except Exception as e:
        print(f"Spike burst cascade error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_multi_neuron_stimulus_overview(voltage_monitors, spike_monitors, stimulus_config,
                                      target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
                                      neurons_per_group=3, analysis_window=(0*ms, 10000*ms),
                                      display_names=None, save_plot=True):
    """
    Multiple neurons from all groups with stimulus pattern overview
    - Unified Y-scale for all neuron membrane potentials (stimulus pA 그래프 제외)
    - Spike voltages clipped to threshold values for consistency
    
    Parameters:
    - voltage_monitors: voltage monitor dictionary
    - spike_monitors: spike monitor dictionary  
    - stimulus_config: stimulus configuration
    - target_groups: list of neuron groups to analyze
    - neurons_per_group: number of neurons to show per group
    - analysis_window: analysis time window
    - display_names: display name mapping
    - save_plot: whether to save plot
    """
    
    try:
        # Filter available groups
        available_groups = []
        total_neurons = 0
        
        for group_name in target_groups:
            if (group_name in voltage_monitors and 
                group_name in spike_monitors and
                len(voltage_monitors[group_name].t) > 0):
                available_groups.append(group_name)
                # Check how many neurons are available
                v_monitor = voltage_monitors[group_name]
                available_neurons = len(v_monitor.v)
                neurons_to_use = min(neurons_per_group, available_neurons)
                total_neurons += neurons_to_use
        
        if not available_groups:
            print("No available neuron groups with voltage data")
            return
        
        # Extract time range
        start_time, end_time = analysis_window
        
        # Get time vector from first available monitor
        first_group = available_groups[0]
        v_monitor = voltage_monitors[first_group]
        time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
        time_ms = v_monitor.t[time_mask] / ms
        
        # Collect all voltage data and process spikes for unified y-scale
        all_voltages = []
        neuron_plot_data = []
        
        for group_name in available_groups:
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            
            # Determine how many neurons to plot
            available_neurons = len(v_monitor.v)
            neurons_to_use = min(neurons_per_group, available_neurons)
            
            # Select neurons (spread across available range)
            if neurons_to_use == 1:
                selected_indices = [0]
            else:
                selected_indices = np.linspace(0, available_neurons-1, neurons_to_use, dtype=int)
            
            for neuron_idx in selected_indices:
                # Extract voltage data
                voltage = v_monitor.v[neuron_idx][time_mask] / mV
                
                # Get spikes for this neuron
                spike_times, spike_indices = get_monitor_spikes(s_monitor)
                neuron_spike_mask = spike_indices == neuron_idx
                neuron_spike_times = spike_times[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
                neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
                
                # Estimate threshold voltage
                threshold_voltage = -20  # Default threshold
                if len(neuron_spike_times_window) > 0:
                    spike_voltages = []
                    for spike_time in neuron_spike_times_window[:5]:  # Use first 5 spikes
                        spike_idx = np.argmin(np.abs(time_ms - spike_time))
                        if spike_idx > 0:
                            spike_voltages.append(voltage[spike_idx-1])  # Pre-spike voltage
                    if spike_voltages:
                        threshold_voltage = np.mean(spike_voltages)
                
                # Apply threshold clipping to voltage trace
                clipped_voltage = voltage.copy()
                if len(neuron_spike_times_window) > 0:
                    for spike_time in neuron_spike_times_window:
                        spike_idx = np.argmin(np.abs(time_ms - spike_time))
                        # Clip spike region to threshold value
                        clip_range = range(max(0, spike_idx-2), min(len(clipped_voltage), spike_idx+3))
                        clipped_voltage[clip_range] = threshold_voltage
                
                neuron_plot_data.append({
                    'group_name': group_name,
                    'neuron_idx': neuron_idx,
                    'voltage': clipped_voltage,
                    'spike_times': neuron_spike_times_window,
                    'threshold': threshold_voltage
                })
                
                all_voltages.extend(clipped_voltage)
        
        # Calculate unified y-range (min/max 기준으로 1mV 여유)
        if unified_y_scale and all_voltages:
            y_min = np.min(all_voltages) - 5.0  # 최소값에서 5mV 여유
            y_max = np.max(all_voltages) + 10.0  # 최대값에서 10mV 여유 (스파이크 포인트를 위해)
            y_range = (y_min, y_max)
        else:
            y_range = None
        
        # Create figure with stimulus at top and neurons below
        fig_height = 3 + total_neurons * 1.2  # 3 for stimulus + 1.2 per neuron
        fig, axes = plt.subplots(total_neurons + 1, 1, figsize=(16, fig_height), 
                                sharex=True, gridspec_kw={'height_ratios': [1] + [1]*total_neurons})
        
        # Top panel: Stimulus pattern (separate y-scale)
        stimulus_pA = np.zeros_like(time_ms)
        
        if stimulus_config and stimulus_config.get('enabled', False):
            stim_start = stimulus_config.get('start_time', 0)
            stim_duration = stimulus_config.get('duration', 0)
            stim_end = stim_start + stim_duration
            
            # Use a representative stimulus amplitude
            stimulus_amplitude = 25  # pA
            stim_mask = (time_ms >= stim_start) & (time_ms <= stim_end)
            stimulus_pA[stim_mask] = stimulus_amplitude
        
        # Plot stimulus (독립적인 y축 사용)
        axes[0].plot(time_ms, stimulus_pA, 'k-', linewidth=2)
        axes[0].set_ylabel('Stimulus (pA)', fontsize=12, fontweight='bold')
        axes[0].set_title('Enhanced Multi-Neuron Membrane Potential with Unified Y-Scale', 
                         fontsize=16, fontweight='bold', pad=20)
        axes[0].set_ylim(-5, max(stimulus_pA) + 10 if max(stimulus_pA) > 0 else 30)
        
        # Add stimulus shading if enabled
        if stimulus_config and stimulus_config.get('enabled', False):
            axes[0].axvspan(stim_start, stim_end, alpha=0.15, color='orange', label='Stimulus Period')
            axes[0].legend(loc='upper right', fontsize=9)
        
        # Plot neurons
        for ax_idx, data in enumerate(neuron_plot_data):
            ax = axes[ax_idx + 1]
            
            group_name = data['group_name']
            neuron_idx = data['neuron_idx']
            voltage = data['voltage']
            spike_times = data['spike_times']
            threshold = data['threshold']
            
            # Plot membrane potential
            ax.plot(time_ms, voltage, 'b-', linewidth=1.2, alpha=0.8)
            
            # Mark spikes
            if len(spike_times) > 0:
                spike_voltages = []
                for spike_time in spike_times:
                    spike_idx = np.argmin(np.abs(time_ms - spike_time))
                    if spike_idx < len(voltage):
                        spike_voltages.append(voltage[spike_idx])
                
                ax.scatter(spike_times, spike_voltages, color='red', s=12, 
                          marker='o', alpha=0.8, zorder=5)
                
                # Threshold line
                if threshold_clipping:
                    ax.axhline(threshold, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
            # Set unified y-scale
            if unified_y_scale and y_range:
                ax.set_ylim(y_range)
            
            # Labels
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            ax.set_ylabel(f'{display_name}\n#{neuron_idx}\n(mV)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            # Add spike count info
            spike_count = len(spike_times)
            if spike_count > 0:
                ax.text(0.98, 0.95, f'{spike_count} spikes', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[-1].set_xlim(start_time/ms, end_time/ms)
        
        plt.tight_layout()
        
        # Results
        print(f"\n=== Enhanced Multi-Neuron Analysis Results ===")
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Total neurons displayed: {len(neuron_plot_data)}")
        print(f"Unified Y-scale: {unified_y_scale}")
        print(f"Threshold clipping: {threshold_clipping}")
        
        if unified_y_scale and y_range:
            print(f"Y-axis range: {y_range[0]:.1f} to {y_range[1]:.1f} mV")
        
        for data in neuron_plot_data:
            group_name = data['group_name']
            neuron_idx = data['neuron_idx']
            spike_count = len(data['spike_times'])
            threshold = data['threshold']
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name} #{neuron_idx}: {spike_count} spikes, threshold: {threshold:.1f} mV")
        
        try:
            print("\nEnhanced multi-neuron overview displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying enhanced multi-neuron overview: {e}")
            
    except Exception as e:
        print(f"Enhanced multi-neuron overview error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_continuous_firing_rate_with_samples(spike_monitors, start_time=0*ms, end_time=10000*ms, bin_size=20*ms, 
                                            plot_order=None, display_names=None, stimulus_config=None, 
                                            smooth_sigma=3, save_plot=True, n_samples=10, neurons_per_sample=30):
    """
    연속된 30개 뉴런을 10번 샘플링하여 firing rate 분석
    3x3 grid로 평균 firing rate만 시각화 (individual curve 없음)
    """
    def calculate_firing_rate_for_neuron_subset(spike_times, spike_indices, 
                                              selected_neurons, total_neurons, time_bins):
        firing_rates = []
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            time_mask = (spike_times >= bin_start) & (spike_times < bin_end)
            neuron_mask = np.isin(spike_indices, selected_neurons)
            combined_mask = time_mask & neuron_mask
            spike_count = np.sum(combined_mask)
            bin_duration = (bin_end - bin_start) / 1000.0
            rate = spike_count / (len(selected_neurons) * bin_duration)
            firing_rates.append(rate)
        return np.array(firing_rates)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        else:
            plot_order = list(spike_monitors.keys())
        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        neuron_names = list(spike_monitors.keys())
        n_groups = len(neuron_names)
        # 3x3 grid
        cols = 3
        rows = (n_groups + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        np.random.seed(2025)
        for idx, name in enumerate(neuron_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            spike_times, spike_indices = get_monitor_spikes(monitor[name])
            if len(spike_times) == 0:
                ax.text(0.5, 0.5, f'{name}\nNo spikes', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{name}', fontweight='bold')
                continue
            spike_times_ms = spike_times / ms
            total_neurons = monitor[name].source.N
            time_mask = (spike_times_ms >= start_time/ms) & (spike_times_ms <= end_time/ms)
            spike_times_window = spike_times_ms[time_mask]
            spike_indices_window = spike_indices[time_mask]
            max_start_neuron = max(0, total_neurons - neurons_per_sample)
            sample_firing_rates = []
            for sample_idx in range(n_samples):
                if max_start_neuron <= 0:
                    selected_neurons = np.arange(min(neurons_per_sample, total_neurons))
                else:
                    start_neuron = np.random.randint(0, max_start_neuron + 1)
                    selected_neurons = np.arange(start_neuron, 
                                                min(start_neuron + neurons_per_sample, total_neurons))
                firing_rate = calculate_firing_rate_for_neuron_subset(
                    spike_times_window, spike_indices_window, selected_neurons, 
                    total_neurons, time_bins)
                if smooth_sigma > 0:
                    from scipy.ndimage import gaussian_filter1d
                    firing_rate_smooth = gaussian_filter1d(firing_rate, sigma=smooth_sigma)
                else:
                    firing_rate_smooth = firing_rate
                sample_firing_rates.append(firing_rate_smooth)
            sample_firing_rates = np.array(sample_firing_rates)
            mean_firing_rate = np.mean(sample_firing_rates, axis=0)
            std_firing_rate = np.std(sample_firing_rates, axis=0)
            upper_bound = mean_firing_rate + std_firing_rate
            lower_bound = mean_firing_rate - std_firing_rate
            color = colors[idx % len(colors)]
            display_name = display_names.get(name, name) if display_names else name
            ax.plot(time_centers, mean_firing_rate, color=color, linewidth=2.5, label=f'{display_name}')
            ax.fill_between(time_centers, lower_bound, upper_bound, color=color, alpha=0.3)
            ax.set_title(f'{display_name} - Average Firing Rate', fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            if stimulus_config and stimulus_config.get('enabled', False):
                stim_start = stimulus_config.get('start_time', 0)
                stim_duration = stimulus_config.get('duration', 0)
                stim_end = stim_start + stim_duration
                ax.axvspan(stim_start, stim_end, alpha=0.2, color='red')
                y_max = ax.get_ylim()[1]
                ax.hlines(y_max * 0.95, stim_start, stim_end, colors='red', linewidth=3, alpha=0.8)
                ax.axvline(stim_start, color='red', linestyle=':', alpha=0.7, linewidth=1)
                ax.axvline(stim_end, color='red', linestyle=':', alpha=0.7, linewidth=1)
            max_mean_rate = np.max(mean_firing_rate)
            min_mean_rate = np.min(mean_firing_rate)
            avg_std = np.mean(std_firing_rate)
            stats_text = (f'Max: {max_mean_rate:.1f} Hz\n'
                         f'Min: {min_mean_rate:.1f} Hz\n'
                         f'Avg Std: {avg_std:.1f} Hz')
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlim(start_time/ms, end_time/ms)
        # Hide unused subplots
        for idx in range(n_groups, rows * cols):
            if rows > 1:
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[idx].set_visible(False)
        for ax in axes.flat if hasattr(axes, 'flat') else axes:
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle('Average Firing Rate (Multi-Sample, 3x3 Grid)', fontsize=16, fontweight='bold', y=0.995)
        if save_plot:
            filename = 'continuous_firing_rate_multi_sample_avg_grid.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nContinuous firing rate (multi-sample, avg only) saved to '{filename}'")
        try:
            print("\nContinuous firing rate (multi-sample, avg only) displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying continuous firing rate: {e}")
    except Exception as e:
        print(f"Continuous firing rate error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_membrane_zoom(voltage_monitors, time_window=(0*ms, 100*ms), plot_order=None):
    if plot_order:
        filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
    else:
        filtered_monitors = voltage_monitors

    for name, monitor in filtered_monitors.items():
        t = monitor.t / ms
        v = monitor.v[0] / mV  # 첫 번째 뉴런 예시
        mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
        plt.figure(figsize=(8, 3))
        plt.plot(t[mask], v[mask])
        plt.title(f"{name} Membrane Potential ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)")
        plt.xlabel("Time (ms)")
        plt.ylabel("V (mV)")
        plt.tight_layout()
        plt.show()

def plot_raster_zoom(spike_monitor, time_window=(0*ms, 100*ms), neuron_indices=None):
    t = spike_monitor.t / ms
    i = spike_monitor.i
    mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
    t_zoom = t[mask]
    i_zoom = i[mask]
    if neuron_indices is not None:
        neuron_mask = np.isin(i_zoom, neuron_indices)
        t_zoom = t_zoom[neuron_mask]
        i_zoom = i_zoom[neuron_mask]
    plt.figure(figsize=(8, 3))
    plt.scatter(t_zoom, i_zoom, s=2)
    plt.title(f"Raster Plot ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.show()