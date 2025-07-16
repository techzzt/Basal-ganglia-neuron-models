import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *

# 한국어 폰트 설정
import platform
system = platform.system()

try:
    if system == 'Darwin':  # macOS
        # macOS에서 한국어 폰트 설정
        from matplotlib import font_manager
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        if os.path.exists(font_path):
            plt.rcParams['font.family'] = 'AppleSDGothicNeo'
        else:
            plt.rcParams['font.family'] = 'Arial Unicode MS'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
        
    # 음수 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
except Exception as e:
    print(f"한국어 폰트 설정 중 오류: {e}")
    print("기본 폰트를 사용합니다.")

# Improved backend setup for graph display on macOS
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # Use non-interactive mode if backend not available

# Graph display settings
# plt.ion()  # Disable interactive mode to prevent graphs from disappearing 

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
        
        # Save the plot to file for permanent viewing
        if save_plot:
            filename = 'raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Raster plot saved to '{filename}'")
        
        try:
            print("Raster plot displayed. Plot saved to file for permanent viewing.")
            print("Close the plot window to continue...")
            plt.show(block=True)  # Wait until user closes the window
        except Exception as e:
            print(f"Error displaying raster plot: {e}")
        finally:
            # Don't close immediately, let it stay open
            pass  

        return firing_rates

    except Exception as e:
        print(f"Raster plot Error: {str(e)}")

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

# track individual neuron
def plot_single_neuron_raster(spike_monitors, neuron_index, plot_order=None):
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

            neuron_mask = spike_indices == neuron_index
            neuron_spike_times = spike_times[neuron_mask]

            if len(neuron_spike_times) == 0:
                print(f"{name} - Neuron {neuron_index} has no spikes.")
                continue

            axes[i].scatter(neuron_spike_times/ms, np.full_like(neuron_spike_times/ms, neuron_index), s=0.5, color='red')
            axes[i].set_title(f'{name} Neuron {neuron_index} Raster')
            axes[i].set_ylabel('Neuron index')
            
            if len(spike_times) > 0:
                axes[i].set_xlim(0, int(spike_times[-1] / ms))
            axes[i].set_ylim(neuron_index - 1, neuron_index + 1)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        try:
            plt.show(block=False)  
            plt.pause(0.1) 
        except Exception as e:
            print(f"Error displaying single neuron raster plot: {e}")
            plt.close() 

    except Exception as e:
        print(f"Single neuron raster plot Error: {str(e)}")


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

def plot_isyn(voltage_monitors, plot_order=None):
    plt.figure(figsize=(10, 6))

    if plot_order:
        filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
    else:
        filtered_monitors = voltage_monitors

    for name, monitor in filtered_monitors.items():
        if hasattr(monitor, 'Isyn'):
            plt.plot(monitor.t / ms, monitor.Isyn[0] / pA, label=f'{name} Isyn')
    
    plt.title('Synaptic Current (Isyn) Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Isyn (pA)')
    plt.legend()
    plt.tight_layout()
    
    try:
        plt.show(block=False)
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying Isyn plot: {e}")
        plt.close()  


def analyze_firing_rates_by_stimulus_periods(spike_monitors, stimulus_config, analysis_start_time=2000*ms, plot_order=None, display_names=None):
    """
    스티뮬러스 구간별 평균 발화율을 각 신경세포 그룹마다 계산하고 출력합니다.
    
    Parameters:
    - spike_monitors: 스파이크 모니터 딕셔너리
    - stimulus_config: 스티뮬러스 설정 (start_time, duration 포함)
    - analysis_start_time: 그래프 분석 시작 시간 (기본값 2000ms)
    - plot_order: 출력 순서
    - display_names: 표시 이름 매핑
    """
    
    if not stimulus_config.get('enabled', False):
        print("스티뮬러스가 비활성화되어 있습니다.")
        return
    
    # 스티뮬러스 시간 설정
    stim_start = stimulus_config.get('start_time', 10000) * ms
    stim_duration = stimulus_config.get('duration', 1000) * ms
    stim_end = stim_start + stim_duration
    
    # 구간 정의
    pre_stim_start = analysis_start_time  # 그래프 시작 시간부터
    pre_stim_end = stim_start
    post_stim_start = stim_end
    
    # 분석할 신경세포 그룹 결정
    if plot_order:
        monitors_to_analyze = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
    else:
        monitors_to_analyze = spike_monitors
    
    print("\n" + "="*60)
    print("스티뮬러스 구간별 평균 발화율 분석")
    print("="*60)
    print(f"분석 시작 시간: {analysis_start_time/ms:.0f}ms")
    print(f"스티뮬러스 구간: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms")
    print(f"Pre-stimulus: {pre_stim_start/ms:.0f}-{pre_stim_end/ms:.0f}ms")
    print(f"During-stimulus: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms") 
    print(f"Post-stimulus: {post_stim_start/ms:.0f}ms 이후")
    print("-"*60)
    
    for name, monitor in monitors_to_analyze.items():
        display_name = display_names.get(name, name) if display_names else name
        
        spike_times, spike_indices = get_monitor_spikes(monitor)
        
        if len(spike_times) == 0:
            print(f"\n[{display_name}] - 스파이크 없음")
            continue
            
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
        
        # 변화율 계산
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
                               smooth_sigma=3, save_plot=True, show_confidence=True):
    """
    시간에 따른 연속적인 firing rate 변화를 전문적 스타일로 시각화합니다.
    
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
    """
    
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
        
        # Apply convolution with padding
        padded_data = np.pad(data, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded_data, kernel, mode='valid')
        return smoothed
    
    def calculate_population_firing_rate(spike_times_ms, spike_indices, time_bins, total_neurons):
        """개별 뉴런별 firing rate 계산하여 population statistics 제공"""
        neuron_rates = []
        
        # 각 뉴런별로 firing rate 계산
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

        # 시간 축 설정 (더 세밀한 bin size)
        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        
        # 색상 팔레트 정의 (전문적인 색상)
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        # 전체적인 스타일 설정
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        
        print(f"\nAdvanced firing rate analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms, bin size: {bin_size/ms:.0f}ms")
        
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            
            # Population firing rate 계산
            if show_confidence and total_neurons > 10:  # 충분한 뉴런이 있을 때만 신뢰구간 계산
                mean_rates, std_rates, sem_rates = calculate_population_firing_rate(
                    spike_times_ms, spike_indices, time_bins, total_neurons)
                firing_rates_smooth = gaussian_smooth(mean_rates, smooth_sigma)
                confidence_upper = gaussian_smooth(mean_rates + sem_rates, smooth_sigma)
                confidence_lower = gaussian_smooth(mean_rates - sem_rates, smooth_sigma)
            else:
                # 전체 population의 평균 firing rate
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
            
            # 메인 곡선 그리기
            axes[i].plot(time_centers, firing_rates_smooth, linewidth=2.5, color=color, 
                        label=display_name, alpha=0.9)
            
            # 신뢰구간 표시
            if show_confidence and confidence_upper is not None and confidence_lower is not None:
                axes[i].fill_between(time_centers, confidence_lower, confidence_upper, 
                                   color=color, alpha=0.3, linewidth=0)
            
            # 스티뮬러스 구간 표시 (상단에 tick marks 스타일)
            if stimulus_config and stimulus_config.get('enabled', False):
                stim_start = stimulus_config.get('start_time', 0)
                stim_duration = stimulus_config.get('duration', 0)
                stim_end = stim_start + stim_duration
                
                # 배경 shading 대신 상단에 bar 표시
                y_max = np.max(firing_rates_smooth) * 1.1
                axes[i].hlines(y_max, stim_start, stim_end, colors='red', linewidth=4, alpha=0.8)
                axes[i].text((stim_start + stim_end)/2, y_max * 1.05, 'Stimulus', 
                           ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
                
                # 수직선으로 시작/끝 표시
                axes[i].axvline(stim_start, color='red', linestyle=':', alpha=0.7, linewidth=1)
                axes[i].axvline(stim_end, color='red', linestyle=':', alpha=0.7, linewidth=1)
            
            # 축 및 제목 설정
            axes[i].set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
            axes[i].set_title(f'{display_name}', fontsize=14, fontweight='bold', pad=15)
            
            # 그리드 및 스타일 설정
            axes[i].grid(True, alpha=0.3, linewidth=0.5)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_linewidth(1.5)
            axes[i].spines['bottom'].set_linewidth(1.5)
            
            # Y축 범위 조정
            if np.max(firing_rates_smooth) > 0:
                axes[i].set_ylim(0, np.max(firing_rates_smooth) * 1.15)
            
            print(f"{display_name}: Peak rate = {np.max(firing_rates_smooth):.2f} Hz, Mean rate = {np.mean(firing_rates_smooth):.2f} Hz")

        # X축 설정 (마지막 subplot에만)
        axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[-1].set_xlim(start_time/ms, end_time/ms)
        
        # 전체 레이아웃 조정
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # 파일 저장
        if save_plot:
            filename = 'advanced_firing_rate.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Advanced firing rate plot saved to '{filename}'")
        
        try:
            print("Advanced firing rate plot displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying firing rate plot: {e}")

    except Exception as e:
        print(f"Firing rate plot Error: {str(e)}")


def plot_beta_oscillation_analysis(spike_monitors, start_time=0*ms, end_time=10000*ms,
                                  plot_order=None, display_names=None, stimulus_config=None,
                                  save_plot=True):
    """
    Beta oscillation (13-30 Hz) specialized high-resolution analysis and visualization
    
    Parameters:
    - spike_monitors: Dictionary of spike monitors
    - start_time, end_time: Analysis time range
    - plot_order: Plot order
    - display_names: Display name mapping
    - stimulus_config: Stimulus configuration
    - save_plot: Whether to save plot
    """
    
    def instantaneous_firing_rate(spike_times_ms, total_neurons, time_vector, kernel_width=15):
        """Calculate instantaneous firing rate optimized for beta oscillation"""
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1
        sigma = kernel_width / dt  # Kernel size suitable for beta oscillation
        
        # Generate Gaussian kernel
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / (np.sum(kernel) * dt / 1000.0)
        
        # Convert spikes to binary time series
        spike_train = np.zeros(len(time_vector))
        for spike_time in spike_times_ms:
            idx = np.argmin(np.abs(time_vector - spike_time))
            if 0 <= idx < len(spike_train):
                spike_train[idx] += 1
        
        # Convolution with Gaussian kernel
        padded_spikes = np.pad(spike_train, kernel_size//2, mode='edge')
        ifr = np.convolve(padded_spikes, kernel, mode='valid')
        
        return ifr / total_neurons
    
    def compute_beta_filtered_signal(signal, dt):
        """Filter beta band (13-30 Hz)"""
        from scipy import signal as sig
        fs = 1000 / dt
        
        # Beta band Butterworth bandpass filter
        nyquist = fs / 2
        low = 13 / nyquist
        high = 30 / nyquist
        
        # Filter design
        b, a = sig.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = sig.filtfilt(b, a, signal)
        return filtered_signal
    
    def compute_beta_power_time_series(signal, dt, window_size=200):
        """Calculate beta power changes over time"""
        from scipy import signal as sig
        
        # Beta band filtering
        beta_signal = compute_beta_filtered_signal(signal, dt)
        
        # Calculate instantaneous power using Hilbert transform
        analytic_signal = sig.hilbert(beta_signal)
        instantaneous_power = np.abs(analytic_signal) ** 2
        
        # Power smoothing with sliding window
        window = np.ones(window_size) / window_size
        smoothed_power = np.convolve(instantaneous_power, window, mode='same')
        
        return beta_signal, smoothed_power
    
    def compute_beta_burst_detection(power_signal, threshold_percentile=75):
        """Detect beta bursts"""
        threshold = np.percentile(power_signal, threshold_percentile)
        bursts = power_signal > threshold
        
        # Find consecutive burst periods
        burst_starts = []
        burst_ends = []
        in_burst = False
        
        for i, is_burst in enumerate(bursts):
            if is_burst and not in_burst:
                burst_starts.append(i)
                in_burst = True
            elif not is_burst and in_burst:
                burst_ends.append(i-1)
                in_burst = False
        
        if in_burst:
            burst_ends.append(len(bursts)-1)
            
        return burst_starts, burst_ends, threshold
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups for beta oscillation analysis.")
            return

        # High-resolution time vector (0.5ms resolution for beta analysis)
        dt = 0.5
        time_vector = np.arange(start_time/ms, end_time/ms, dt)
        
        # Color palette for beta oscillation
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E']
        
        n_groups = len(spike_monitors)
        
        # Simplified 2x2 subplot layout for each neuron group
        fig = plt.figure(figsize=(15, 6 * n_groups))
        
        print(f"\nBeta Oscillation Analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms, Resolution: {dt}ms")
        print("Beta Band: 13-30 Hz")
        
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"{name}: No spikes recorded")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            display_name = display_names.get(name, name) if display_names else name
            color = colors[i % len(colors)]
            
            # 1. Calculate instantaneous firing rate
            ifr = instantaneous_firing_rate(spike_times_ms, total_neurons, time_vector)
            
            # 2. Beta signal processing
            beta_signal, beta_power = compute_beta_power_time_series(ifr, dt)
            
            # 3. Beta burst detection
            burst_starts, burst_ends, power_threshold = compute_beta_burst_detection(beta_power)
            
            # Plot 1: Original vs Beta signal
            ax1 = plt.subplot(n_groups, 4, i*4 + 1)
            ax1.plot(time_vector, ifr, color='gray', linewidth=1, alpha=0.7, label='Original')
            ax1.plot(time_vector, beta_signal + np.mean(ifr), color=color, linewidth=1.5, label='Beta Component')
            ax1.set_title(f'{display_name} - Original vs Beta Signal', fontweight='bold')
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Beta filtered signal
            ax2 = plt.subplot(n_groups, 4, i*4 + 2)
            ax2.plot(time_vector, beta_signal, color=color, linewidth=1.5)
            ax2.set_title(f'{display_name} - Beta Band (13-30Hz)', fontweight='bold')
            ax2.set_ylabel('Beta Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Beta power & bursts
            ax3 = plt.subplot(n_groups, 4, i*4 + 3)
            ax3.plot(time_vector, beta_power, color='orange', linewidth=2, label='Beta Power')
            ax3.axhline(power_threshold, color='red', linestyle='--', alpha=0.7, label='Burst Threshold')
            
            # Mark beta burst periods
            for start_idx, end_idx in zip(burst_starts, burst_ends):
                ax3.axvspan(time_vector[start_idx], time_vector[end_idx], 
                           alpha=0.3, color='red', label='Beta Burst' if start_idx == burst_starts[0] else "")
            
            ax3.set_title(f'{display_name} - Beta Power & Bursts', fontweight='bold')
            ax3.set_ylabel('Beta Power')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Beta power summary
            ax4 = plt.subplot(n_groups, 4, i*4 + 4)
            
            # Beta power by time periods (pre/during/post stimulus)
            if stimulus_config and stimulus_config.get('enabled', False):
                stim_start = stimulus_config.get('start_time', 0)
                stim_duration = stimulus_config.get('duration', 0)
                stim_end = stim_start + stim_duration
                
                # Calculate time indices
                pre_mask = time_vector < stim_start
                stim_mask = (time_vector >= stim_start) & (time_vector < stim_end)
                post_mask = time_vector >= stim_end
                
                pre_power = np.mean(beta_power[pre_mask]) if np.any(pre_mask) else 0
                stim_power = np.mean(beta_power[stim_mask]) if np.any(stim_mask) else 0
                post_power = np.mean(beta_power[post_mask]) if np.any(post_mask) else 0
                
                periods = ['Pre-stim', 'During-stim', 'Post-stim']
                powers = [pre_power, stim_power, post_power]
                colors_bar = ['blue', 'red', 'green']
                
                bars = ax4.bar(periods, powers, color=colors_bar, alpha=0.7)
                ax4.set_title(f'Beta Power by Period', fontweight='bold')
                ax4.set_ylabel('Mean Beta Power')
                
                # Show values
                for bar, power in zip(bars, powers):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                           f'{power:.3f}', ha='center', va='bottom')
            else:
                total_power = np.mean(beta_power)
                ax4.bar(['Total Period'], [total_power], color=color, alpha=0.7)
                ax4.set_title(f'Total Beta Power: {total_power:.3f}', fontweight='bold')
                ax4.set_ylabel('Mean Beta Power')
            
            # Add stimulus marking (for time axis plots)
            if stimulus_config and stimulus_config.get('enabled', False):
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(stim_start, stim_end, alpha=0.2, color='red', linewidth=0)
            
            # Print results
            peak_beta_power = np.max(beta_power)
            mean_beta_power = np.mean(beta_power)
            num_bursts = len(burst_starts)
            total_burst_time = sum([(burst_ends[j] - burst_starts[j]) * dt for j in range(len(burst_starts))]) if burst_starts else 0
            burst_percentage = (total_burst_time / (len(time_vector) * dt)) * 100
            
            print(f"\n{display_name} Beta Analysis Results:")
            print(f"  Peak Beta Power: {peak_beta_power:.4f}")
            print(f"  Mean Beta Power: {mean_beta_power:.4f}")
            print(f"  Number of Beta Bursts: {num_bursts}")
            print(f"  Burst Time Percentage: {burst_percentage:.1f}%")

        # X-axis settings
        for i in range(n_groups):
            for j in range(1, 4):  # Time axis plots
                ax = plt.subplot(n_groups, 4, i*4 + j)
                ax.set_xlim(start_time/ms, end_time/ms)
                if i == n_groups - 1:
                    ax.set_xlabel('Time (ms)')
        
        plt.tight_layout(pad=2.0)  # Add more padding between subplots
        
        # Save file
        if save_plot:
            filename = 'beta_oscillation_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nBeta oscillation analysis plot saved to '{filename}'")
        
        try:
            print("\nBeta oscillation analysis displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying beta analysis plot: {e}")

    except Exception as e:
        print(f"Beta Oscillation Analysis Error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_lfp_like_analysis(spike_monitors, start_time=0*ms, end_time=10000*ms, 
                          plot_order=None, display_names=None, stimulus_config=None, 
                          save_plot=True):
    """
    Beta oscillation focused LFP-like analysis (modified version)
    """
    
    def instantaneous_firing_rate(spike_times_ms, total_neurons, time_vector, kernel_width=25):
        """고해상도 순간 발화율 계산 (LFP-like resolution)"""
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1
        sigma = kernel_width / dt
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / (np.sum(kernel) * dt / 1000.0)
        
        # Convert spikes to binary time series
        spike_train = np.zeros(len(time_vector))
        for spike_time in spike_times_ms:
            idx = np.argmin(np.abs(time_vector - spike_time))
            if 0 <= idx < len(spike_train):
                spike_train[idx] += 1
        
        # Convolve with Gaussian kernel
        padded_spikes = np.pad(spike_train, kernel_size//2, mode='edge')
        ifr = np.convolve(padded_spikes, kernel, mode='valid')
        
        return ifr / total_neurons
    
    def compute_power_spectrum(signal, dt, nperseg=None):
        """Power spectral density 계산"""
        from scipy import signal as sig
        if nperseg is None:
            nperseg = min(len(signal)//4, 1024)
        
        frequencies, psd = sig.welch(signal, fs=1000/dt, nperseg=nperseg, 
                                   window='hann', overlap=0.5)
        return frequencies, psd
    
    def compute_spectrogram(signal, dt, nperseg=None):
        """베타 대역 중심 spectrogram"""
        from scipy import signal as sig
        if nperseg is None:
            nperseg = min(len(signal)//8, 256)
            
        frequencies, times, Sxx = sig.spectrogram(signal, fs=1000/dt, nperseg=nperseg,
                                                window='hann', overlap=0.75)
        return frequencies, times, Sxx
    
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups for beta-focused LFP analysis.")
            return

        # High-resolution time vector (1ms resolution for LFP-like analysis)
        dt = 1.0  # 1ms
        time_vector = np.arange(start_time/ms, end_time/ms, dt)
        
        # Color palette
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        
        n_groups = len(spike_monitors)
        
        # 4x1 subplot layout: IFR, PSD (beta-focused), Spectrogram (beta-focused), beta power only
        fig = plt.figure(figsize=(15, 4 * n_groups))
        
        print(f"\nBeta-focused LFP analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms, Resolution: {dt}ms")
        
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_times) == 0:
                print(f"{name}: No spikes recorded")
                continue

            total_neurons = monitor.source.N
            spike_times_ms = spike_times / ms
            display_name = display_names.get(name, name) if display_names else name
            color = colors[i % len(colors)]
            
            # 1. Instantaneous Firing Rate (LFP-like signal)
            ifr = instantaneous_firing_rate(spike_times_ms, total_neurons, time_vector)
            
            # Plot 1: High-resolution firing rate
            ax1 = plt.subplot(n_groups, 4, i*4 + 1)
            ax1.plot(time_vector, ifr, color=color, linewidth=1.5, alpha=0.8)
            ax1.set_title(f'{display_name} - LFP-like Signal', fontweight='bold')
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.grid(True, alpha=0.3)
            
            # Add stimulus marking
            if stimulus_config and stimulus_config.get('enabled', False):
                stim_start = stimulus_config.get('start_time', 0)
                stim_duration = stimulus_config.get('duration', 0)
                stim_end = stim_start + stim_duration
                ax1.axvspan(stim_start, stim_end, alpha=0.3, color='red')
            
            # Plot 2: Power Spectral Density (beta-focused)
            ax2 = plt.subplot(n_groups, 4, i*4 + 2)
            try:
                frequencies, psd = compute_power_spectrum(ifr, dt)
                ax2.loglog(frequencies, psd, color=color, linewidth=2)
                ax2.set_title(f'{display_name} - Power Spectrum (Beta-focused)', fontweight='bold')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('PSD (Hz²/Hz)')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(5, 50)  # Focus on beta band
                
                # Highlight beta band only
                ax2.axvspan(13, 30, alpha=0.3, color='orange', label='Beta Band (13-30Hz)')
                ax2.legend(fontsize=8)
                
            except Exception as e:
                ax2.text(0.5, 0.5, f'PSD Error: {str(e)}', transform=ax2.transAxes, 
                        ha='center', va='center')
            
            # Plot 3: Spectrogram (beta band focused)
            ax3 = plt.subplot(n_groups, 4, i*4 + 3)
            try:
                frequencies, times_spec, Sxx = compute_spectrogram(ifr, dt)
                
                # Limit frequency range to beta band focus
                freq_mask = (frequencies >= 5) & (frequencies <= 50)
                Sxx_filtered = Sxx[freq_mask, :]
                freq_filtered = frequencies[freq_mask]
                
                # Convert to power in dB
                Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)
                
                im = ax3.pcolormesh(times_spec + start_time/ms, freq_filtered, Sxx_db, 
                                  shading='gouraud', cmap='plasma')
                ax3.set_title(f'{display_name} - Spectrogram (Beta-focused)', fontweight='bold')
                ax3.set_ylabel('Frequency (Hz)')
                ax3.set_ylim(5, 50)  # Beta band focus
                
                # Highlight beta band
                ax3.axhspan(13, 30, alpha=0.4, color='white', linestyle='--', linewidth=2)
                ax3.text(0.02, 21, 'Beta', transform=ax3.get_yaxis_transform(), 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                        fontweight='bold', color='black')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3)
                cbar.set_label('Power (dB)', rotation=270, labelpad=15)
                
                # Add stimulus marking
                if stimulus_config and stimulus_config.get('enabled', False):
                    ax3.axvline(stim_start, color='white', linestyle='--', alpha=0.8, linewidth=2)
                    ax3.axvline(stim_end, color='white', linestyle='--', alpha=0.8, linewidth=2)
                    
            except Exception as e:
                ax3.text(0.5, 0.5, f'Spectrogram Error: {str(e)}', transform=ax3.transAxes,
                        ha='center', va='center')
            
            # Plot 4: Beta band power only
            ax4 = plt.subplot(n_groups, 4, i*4 + 4)
            try:
                # Calculate beta band power only
                frequencies, psd = compute_power_spectrum(ifr, dt)
                
                # Extract beta band only
                beta_mask = (frequencies >= 13) & (frequencies <= 30)
                if np.any(beta_mask):
                    beta_power = np.trapz(psd[beta_mask], frequencies[beta_mask])
                    
                    # Beta power bar chart
                    ax4.bar(['Beta Power\n(13-30Hz)'], [beta_power], color='orange', alpha=0.8, width=0.5)
                    ax4.set_title(f'{display_name} - Beta Band Power', fontweight='bold')
                    ax4.set_ylabel('Integrated Power')
                    
                    # Show values
                    ax4.text(0, beta_power, f'{beta_power:.3e}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=10)
                    
                    print(f"{display_name}: Beta Power = {beta_power:.3e}")
                else:
                    ax4.text(0.5, 0.5, 'No Beta Band Data', transform=ax4.transAxes,
                            ha='center', va='center')
                           
            except Exception as e:
                ax4.text(0.5, 0.5, f'Beta Power Error: {str(e)}', transform=ax4.transAxes,
                        ha='center', va='center')

        # Set common x-axis for time plots
        for i in range(n_groups):
            ax = plt.subplot(n_groups, 4, i*4 + 1)
            ax.set_xlim(start_time/ms, end_time/ms)
            if i == n_groups - 1:
                ax.set_xlabel('Time (ms)')
                
            ax = plt.subplot(n_groups, 4, i*4 + 3)  # Spectrogram
            ax.set_xlim(start_time/ms, end_time/ms)
            if i == n_groups - 1:
                ax.set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        # Save file
        if save_plot:
            filename = 'beta_focused_lfp_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Beta-focused LFP analysis plot saved to '{filename}'")
        
        try:
            print("Beta-focused LFP analysis displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying beta-focused LFP analysis: {e}")

    except Exception as e:
        print(f"Beta-focused LFP analysis error: {str(e)}")


def plot_network_connectivity_analysis(spike_monitors, connections_config, plot_order=None, 
                                      display_names=None, target_neuron='FSN', 
                                      time_window=50*ms, save_plot=True):
    """
    Analyze and visualize network connectivity and spike propagation.
    Specialized for input analysis to a specific target neuron (e.g., FSN).
    
    Parameters:
    - spike_monitors: Dictionary of spike monitors
    - connections_config: Connection configuration information
    - plot_order: Plot order
    - display_names: Display name mapping
    - target_neuron: Target neuron to analyze (default: 'FSN')
    - time_window: Time window for spike propagation analysis
    - save_plot: Whether to save plot
    """
    
    def extract_connections_to_target(connections_config, target):
        """Extract connections going to target neuron"""
        incoming_connections = {}
        outgoing_connections = {}
        
        for conn_name, conn_info in connections_config.items():
            if conn_info['post'] == target:
                incoming_connections[conn_name] = conn_info
            elif conn_info['pre'] == target:
                outgoing_connections[conn_name] = conn_info
                
        return incoming_connections, outgoing_connections
    
    def create_connectivity_matrix(connections_config, neuron_names):
        """Create connectivity matrix"""
        n = len(neuron_names)
        connectivity_matrix = np.zeros((n, n))
        weight_matrix = np.zeros((n, n))
        
        name_to_idx = {name: i for i, name in enumerate(neuron_names)}
        
        for conn_name, conn_info in connections_config.items():
            pre = conn_info['pre']
            post = conn_info['post']
            
            # Cortex_ 접두사 제거
            if pre.startswith('Cortex_'):
                pre = pre.replace('Cortex_', '')
            if pre.startswith('Ext_'):
                pre = pre.replace('Ext_', '')
                
            if pre in name_to_idx and post in name_to_idx:
                i, j = name_to_idx[pre], name_to_idx[post]
                connectivity_matrix[i, j] = conn_info['p']  # 연결 확률
                weight_matrix[i, j] = conn_info['weight']    # 시냅스 가중치
                
        return connectivity_matrix, weight_matrix
    
    def analyze_spike_timing_correlation(spike_monitors, pre_neuron, post_neuron, 
                                       time_window=50*ms, max_delay=20*ms):
        """두 뉴런 그룹 간의 스파이크 타이밍 상관관계 분석"""
        if pre_neuron not in spike_monitors or post_neuron not in spike_monitors:
            return None, None, None
            
        pre_times, pre_indices = get_monitor_spikes(spike_monitors[pre_neuron])
        post_times, post_indices = get_monitor_spikes(spike_monitors[post_neuron])
        
        if len(pre_times) == 0 or len(post_times) == 0:
            return None, None, None
        
        # Cross-correlation 계산을 위한 시간 지연 배열
        delays = np.arange(-max_delay/ms, max_delay/ms + 1, 1)  # 1ms 단위
        cross_corr = []
        
        for delay in delays:
            correlation_count = 0
            total_pre_spikes = 0
            
            for pre_time in pre_times/ms:
                # 시간 윈도우 내에서 post 뉴런의 스파이크 확인
                target_time = pre_time + delay
                post_spikes_in_window = np.sum(
                    (post_times/ms >= target_time) & 
                    (post_times/ms < target_time + time_window/ms)
                )
                
                if post_spikes_in_window > 0:
                    correlation_count += 1
                total_pre_spikes += 1
            
            if total_pre_spikes > 0:
                cross_corr.append(correlation_count / total_pre_spikes)
            else:
                cross_corr.append(0)
        
        return delays, np.array(cross_corr), len(pre_times)
    
    try:
        # 모든 post 뉴런 찾기
        post_neurons = get_all_post_neurons(connections_config)
        print(f"\nFound post neurons for connectivity analysis: {post_neurons}")
        
        # 각 post 뉴런별로 개별 분석
        for target_neuron in post_neurons:
            if target_neuron not in spike_monitors:
                print(f"Skipping {target_neuron} - no spike monitor found")
                continue
                
            print(f"\n=== Analyzing connectivity for {target_neuron} ===")
            
            # 해당 뉴런의 연결 정보 추출
            incoming_connections, outgoing_connections = extract_connections_to_target(
                connections_config, target_neuron)
            
            if not incoming_connections and not outgoing_connections:
                print(f"No connections found for {target_neuron}")
                continue
            
            # 각 뉴런별로 개별 그래프 생성 (2x2 레이아웃으로 간소화)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{target_neuron} Network Connectivity Analysis', fontsize=16, fontweight='bold')
            
            # 1. 타겟 뉴런 중심의 간소화된 네트워크 그래프 (왼쪽 위)
            ax1 = axes[0, 0]
            
            try:
                import networkx as nx
                
                # 타겟 뉴런과 직접 연결된 뉴런들만 포함하는 서브그래프 생성
                relevant_neurons = set([target_neuron])
                
                # 입력 뉴런들 추가
                for conn_name, conn_info in incoming_connections.items():
                    pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
                    if pre_neuron in spike_monitors:
                        relevant_neurons.add(pre_neuron)
                
                # 출력 뉴런들 추가
                for conn_name, conn_info in outgoing_connections.items():
                    post_neuron = conn_info['post']
                    if post_neuron in spike_monitors:
                        relevant_neurons.add(post_neuron)
                
                relevant_neurons = list(relevant_neurons)
                print(f"  Relevant neurons for {target_neuron}: {relevant_neurons}")
                
                # 서브그래프 생성
                G = nx.DiGraph()
                
                # 관련 뉴런들만 노드로 추가
                for neuron in relevant_neurons:
                    G.add_node(neuron)
                
                # 관련 연결들만 엣지로 추가
                for conn_name, conn_info in incoming_connections.items():
                    pre = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
                    post = conn_info['post']
                    if pre in relevant_neurons and post in relevant_neurons:
                        G.add_edge(pre, post, weight=conn_info['p'], 
                                 synaptic_weight=conn_info['weight'])
                
                for conn_name, conn_info in outgoing_connections.items():
                    pre = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
                    post = conn_info['post']
                    if pre in relevant_neurons and post in relevant_neurons:
                        G.add_edge(pre, post, weight=conn_info['p'], 
                                 synaptic_weight=conn_info['weight'])
                
                # 타겟 뉴런을 중심으로 한 원형 레이아웃
                if len(relevant_neurons) > 1:
                    pos = {}
                    # 타겟 뉴런을 중앙에 배치
                    pos[target_neuron] = (0, 0)
                    
                    # 다른 뉴런들을 원형으로 배치
                    other_neurons = [n for n in relevant_neurons if n != target_neuron]
                    if other_neurons:
                        angles = np.linspace(0, 2*np.pi, len(other_neurons), endpoint=False)
                        radius = 1.5
                        for i, neuron in enumerate(other_neurons):
                            pos[neuron] = (radius * np.cos(angles[i]), 
                                         radius * np.sin(angles[i]))
                else:
                    pos = {target_neuron: (0, 0)}
                
                # 노드 색상 설정 (간소화)
                node_colors = []
                for node in G.nodes():
                    if node == target_neuron:
                        node_colors.append('red')  # 타겟 뉴런
                    elif any(conn['pre'].replace('Cortex_', '').replace('Ext_', '') == node 
                            for conn in incoming_connections.values()):
                        node_colors.append('lightblue')  # 입력 뉴런
                    elif any(conn['post'] == node for conn in outgoing_connections.values()):
                        node_colors.append('lightgreen')  # 출력 뉴런
                    else:
                        node_colors.append('lightgray')
                
                # 그래프 그리기 (크기와 글자 크기 증가)
                nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors,
                       node_size=2500, font_size=12, font_weight='bold',
                       edge_color='gray', arrows=True, arrowsize=25, 
                       arrowstyle='->', width=2)
                
                ax1.set_title(f'{target_neuron} Network Connections\n({len(relevant_neurons)} neurons)', 
                             fontweight='bold', fontsize=14)
                
                # 범례 추가
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markersize=15, label='Target'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                             markersize=15, label='Input'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                             markersize=15, label='Output')
                ]
                ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                
            except ImportError:
                ax1.text(0.5, 0.5, 'NetworkX required\npip install networkx', 
                        transform=ax1.transAxes, ha='center', va='center', fontsize=14)
            
            # 2. 입력 연결 강도 바 차트 (오른쪽 위)
            ax2 = axes[0, 1]
            
            input_neurons = []
            connection_probs = []
            connection_weights = []
            
            for conn_name, conn_info in incoming_connections.items():
                pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
                input_neurons.append(pre_neuron)
                connection_probs.append(conn_info['p'])
                connection_weights.append(conn_info['weight'])
            
            if input_neurons:
                x = np.arange(len(input_neurons))
                width = 0.35
                
                bars1 = ax2.bar(x - width/2, connection_probs, width, label='Connection Prob', alpha=0.8, color='skyblue')
                bars2 = ax2.bar(x + width/2, connection_weights, width, label='Synaptic Weight', alpha=0.8, color='lightcoral')
                
                ax2.set_xlabel('Input Neurons', fontsize=12)
                ax2.set_ylabel('Strength', fontsize=12)
                ax2.set_title(f'Input Connection Strengths to {target_neuron}', fontsize=13)
                ax2.set_xticks(x)
                ax2.set_xticklabels(input_neurons, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 값 표시
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No incoming connections', transform=ax2.transAxes, 
                        ha='center', va='center', fontsize=14)
            
            # 3. 주요 입력 뉴런과의 Cross-correlation (왼쪽 아래)
            ax3 = axes[1, 0]
            
            if input_neurons:
                # 가장 강한 연결을 가진 입력 뉴런 선택
                if connection_probs:
                    max_prob_idx = np.argmax(connection_probs)
                    main_input = input_neurons[max_prob_idx]
                    
                    delays, cross_corr, n_spikes = analyze_spike_timing_correlation(
                        spike_monitors, main_input, target_neuron, time_window)
                    
                    if cross_corr is not None:
                        ax3.plot(delays, cross_corr, linewidth=3, color='darkgreen', marker='o', markersize=4)
                        ax3.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
                        ax3.set_title(f'{main_input} → {target_neuron} Cross-correlation', fontsize=13, fontweight='bold')
                        ax3.set_xlabel('Delay (ms)', fontsize=12)
                        ax3.set_ylabel('Correlation', fontsize=12)
                        ax3.grid(True, alpha=0.4)
                        
                        # 최대 상관관계 지점 표시
                        if len(cross_corr) > 0:
                            max_idx = np.argmax(cross_corr)
                            max_delay = delays[max_idx]
                            max_corr = cross_corr[max_idx]
                            ax3.plot(max_delay, max_corr, 'ro', markersize=10)
                            ax3.text(max_delay, max_corr + 0.02, f'{max_delay:.0f}ms\n{max_corr:.3f}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                            
                            print(f"  {main_input} → {target_neuron}: Max correlation at {max_delay:.1f}ms, corr = {max_corr:.3f}")
                    else:
                        ax3.text(0.5, 0.5, f'Insufficient data\nfor {main_input} → {target_neuron}', 
                                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
                else:
                    ax3.text(0.5, 0.5, 'No cross-correlation data', transform=ax3.transAxes, 
                            ha='center', va='center', fontsize=12)
            else:
                ax3.text(0.5, 0.5, 'No input neurons to analyze', transform=ax3.transAxes, 
                        ha='center', va='center', fontsize=12)
            
            # 4. 발화율 비교 (오른쪽 아래)
            ax4 = axes[1, 1]
            
            # 각 뉴런의 발화율 계산
            neuron_names = []
            firing_rates = []
            
            # Target 뉴런의 발화율
            if target_neuron in spike_monitors:
                target_times, target_indices = get_monitor_spikes(spike_monitors[target_neuron])
                if len(target_times) > 0:
                    duration = max(target_times) - min(target_times)
                    if duration > 0:
                        target_rate = len(target_times) / (duration/second) / spike_monitors[target_neuron].source.N
                        neuron_names.append(target_neuron)
                        firing_rates.append(target_rate)
            
            # 입력 뉴런들의 발화율
            for input_neuron in input_neurons:
                if input_neuron in spike_monitors:
                    input_times, input_indices = get_monitor_spikes(spike_monitors[input_neuron])
                    if len(input_times) > 0:
                        duration = max(input_times) - min(input_times)
                        if duration > 0:
                            input_rate = len(input_times) / (duration/second) / spike_monitors[input_neuron].source.N
                            neuron_names.append(input_neuron)
                            firing_rates.append(input_rate)
            
            if neuron_names:
                colors = ['red' if name == target_neuron else 'lightblue' for name in neuron_names]
                bars = ax4.bar(neuron_names, firing_rates, color=colors, alpha=0.8)
                ax4.set_xlabel('Neuron Groups', fontsize=12)
                ax4.set_ylabel('Firing Rate (Hz)', fontsize=12)
                ax4.set_title(f'Firing Rates (Target: {target_neuron})', fontsize=13)
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # 값 표시
                for bar, rate in zip(bars, firing_rates):
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{rate:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No firing rate data available', transform=ax4.transAxes, 
                        ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            
            # 개별 파일 저장
            if save_plot:
                filename = f'{target_neuron}_separated_connectivity.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Separated connectivity analysis for {target_neuron} saved to '{filename}'")
            
            try:
                print(f"{target_neuron} connectivity analysis displayed. Close to continue...")
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying {target_neuron} connectivity plot: {e}")
            finally:
                plt.close()
            
    except Exception as e:
        print(f"Network connectivity analysis error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_spike_propagation_analysis(spike_monitors, connections_config, 
                                   target_neuron='FSN', analysis_window=(2000*ms, 3000*ms),
                                   propagation_delay=20*ms, save_plot=True):
    """
    스파이크 전파 패턴을 시간에 따라 분석하고 시각화합니다.
    특정 타겟 뉴런으로의 스파이크 전파 과정을 상세히 보여줍니다.
    
    Parameters:
    - spike_monitors: 스파이크 모니터 딕셔너리
    - connections_config: 연결 설정 정보
    - target_neuron: 분석할 타겟 뉴런
    - analysis_window: 분석할 시간 구간 (start_time, end_time)
    - propagation_delay: 스파이크 전파 지연시간 고려 범위
    - save_plot: 플롯 저장 여부
    """
    
    def find_target_spikes_with_preceding_inputs(target_spikes, input_spikes_dict, 
                                               propagation_delay=20*ms):
        """타겟 스파이크 전에 발생한 입력 스파이크들을 찾습니다"""
        propagation_events = []
        
        for target_time in target_spikes:
            event = {
                'target_time': target_time,
                'inputs': {}
            }
            
            # 각 입력 뉴런에서 이 타겟 스파이크 전에 발생한 스파이크들 찾기
            for input_name, input_times in input_spikes_dict.items():
                # 전파 지연시간 고려하여 선행 스파이크 찾기
                preceding_spikes = input_times[
                    (input_times >= target_time - propagation_delay) & 
                    (input_times < target_time)
                ]
                
                if len(preceding_spikes) > 0:
                    event['inputs'][input_name] = preceding_spikes
            
            # 적어도 하나의 입력이 있는 경우만 추가
            if len(event['inputs']) > 0:
                propagation_events.append(event)
        
        return propagation_events
    
    try:
        start_time, end_time = analysis_window
        
        # 타겟 뉴런으로의 연결 정보 추출
        incoming_connections = {}
        for conn_name, conn_info in connections_config.items():
            if conn_info['post'] == target_neuron:
                incoming_connections[conn_name] = conn_info
        
        print(f"\n=== {target_neuron} 스파이크 전파 분석 ===")
        print(f"분석 구간: {start_time/ms:.0f}-{end_time/ms:.0f}ms")
        print(f"전파 지연시간 고려 범위: {propagation_delay/ms:.0f}ms")
        
        # 타겟 뉴런 스파이크 추출
        if target_neuron not in spike_monitors:
            print(f"오류: {target_neuron} 모니터를 찾을 수 없습니다.")
            return
        
        target_times, target_indices = get_monitor_spikes(spike_monitors[target_neuron])
        target_times_in_window = target_times[
            (target_times >= start_time) & (target_times <= end_time)
        ]
        
        if len(target_times_in_window) == 0:
            print(f"분석 구간에서 {target_neuron} 스파이크가 없습니다.")
            return
        
        # 입력 뉴런들의 스파이크 추출
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
        
        # 스파이크 전파 이벤트 찾기
        propagation_events = find_target_spikes_with_preceding_inputs(
            target_times_in_window, input_spikes_dict, propagation_delay)
        
        print(f"총 {len(target_times_in_window)}개의 {target_neuron} 스파이크 중 "
              f"{len(propagation_events)}개가 선행 입력과 연관됨")
        
        # 시각화
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 전체 스파이크 raster (전파 이벤트 강조)
        ax1 = plt.subplot(3, 2, 1)
        
        y_offset = 0
        neuron_positions = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(input_neuron_names) + 1))
        
        # 입력 뉴런들 그리기
        for i, neuron_name in enumerate(input_neuron_names):
            if neuron_name in input_spikes_dict:
                spike_times = input_spikes_dict[neuron_name] / ms
                y_pos = np.full_like(spike_times, y_offset)
                ax1.scatter(spike_times, y_pos, s=15, c=[colors[i]], 
                           alpha=0.7, label=neuron_name)
                neuron_positions[neuron_name] = y_offset
                y_offset += 1
        
        # 타겟 뉴런 그리기
        target_spike_times = target_times_in_window / ms
        y_pos = np.full_like(target_spike_times, y_offset)
        ax1.scatter(target_spike_times, y_pos, s=30, c='red', 
                   marker='v', alpha=0.9, label=f'{target_neuron} (target)')
        neuron_positions[target_neuron] = y_offset
        
        # 전파 이벤트 연결선 그리기
        for event in propagation_events[:20]:  # 처음 20개만 표시
            target_time = event['target_time'] / ms
            target_y = neuron_positions[target_neuron]
            
            for input_name, input_times in event['inputs'].items():
                input_y = neuron_positions[input_name]
                for input_time in input_times / ms:
                    ax1.plot([input_time, target_time], [input_y, target_y], 
                           'k-', alpha=0.3, linewidth=0.5)
        
        ax1.set_xlim(start_time/ms, end_time/ms)
        ax1.set_xlabel('시간 (ms)')
        ax1.set_ylabel('뉴런')
        ax1.set_title(f'{target_neuron} 스파이크 전파 패턴', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 전파 지연시간 히스토그램
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
        
        # 3. 입력별 기여도 분석
        ax3 = plt.subplot(3, 2, 3)
        
        contribution_counts = []
        contribution_names = []
        
        for input_name in input_neuron_names:
            count = len(input_contributions[input_name])
            contribution_counts.append(count)
            contribution_names.append(input_name)
        
        if len(contribution_counts) > 0:
            bars = ax3.bar(contribution_names, contribution_counts, 
                          color=colors[:len(contribution_names)], alpha=0.8)
            ax3.set_xlabel('입력 뉴런')
            ax3.set_ylabel('기여한 스파이크 수')
            ax3.set_title(f'{target_neuron} 스파이크에 대한 입력별 기여도', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, count in zip(bars, contribution_counts):
                if count > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 시간별 전파 이벤트 밀도
        ax4 = plt.subplot(3, 2, 4)
        
        if len(propagation_events) > 0:
            event_times = [event['target_time']/ms for event in propagation_events]
            
            # 시간 bins 생성
            time_bins = np.linspace(start_time/ms, end_time/ms, 50)
            hist, bin_edges = np.histogram(event_times, bins=time_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            ax4.plot(bin_centers, hist, linewidth=2, color='purple')
            ax4.fill_between(bin_centers, hist, alpha=0.3, color='purple')
            ax4.set_xlabel('시간 (ms)')
            ax4.set_ylabel('전파 이벤트 수')
            ax4.set_title('시간별 스파이크 전파 이벤트 밀도', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. 연결 강도 vs 기여도 상관관계
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
            
            # 상관계수 계산
            if len(connection_strengths) > 1:
                correlation = np.corrcoef(connection_strengths, contributions)[0, 1]
                ax5.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                        transform=ax5.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 6. 상세 통계 정보
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        # 통계 정보 텍스트
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
        
        # 파일 저장
        if save_plot:
            filename = f'{target_neuron}_spike_propagation_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n스파이크 전파 분석 플롯이 '{filename}'에 저장되었습니다.")
        
        try:
            print(f"\n{target_neuron} 스파이크 전파 분석이 표시되었습니다. 플롯 창을 닫으면 계속됩니다...")
            plt.show(block=True)
        except Exception as e:
            print(f"스파이크 전파 분석 표시 오류: {e}")
            
    except Exception as e:
        print(f"스파이크 전파 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_grouped_raster_by_target(spike_monitors, sample_size=8, plot_order=None, 
                                 start_time=0*ms, end_time=1000*ms, display_names=None, save_plot=True):
    """
    Post 뉴런 기준으로 그룹별로 나누어서 raster plot을 표시
    각 그룹의 뉴런 수를 줄여서 겹침 문제 해결
    """
    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        # 각 뉴런 그룹별로 개별 그래프 생성
        for group_name, monitor in spike_monitors.items():
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))  # 더 큰 그래프 크기
            
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {group_name}")
                continue

            total_neurons = monitor.source.N
            # 샘플 크기를 매우 작게 하여 겹침 완전 방지
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)
            chosen_neurons = sorted(chosen_neurons)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            # 뉴런 인덱스를 연속적으로 재배열 (0, 1, 2, 3, ... 형태로)
            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(group_name, group_name) if display_names else group_name
            
            # 큰 점 크기와 높은 투명도로 명확한 표시
            ax.scatter(display_t / ms, remapped_i, s=8.0, alpha=0.9, edgecolors='black', linewidth=0.5)
            ax.set_title(f'{display_name} Raster Plot ({len(chosen_neurons)} neurons)', fontsize=16, pad=20)
            ax.set_ylabel('Neuron Index (Remapped)', fontsize=14)
            ax.set_xlabel('Time (ms)', fontsize=14)

            # Y축을 재배열된 인덱스에 맞게 조정
            if len(chosen_neurons) > 0:
                ax.set_ylim(-0.5, len(chosen_neurons) - 0.5)
                # Y축 틱을 실제 뉴런 번호로 표시
                ax.set_yticks(range(len(chosen_neurons)))
                ax.set_yticklabels([f'N{n}' for n in chosen_neurons])
            ax.set_xlim(int(start_time/ms), int(end_time/ms))
            
            # 가로선 추가로 뉴런 구분
            for i in range(len(chosen_neurons)):
                ax.axhline(y=i-0.5, color='gray', alpha=0.2, linewidth=0.5)
            
            # 격자 추가로 가독성 향상
            ax.grid(True, alpha=0.4, axis='x')
            
            plt.tight_layout(pad=3.0)
            
            # 개별 그룹별로 파일 저장
            if save_plot:
                filename = f'raster_{group_name}_grouped.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Grouped raster plot for {display_name} saved to '{filename}'")
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")
            
            try:
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying grouped raster plot for {group_name}: {e}")
            finally:
                plt.close()

    except Exception as e:
        print(f"Grouped raster plot Error: {str(e)}")

def plot_stimulus_zoom_raster(spike_monitors, stimulus_periods, sample_size=6, 
                             zoom_margin=50*ms, plot_order=None, display_names=None, save_plot=True):
    """
    Stimulus 구간과 주변을 확대해서 spike 패턴을 자세히 보는 함수
    """
    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        # stimulus 구간별로 확대 그래프 생성
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

                # 뉴런 인덱스를 연속적으로 재배열
                neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
                remapped_i = [neuron_mapping[original] for original in display_i]

                display_name = display_names.get(name, name) if display_names else name
                
                # 큰 점으로 표시하여 패턴 명확히 보이게
                axes[i].scatter(display_t / ms, remapped_i, s=12.0, alpha=0.95, edgecolors='black', linewidth=0.8)
                axes[i].set_title(f'{display_name} - Stimulus Period {period_idx + 1} Zoom', fontsize=14, pad=15)
                axes[i].set_ylabel('Neuron Index', fontsize=12)

                if len(chosen_neurons) > 0:
                    axes[i].set_ylim(-0.5, len(chosen_neurons) - 0.5)
                    # Y축 틱을 실제 뉴런 번호로 표시
                    axes[i].set_yticks(range(len(chosen_neurons)))
                    axes[i].set_yticklabels([f'N{n}' for n in chosen_neurons])
                axes[i].set_xlim(int(zoom_start/ms), int(zoom_end/ms))
                
                # 가로선으로 뉴런 구분
                for j in range(len(chosen_neurons)):
                    axes[i].axhline(y=j-0.5, color='gray', alpha=0.3, linewidth=0.5)
                
                # stimulus 구간을 배경색으로 강조
                axes[i].axvspan(stim_start/ms, stim_end/ms, alpha=0.25, color='red', label='Stimulus')
                
                # 격자 추가 (x축만)
                axes[i].grid(True, alpha=0.4, axis='x')
                
                print(f"  {display_name}: {len(display_t)} spikes in zoom window ({sample_size} neurons)")

            axes[-1].set_xlabel('Time (ms)', fontsize=12)
            plt.tight_layout(pad=3.0)
            
            if save_plot:
                filename = f'stimulus_zoom_period_{period_idx + 1}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Stimulus zoom plot period {period_idx + 1} saved to '{filename}'")
            
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
    """
    전체 raster plot 개선 버전: 점 크기와 간격을 조정하여 패턴 식별 용이하게
    """
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

            # 뉴런 인덱스를 연속적으로 재배열
            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(name, name) if display_names else name
            
            # 개선된 점 표시: 크기와 투명도 조정
            axes[i].scatter(display_t / ms, remapped_i, s=5.0, alpha=0.9, edgecolors='black', linewidth=0.3)
            axes[i].set_title(f'{display_name} Raster Plot', fontsize=14, pad=15)
            axes[i].set_ylabel('Neuron Index', fontsize=12)
            
            # 샘플 개수를 그래프 오른쪽 위에 표시
            axes[i].text(0.98, 0.95, f'n={len(chosen_neurons)}', transform=axes[i].transAxes,
                        horizontalalignment='right', verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

            if len(chosen_neurons) > 0:
                axes[i].set_ylim(-0.5, len(chosen_neurons) - 0.5)
                # Y축 틱을 실제 뉴런 번호로 표시 (일부만)
                if len(chosen_neurons) <= 15:  # 뉴런이 적으면 모두 표시
                    axes[i].set_yticks(range(len(chosen_neurons)))
                    axes[i].set_yticklabels([f'N{n}' for n in chosen_neurons])
                else:  # 뉴런이 많으면 일부만 표시
                    tick_indices = range(0, len(chosen_neurons), max(1, len(chosen_neurons)//5))
                    axes[i].set_yticks(tick_indices)
                    axes[i].set_yticklabels([f'N{chosen_neurons[j]}' for j in tick_indices])
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))
            
            # 가로선으로 뉴런 구분 (적당한 간격으로)
            for j in range(0, len(chosen_neurons), max(1, len(chosen_neurons)//10)):
                axes[i].axhline(y=j-0.5, color='gray', alpha=0.2, linewidth=0.3)
            
            # stimulus 구간들을 배경색으로 표시
            if stimulus_periods:
                for period_idx, (stim_start, stim_end) in enumerate(stimulus_periods):
                    if stim_start >= start_time and stim_end <= end_time:
                        axes[i].axvspan(stim_start/ms, stim_end/ms, alpha=0.2, color='red')
            
            # 격자 추가로 가독성 향상 (x축만)
            axes[i].grid(True, alpha=0.4, axis='x')
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")

        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        plt.tight_layout(pad=3.0)
        
        if save_plot:
            filename = 'improved_raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Improved raster plot saved to '{filename}'")
        
        try:
            print("Improved raster plot displayed. Plot saved to file for permanent viewing.")
            print("Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying improved raster plot: {e}")
        finally:
            pass

    except Exception as e:
        print(f"Improved raster plot Error: {str(e)}")

def plot_separated_network_connectivity_analysis(spike_monitors, connections_config, plot_order=None, 
                                                display_names=None, time_window=50*ms, save_plot=True):
    """
    Post 뉴런별로 분리된 네트워크 연결성 분석
    각 post 뉴런에 대해 개별적으로 connectivity analysis를 수행하고 별도 그래프로 표시
    """
    
    def get_all_post_neurons(connections_config):
        """모든 post 뉴런 목록을 추출"""
        post_neurons = set()
        for conn_name, conn_info in connections_config.items():
            post_neurons.add(conn_info['post'])
        return sorted(list(post_neurons))
    
    def extract_connections_to_target(connections_config, target):
        """Extract connections going to target neuron"""
        incoming_connections = {}
        outgoing_connections = {}
        
        for conn_name, conn_info in connections_config.items():
            if conn_info['post'] == target:
                incoming_connections[conn_name] = conn_info
            elif conn_info['pre'] == target:
                outgoing_connections[conn_name] = conn_info
                
        return incoming_connections, outgoing_connections
    
    def analyze_spike_timing_correlation(spike_monitors, pre_neuron, post_neuron, 
                                       time_window=50*ms, max_delay=20*ms):
        """두 뉴런 그룹 간의 스파이크 타이밍 상관관계 분석"""
        if pre_neuron not in spike_monitors or post_neuron not in spike_monitors:
            return None, None, None
            
        pre_times, pre_indices = get_monitor_spikes(spike_monitors[pre_neuron])
        post_times, post_indices = get_monitor_spikes(spike_monitors[post_neuron])
        
        if len(pre_times) == 0 or len(post_times) == 0:
            return None, None, None
        
        # Cross-correlation 계산을 위한 시간 지연 배열
        delays = np.arange(-max_delay/ms, max_delay/ms + 1, 1)  # 1ms 단위
        cross_corr = []
        
        for delay in delays:
            correlation_count = 0
            total_pre_spikes = 0
            
            for pre_time in pre_times/ms:
                # 시간 윈도우 내에서 post 뉴런의 스파이크 확인
                target_time = pre_time + delay
                post_spikes_in_window = np.sum(
                    (post_times/ms >= target_time) & 
                    (post_times/ms < target_time + time_window/ms)
                )
                
                if post_spikes_in_window > 0:
                    correlation_count += 1
                total_pre_spikes += 1
            
            if total_pre_spikes > 0:
                cross_corr.append(correlation_count / total_pre_spikes)
            else:
                cross_corr.append(0)
        
        return delays, np.array(cross_corr), len(pre_times)
    
    try:
        # 타겟 뉴런에 대한 연결 분석
        incoming_connections, outgoing_connections = extract_connections_to_target(
            connections_config, target_neuron)
        
        print(f"\n=== {target_neuron} Neuron Connectivity Analysis ===")
        print(f"Incoming connections: {len(incoming_connections)}")
        print(f"Outgoing connections: {len(outgoing_connections)}")
        
        # 뉴런 이름 목록 생성 (Cortex_, Ext_ 접두사 제거)
        if plot_order:
            neuron_names = [name for name in plot_order if name in spike_monitors]
        else:
            neuron_names = list(spike_monitors.keys())
        
        # 연결성 매트릭스 생성
        connectivity_matrix, weight_matrix = create_connectivity_matrix(
            connections_config, neuron_names)
        
        # 그림 설정
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 네트워크 그래프 시각화
        ax1 = plt.subplot(3, 3, 1)
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # 노드 추가
            for name in neuron_names:
                G.add_node(name)
            
            # 엣지 추가 (연결성 기반)
            for i, pre in enumerate(neuron_names):
                for j, post in enumerate(neuron_names):
                    if connectivity_matrix[i, j] > 0:
                        G.add_edge(pre, post, weight=connectivity_matrix[i, j])
            
            # 타겟 뉴런을 중심으로 한 레이아웃
            if target_neuron in neuron_names:
                # Spring layout 사용
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                # 타겟 뉴런을 중앙에 위치
                if target_neuron in pos:
                    center_x, center_y = pos[target_neuron]
                    # 다른 노드들을 타겟 주변에 재배치
                    for node in pos:
                        if node != target_neuron:
                            angle = np.random.uniform(0, 2*np.pi)
                            radius = 0.8
                            pos[node] = (center_x + radius*np.cos(angle), 
                                       center_y + radius*np.sin(angle))
            else:
                pos = nx.spring_layout(G)
            
            # 노드 색상 설정
            node_colors = []
            for node in G.nodes():
                if node == target_neuron:
                    node_colors.append('red')  # 타겟 뉴런
                elif any(conn['pre'].replace('Cortex_', '').replace('Ext_', '') == node 
                        for conn in incoming_connections.values()):
                    node_colors.append('lightblue')  # 입력 뉴런
                elif any(conn['post'] == node for conn in outgoing_connections.values()):
                    node_colors.append('lightgreen')  # 출력 뉴런
                else:
                    node_colors.append('lightgray')
            
            # 그래프 그리기
            nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors,
                   node_size=1500, font_size=8, font_weight='bold',
                   edge_color='gray', arrows=True, arrowsize=20)
            ax1.set_title(f'{target_neuron} 중심 네트워크 구조', fontweight='bold')
            
        except ImportError:
            ax1.text(0.5, 0.5, 'NetworkX 필요\npip install networkx', 
                    transform=ax1.transAxes, ha='center', va='center')
        
        # 2. 연결성 매트릭스 히트맵
        ax2 = plt.subplot(3, 3, 2)
        im = ax2.imshow(connectivity_matrix, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(neuron_names)))
        ax2.set_yticks(range(len(neuron_names)))
        ax2.set_xticklabels(neuron_names, rotation=45, ha='right')
        ax2.set_yticklabels(neuron_names)
        ax2.set_title('연결 확률 매트릭스', fontweight='bold')
        ax2.set_xlabel('Post-synaptic')
        ax2.set_ylabel('Pre-synaptic')
        
        # 값 표시
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if connectivity_matrix[i, j] > 0:
                    ax2.text(j, i, f'{connectivity_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax2, label='연결 확률')
        
        # 3. 시냅스 가중치 매트릭스
        ax3 = plt.subplot(3, 3, 3)
        im = ax3.imshow(weight_matrix, cmap='Reds', aspect='auto')
        ax3.set_xticks(range(len(neuron_names)))
        ax3.set_yticks(range(len(neuron_names)))
        ax3.set_xticklabels(neuron_names, rotation=45, ha='right')
        ax3.set_yticklabels(neuron_names)
        ax3.set_title('시냅스 가중치 매트릭스', fontweight='bold')
        ax3.set_xlabel('Post-synaptic')
        ax3.set_ylabel('Pre-synaptic')
        
        # 값 표시
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if weight_matrix[i, j] > 0:
                    ax3.text(j, i, f'{weight_matrix[i, j]:.1f}', 
                           ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='시냅스 가중치')
        
        # 4-6. 타겟 뉴런에 대한 입력 분석 (Cross-correlation)
        input_neurons = []
        for conn_name, conn_info in incoming_connections.items():
            pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
            if pre_neuron in spike_monitors:
                input_neurons.append(pre_neuron)
        
        for idx, pre_neuron in enumerate(input_neurons[:3]):  # 최대 3개까지
            ax = plt.subplot(3, 3, 4 + idx)
            
            delays, cross_corr, n_spikes = analyze_spike_timing_correlation(
                spike_monitors, pre_neuron, target_neuron, time_window)
            
            if cross_corr is not None:
                ax.plot(delays, cross_corr, linewidth=2, color=f'C{idx}')
                ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_title(f'{pre_neuron} → {target_neuron}\n교차상관', fontweight='bold')
                ax.set_xlabel('지연시간 (ms)')
                ax.set_ylabel('상관계수')
                ax.grid(True, alpha=0.3)
                
                # 최대 상관관계 지점 표시
                max_idx = np.argmax(cross_corr)
                max_delay = delays[max_idx]
                max_corr = cross_corr[max_idx]
                ax.plot(max_delay, max_corr, 'ro', markersize=8)
                ax.text(max_delay, max_corr + 0.01, f'{max_delay:.0f}ms', 
                       ha='center', va='bottom', fontweight='bold')
                
                print(f"{pre_neuron} → {target_neuron}: 최대 상관관계 지연시간 = {max_delay:.1f}ms, "
                      f"상관계수 = {max_corr:.3f}")
            else:
                ax.text(0.5, 0.5, '데이터 부족', transform=ax.transAxes, 
                       ha='center', va='center')
        
        # 7. 타겟 뉴런의 발화율 vs 입력 강도
        ax7 = plt.subplot(3, 3, 7)
        
        # 각 입력 뉴런의 발화율 계산
        input_rates = []
        input_names = []
        connection_strengths = []
        
        for conn_name, conn_info in incoming_connections.items():
            pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
            if pre_neuron in spike_monitors:
                pre_times, _ = get_monitor_spikes(spike_monitors[pre_neuron])
                if len(pre_times) > 0:
                    # 간단한 발화율 계산
                    duration = np.max(pre_times) if len(pre_times) > 0 else 1*second
                    rate = len(pre_times) / (duration/second)
                    input_rates.append(rate)
                    input_names.append(pre_neuron)
                    
                    # 연결 강도 (확률 × 가중치)
                    strength = conn_info['p'] * conn_info['weight']
                    connection_strengths.append(strength)
        
        if len(input_rates) > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, len(input_rates)))
            scatter = ax7.scatter(input_rates, connection_strengths, 
                                 c=colors, s=100, alpha=0.7)
            
            for i, name in enumerate(input_names):
                ax7.annotate(name, (input_rates[i], connection_strengths[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax7.set_xlabel('입력 뉴런 발화율 (Hz)')
            ax7.set_ylabel('연결 강도 (p × w)')
            ax7.set_title(f'{target_neuron}으로의 입력 분석', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 8. 연결성 통계
        ax8 = plt.subplot(3, 3, 8)
        
        # 각 뉴런의 입력/출력 연결 개수
        in_degrees = np.sum(connectivity_matrix > 0, axis=0)
        out_degrees = np.sum(connectivity_matrix > 0, axis=1)
        
        x = np.arange(len(neuron_names))
        width = 0.35
        
        bars1 = ax8.bar(x - width/2, in_degrees, width, label='입력 연결', alpha=0.8, color='lightblue')
        bars2 = ax8.bar(x + width/2, out_degrees, width, label='출력 연결', alpha=0.8, color='lightcoral')
        
        ax8.set_xlabel('뉴런')
        ax8.set_ylabel('연결 개수')
        ax8.set_title('뉴런별 연결 통계', fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(neuron_names, rotation=45, ha='right')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if in_degrees[i] > 0:
                ax8.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(), 
                        f'{int(in_degrees[i])}', ha='center', va='bottom', fontsize=8)
            if out_degrees[i] > 0:
                ax8.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(), 
                        f'{int(out_degrees[i])}', ha='center', va='bottom', fontsize=8)
        
        # 9. 타겟 뉴런 상세 정보
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # 타겟 뉴런 정보 텍스트
        info_text = f"{target_neuron} 뉴런 상세 정보\n\n"
        info_text += f"들어오는 연결:\n"
        for conn_name, conn_info in incoming_connections.items():
            pre = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
            p = conn_info['p']
            w = conn_info['weight']
            receptor = conn_info['receptor_type']
            if isinstance(receptor, list):
                receptor = ', '.join(receptor)
            info_text += f"  {pre} → {target_neuron}\n"
            info_text += f"    확률: {p:.3f}, 가중치: {w:.1f}\n"
            info_text += f"    수용체: {receptor}\n\n"
        
        info_text += f"나가는 연결:\n"
        for conn_name, conn_info in outgoing_connections.items():
            post = conn_info['post']
            p = conn_info['p']
            w = conn_info['weight']
            receptor = conn_info['receptor_type']
            if isinstance(receptor, list):
                receptor = ', '.join(receptor)
            info_text += f"  {target_neuron} → {post}\n"
            info_text += f"    확률: {p:.3f}, 가중치: {w:.1f}\n"
            info_text += f"    수용체: {receptor}\n\n"
        
        ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save file
        if save_plot:
            filename = f'{target_neuron}_connectivity_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nConnectivity analysis plot saved to '{filename}'")
        
        try:
            print(f"\n{target_neuron} connectivity analysis displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying connectivity analysis plot: {e}")
            
    except Exception as e:
        print(f"Network connectivity analysis error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_separated_spike_propagation_analysis(spike_monitors, connections_config, 
                                            analysis_window=(0*ms, 2000*ms), propagation_delay=20*ms, 
                                            save_plot=True):
    """
    Post 뉴런별로 분리된 spike propagation 분석
    각 post 뉴런에 대해 개별적으로 spike propagation을 분석하고 별도 그래프로 표시
    """
    
    def get_all_post_neurons(connections_config):
        """모든 post 뉴런 목록을 추출"""
        post_neurons = set()
        for conn_name, conn_info in connections_config.items():
            post_neurons.add(conn_info['post'])
        return sorted(list(post_neurons))
    
    def extract_connections_to_target(connections_config, target):
        """Extract connections going to target neuron"""
        incoming_connections = {}
        
        for conn_name, conn_info in connections_config.items():
            if conn_info['post'] == target:
                incoming_connections[conn_name] = conn_info
                
        return incoming_connections
    
    def analyze_spike_propagation_to_target(spike_monitors, pre_neuron, post_neuron, 
                                          analysis_window, propagation_delay):
        """특정 pre 뉴런에서 post 뉴런으로의 spike propagation 분석"""
        if pre_neuron not in spike_monitors or post_neuron not in spike_monitors:
            return None, None, None, None
            
        pre_times, pre_indices = get_monitor_spikes(spike_monitors[pre_neuron])
        post_times, post_indices = get_monitor_spikes(spike_monitors[post_neuron])
        
        # 분석 윈도우 내의 스파이크만 고려
        pre_mask = (pre_times >= analysis_window[0]) & (pre_times <= analysis_window[1])
        post_mask = (post_times >= analysis_window[0]) & (post_times <= analysis_window[1])
        
        pre_times_window = pre_times[pre_mask]
        pre_indices_window = pre_indices[pre_mask]
        post_times_window = post_times[post_mask]
        post_indices_window = post_indices[post_mask]
        
        if len(pre_times_window) == 0 or len(post_times_window) == 0:
            return None, None, None, None
        
        # Propagation 이벤트 찾기
        propagation_events = []
        delays = []
        
        for pre_time in pre_times_window:
            # 각 pre 스파이크 이후 propagation_delay 시간 내에 post 스파이크 찾기
            time_window_start = pre_time
            time_window_end = pre_time + propagation_delay
            
            # 해당 시간 윈도우 내의 post 스파이크들
            post_spikes_in_window = post_times_window[
                (post_times_window >= time_window_start) & 
                (post_times_window <= time_window_end)
            ]
            
            if len(post_spikes_in_window) > 0:
                # 가장 가까운 post 스파이크 선택
                closest_post_spike = post_spikes_in_window[0]
                delay = (closest_post_spike - pre_time) / ms
                
                propagation_events.append((pre_time/ms, closest_post_spike/ms))
                delays.append(delay)
        
        return propagation_events, delays, len(pre_times_window), len(post_times_window)
    
    try:
        # 모든 post 뉴런 찾기
        post_neurons = get_all_post_neurons(connections_config)
        print(f"\nFound post neurons for propagation analysis: {post_neurons}")
        
        # 각 post 뉴런별로 개별 분석
        for target_neuron in post_neurons:
            if target_neuron not in spike_monitors:
                print(f"Skipping {target_neuron} - no spike monitor found")
                continue
                
            print(f"\n=== Analyzing spike propagation for {target_neuron} ===")
            
            # 해당 뉴런의 입력 연결 정보 추출
            incoming_connections = extract_connections_to_target(connections_config, target_neuron)
            
            if not incoming_connections:
                print(f"No incoming connections found for {target_neuron}")
                continue
            
            # 각 뉴런별로 개별 그래프 생성 (2x2 레이아웃)
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{target_neuron} Spike Propagation Analysis', fontsize=16, fontweight='bold')
            
            # 1. Raster plot with propagation events (왼쪽 위)
            ax1 = axes[0, 0]
            
            # Target 뉴런 스파이크 표시
            target_times, target_indices = get_monitor_spikes(spike_monitors[target_neuron])
            time_mask = (target_times >= analysis_window[0]) & (target_times <= analysis_window[1])
            target_times_window = target_times[time_mask]
            target_indices_window = target_indices[time_mask]
            
            # 샘플링된 뉴런만 표시 (겹침 방지)
            if len(target_indices_window) > 0:
                unique_indices = np.unique(target_indices_window)
                sample_indices = np.random.choice(unique_indices, 
                                                size=min(8, len(unique_indices)), replace=False)
                
                mask = np.isin(target_indices_window, sample_indices)
                ax1.scatter(target_times_window[mask]/ms, target_indices_window[mask], 
                           s=15, alpha=0.8, color='red', label=f'{target_neuron} spikes')
            
            # 주요 입력 뉴런의 스파이크와 propagation 표시
            input_neurons = []
            colors = ['blue', 'green', 'orange', 'purple']
            
            propagation_stats = {}
            
            for idx, (conn_name, conn_info) in enumerate(list(incoming_connections.items())[:3]):  # 최대 3개
                pre_neuron = conn_info['pre'].replace('Cortex_', '').replace('Ext_', '')
                if pre_neuron not in spike_monitors:
                    continue
                    
                input_neurons.append(pre_neuron)
                color = colors[idx % len(colors)]
                
                pre_times, pre_indices = get_monitor_spikes(spike_monitors[pre_neuron])
                time_mask = (pre_times >= analysis_window[0]) & (pre_times <= analysis_window[1])
                pre_times_window = pre_times[time_mask]
                pre_indices_window = pre_indices[time_mask]
                
                # 샘플링된 뉴런만 표시
                if len(pre_indices_window) > 0:
                    unique_indices = np.unique(pre_indices_window)
                    sample_indices = np.random.choice(unique_indices, 
                                                    size=min(6, len(unique_indices)), replace=False)
                    
                    mask = np.isin(pre_indices_window, sample_indices)
                    y_offset = (idx + 1) * 50  # Y축 오프셋으로 뉴런 그룹 구분
                    ax1.scatter(pre_times_window[mask]/ms, pre_indices_window[mask] + y_offset, 
                               s=10, alpha=0.7, color=color, label=f'{pre_neuron} spikes')
                
                # Propagation 이벤트 분석
                prop_events, delays, n_pre, n_post = analyze_spike_propagation_to_target(
                    spike_monitors, pre_neuron, target_neuron, analysis_window, propagation_delay)
                
                if prop_events:
                    propagation_stats[pre_neuron] = {
                        'events': len(prop_events),
                        'delays': delays,
                        'pre_spikes': n_pre,
                        'post_spikes': n_post,
                        'propagation_rate': len(prop_events) / n_pre if n_pre > 0 else 0
                    }
                    
                    # Propagation 연결선 표시 (일부만)
                    for i, (pre_time, post_time) in enumerate(prop_events[:20]):  # 최대 20개
                        ax1.plot([pre_time, post_time], [y_offset, 0], 
                                color=color, alpha=0.3, linewidth=0.5)
            
            ax1.set_xlabel('Time (ms)', fontsize=12)
            ax1.set_ylabel('Neuron Index (with offset)', fontsize=12)
            ax1.set_title(f'Spike Propagation to {target_neuron}', fontsize=13)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(analysis_window[0]/ms, analysis_window[1]/ms)
            
            # 2. Propagation delay histogram (오른쪽 위)
            ax2 = axes[0, 1]
            
            all_delays = []
            delay_labels = []
            
            for pre_neuron, stats in propagation_stats.items():
                if stats['delays']:
                    all_delays.extend(stats['delays'])
                    delay_labels.extend([pre_neuron] * len(stats['delays']))
            
            if all_delays:
                ax2.hist(all_delays, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel('Propagation Delay (ms)', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title(f'Propagation Delay Distribution to {target_neuron}', fontsize=13)
                ax2.axvline(np.mean(all_delays), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_delays):.1f}ms')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No propagation events detected', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            
            # 3. Propagation statistics bar chart (왼쪽 아래)
            ax3 = axes[1, 0]
            
            if propagation_stats:
                neurons = list(propagation_stats.keys())
                prop_rates = [propagation_stats[n]['propagation_rate'] for n in neurons]
                n_events = [propagation_stats[n]['events'] for n in neurons]
                
                x = np.arange(len(neurons))
                width = 0.35
                
                bars1 = ax3.bar(x - width/2, prop_rates, width, label='Propagation Rate', 
                               alpha=0.8, color='lightcoral')
                bars2 = ax3.bar(x + width/2, np.array(n_events)/max(max(n_events), 1), width, 
                               label='Normalized Events', alpha=0.8, color='lightblue')
                
                ax3.set_xlabel('Input Neurons', fontsize=12)
                ax3.set_ylabel('Rate / Normalized Count', fontsize=12)
                ax3.set_title(f'Propagation Statistics to {target_neuron}', fontsize=13)
                ax3.set_xticks(x)
                ax3.set_xticklabels(neurons, rotation=45, ha='right')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # 값 표시
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    ax3.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                            f'{prop_rates[i]:.3f}', ha='center', va='bottom', fontsize=9)
                    ax3.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                            f'{n_events[i]}', ha='center', va='bottom', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'No propagation statistics available', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            
            # 4. 상세 정보 텍스트 (오른쪽 아래)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            info_text = f"{target_neuron} Propagation Summary\n\n"
            info_text += f"Analysis Window: {analysis_window[0]/ms:.0f} - {analysis_window[1]/ms:.0f} ms\n"
            info_text += f"Max Propagation Delay: {propagation_delay/ms:.0f} ms\n\n"
            
            for pre_neuron, stats in propagation_stats.items():
                info_text += f"{pre_neuron} → {target_neuron}:\n"
                info_text += f"  Total Events: {stats['events']}\n"
                info_text += f"  Pre Spikes: {stats['pre_spikes']}\n"
                info_text += f"  Propagation Rate: {stats['propagation_rate']:.3f}\n"
                if stats['delays']:
                    info_text += f"  Mean Delay: {np.mean(stats['delays']):.1f} ms\n"
                    info_text += f"  Delay Range: {min(stats['delays']):.1f} - {max(stats['delays']):.1f} ms\n"
                info_text += "\n"
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
            plt.tight_layout()
            
            # 개별 파일 저장
            if save_plot:
                filename = f'{target_neuron}_separated_propagation.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Separated propagation analysis for {target_neuron} saved to '{filename}'")
            
            # 통계 출력
            print(f"  Propagation events summary:")
            for pre_neuron, stats in propagation_stats.items():
                print(f"    {pre_neuron} → {target_neuron}: {stats['events']} events, "
                      f"rate = {stats['propagation_rate']:.3f}")
            
            try:
                print(f"{target_neuron} propagation analysis displayed. Close to continue...")
                plt.show(block=True)
            except Exception as e:
                print(f"Error displaying {target_neuron} propagation plot: {e}")
            finally:
                plt.close()
                
    except Exception as e:
        print(f"Separated spike propagation analysis error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_circuit_flow_heatmap(spike_monitors, connections_config, 
                             start_time=0*ms, end_time=2000*ms, bin_size=10*ms,
                             plot_order=None, display_names=None, save_plot=True):
    """
    뉴런 circuit을 통한 spike propagation을 heat map으로 시각화
    시간에 따른 각 뉴런 그룹의 활성도를 색깔로 표시하여 propagation flow를 명확히 보여줌
    
    Parameters:
    - spike_monitors: 스파이크 모니터 딕셔너리
    - connections_config: 연결 설정 정보
    - start_time, end_time: 분석 시간 범위
    - bin_size: 시간 해상도 (기본값 10ms)
    - plot_order: 뉴런 그룹 순서
    - display_names: 표시 이름 매핑
    - save_plot: 플롯 저장 여부
    """
    
    def calculate_instantaneous_activity(spike_times_ms, total_neurons, time_bins):
        """각 시간 구간에서의 순간 활성도 계산 (0-1 normalized)"""
        activity = []
        for i in range(len(time_bins)-1):
            bin_start = time_bins[i]
            bin_end = time_bins[i+1]
            spikes_in_bin = np.sum((spike_times_ms >= bin_start) & (spike_times_ms < bin_end))
            # Normalize by time window and number of neurons
            normalized_activity = spikes_in_bin / (total_neurons * (bin_end - bin_start) / 1000.0)
            activity.append(normalized_activity)
        return np.array(activity)
    
    def detect_propagation_waves(activity_matrix, time_centers, neuron_order):
        """propagation wave 이벤트를 감지"""
        waves = []
        
        # 각 시간점에서 활성도가 높은 순서대로 wave 감지
        for t_idx in range(len(time_centers)):
            activities = activity_matrix[:, t_idx]
            
            # 임계값 이상의 활성도를 가진 뉴런 그룹들
            active_groups = np.where(activities > np.percentile(activities, 75))[0]
            
            if len(active_groups) > 1:
                # 활성화 순서가 circuit flow와 일치하는지 확인
                is_wave = True
                for i in range(len(active_groups)-1):
                    if active_groups[i] >= active_groups[i+1]:  # 순서가 맞지 않으면
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
        
        # 시간 축 설정
        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        
        # 각 뉴런 그룹의 활성도 계산
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
        
        # Propagation wave 감지
        waves = detect_propagation_waves(activity_matrix, time_centers, neuron_names)
        
        # 시각화 - 단일 메인 heat map만 표시
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Heat map 생성 (더 강렬한 colormap 사용)
        im = ax.imshow(activity_matrix, cmap='plasma', aspect='auto', 
                      interpolation='bilinear', alpha=0.9)
        
        # 축 설정
        # X축 - 100ms 단위로 시간 축 표시
        time_start = time_centers[0]
        time_end = time_centers[-1]
        
        # 100ms 간격으로 tick 위치 계산
        tick_interval = 100  # 100ms 간격
        tick_times = np.arange(
            np.ceil(time_start / tick_interval) * tick_interval,  # 첫 번째 100ms 단위
            time_end + tick_interval,
            tick_interval
        )
        
        # tick_times에 해당하는 인덱스 찾기
        time_tick_indices = []
        tick_labels = []
        for tick_time in tick_times:
            if tick_time <= time_end:
                # 가장 가까운 시간 인덱스 찾기
                closest_idx = np.argmin(np.abs(time_centers - tick_time))
                time_tick_indices.append(closest_idx)
                tick_labels.append(f'{tick_time:.0f}')
        
        ax.set_xticks(time_tick_indices)
        ax.set_xticklabels(tick_labels)
        
        # Y축 - 뉴런 그룹
        ax.set_yticks(range(len(neuron_names)))
        
        # Display names 적용
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
        
        # 그리드 제거 - 더 깔끔한 시각화
        
        plt.tight_layout()
        
        # 결과 출력
        print(f"\n=== Circuit Flow Analysis Results ===")
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
        
        # 파일 저장
        if save_plot:
            filename = 'circuit_flow_heatmap.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nCircuit flow heat map saved to '{filename}'")
        
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
    """
    Spike burst가 circuit을 통해 cascade되는 패턴을 시각화 (연결 관계 기반)
    각 burst 이벤트를 감지하고 실제 pre-post 연결을 따라 cascade propagation을 추적
    
    Parameters:
    - spike_monitors: 스파이크 모니터 딕셔너리
    - connections_config: 연결 설정 정보 (pre-post 관계)
    - start_time, end_time: 분석 시간 범위
    - burst_threshold: burst 감지 임계값 (percentile)
    - cascade_window: cascade 감지 시간 윈도우
    - plot_order: 뉴런 그룹 순서
    - display_names: 표시 이름 매핑
    - save_plot: 플롯 저장 여부
    """
    
    def detect_bursts(spike_times_ms, total_neurons, bin_size=20):
        """Burst 이벤트 감지"""
        if len(spike_times_ms) == 0:
            return [], []
        
        # 전체 분석 구간 기준으로 bins 생성
        bins = np.arange(start_time/ms, end_time/ms + bin_size, bin_size)
        
        # 각 bin에서의 firing rate 계산
        hist, bin_edges = np.histogram(spike_times_ms, bins=bins)
        firing_rates = hist / (total_neurons * bin_size / 1000.0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Burst 임계값 계산 (0이 아닌 값들만 사용)
        non_zero_rates = firing_rates[firing_rates > 0]
        if len(non_zero_rates) > 0:
            threshold = np.percentile(non_zero_rates, burst_threshold * 100)
        else:
            threshold = 0
        
        # Burst 구간 찾기
        burst_indices = np.where(firing_rates > threshold)[0]
        burst_times = bin_centers[burst_indices]
        burst_strengths = firing_rates[burst_indices]
        
        return burst_times, burst_strengths
    
    def extract_connection_map(connections_config):
        """연결 관계 매핑 생성 (pre -> [post1, post2, ...])"""
        connection_map = {}
        for conn_name, conn_info in connections_config.items():
            pre = conn_info['pre']
            post = conn_info['post']
            
            if pre not in connection_map:
                connection_map[pre] = []
            connection_map[pre].append(post)
        
        return connection_map
    
    def find_connection_based_cascades(burst_data, connection_map, cascade_window_ms):
        """연결 관계 기반 cascade 이벤트 감지"""
        cascades = []
        
        # 모든 뉴런 그룹에서 burst 시작점을 찾아 cascade 추적
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
                
                # 현재 그룹에서 시작해서 연결된 그룹들로 cascade 추적
                current_groups = [start_group]
                visited_groups = {start_group}
                
                while current_groups:
                    next_groups = []
                    
                    for current_group in current_groups:
                        # 현재 그룹에서 연결된 타겟 그룹들 찾기
                        if current_group in connection_map:
                            targets = connection_map[current_group]
                            
                            for target_group in targets:
                                if target_group in visited_groups or target_group not in burst_data:
                                    continue
                                
                                # 타겟 그룹에서 cascade window 내의 burst 찾기
                                target_bursts = burst_data[target_group]['times']
                                window_bursts = target_bursts[
                                    (target_bursts >= burst_time) & 
                                    (target_bursts <= burst_time + cascade_window_ms)
                                ]
                                
                                if len(window_bursts) > 0:
                                    # 가장 가까운 burst 선택
                                    closest_burst = window_bursts[0]
                                    
                                    cascade['propagation_chain'].append((target_group, closest_burst))
                                    cascade['connections_used'].append((current_group, target_group))
                                    
                                    visited_groups.add(target_group)
                                    next_groups.append(target_group)
                    
                    current_groups = next_groups
                
                # 적어도 2개 그룹이 참여한 경우만 cascade로 인정
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
        
        # 연결 관계 매핑 생성
        connection_map = extract_connection_map(connections_config)
        print(f"Connection map: {connection_map}")
        
        # 각 뉴런 그룹의 burst 감지
        burst_data = {}
        neuron_names = list(spike_monitors.keys())
        
        for name in neuron_names:
            spike_times, _ = get_monitor_spikes(spike_monitors[name])
            if len(spike_times) > 0:
                spike_times_ms = spike_times / ms
                total_neurons = spike_monitors[name].source.N
                
                # 분석 구간 내의 스파이크만 사용
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
        
        # 연결 관계 기반 Cascade 이벤트 감지
        cascades = find_connection_based_cascades(burst_data, connection_map, cascade_window/ms)
        
        # 시각화
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Burst Cascade Visualization (상단 메인 플롯)
        ax1 = plt.subplot(2, 2, (1, 2))  # 상단 전체
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(neuron_names)))
        
        # 각 뉴런 그룹의 burst 표시
        for i, name in enumerate(neuron_names):
            if name in burst_data and len(burst_data[name]['times']) > 0:
                burst_times = burst_data[name]['times']
                burst_strengths = burst_data[name]['strengths']
                
                # Burst 강도를 점 크기로 표현
                sizes = (burst_strengths / np.max(burst_strengths) * 100) if np.max(burst_strengths) > 0 else [20]
                
                display_name = display_names.get(name, name) if display_names else name
                ax1.scatter(burst_times, [i] * len(burst_times), s=sizes, c=[colors[i]], 
                           alpha=0.8, edgecolors='black', linewidth=1, label=display_name)
        
        # 연결 관계 기반 Cascade 연결선 그리기 (모든 cascade 표시)
        arrow_colors = plt.cm.rainbow(np.linspace(0, 1, min(len(cascades), 50)))  # 최대 50개까지 다른 색상
        
        for idx, cascade in enumerate(cascades):
            if idx >= 50:  # 너무 많으면 50개까지만 표시
                break
                
            chain = cascade['propagation_chain']
            connections = cascade['connections_used']
            color = arrow_colors[idx] if idx < len(arrow_colors) else 'red'
            
            # propagation chain을 따라 화살표 그리기
            for i, (connection_pre, connection_post) in enumerate(connections):
                if connection_pre in neuron_names and connection_post in neuron_names:
                    y1 = neuron_names.index(connection_pre)
                    y2 = neuron_names.index(connection_post)
                    
                    # 해당 연결에서의 시간 찾기
                    pre_time = None
                    post_time = None
                    
                    for group, timing in chain:
                        if group == connection_pre:
                            pre_time = timing
                        elif group == connection_post:
                            post_time = timing
                    
                    if pre_time is not None and post_time is not None:
                        # 화살표로 방향 표시 (더 굵고 뚜렷하게)
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
            # Cascade 참여 그룹 수 히스토그램
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
        
        # 값 표시
        for bar, count in zip(bars, burst_counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 결과 출력
        print(f"\n=== Connection-Based Burst Cascade Analysis Results ===")
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Burst threshold: {burst_threshold*100:.0f}th percentile")
        print(f"Cascade window: {cascade_window/ms:.0f} ms")
        print(f"Detected cascade events: {len(cascades)}")
        print(f"Total connections available: {sum(len(targets) for targets in connection_map.values())}")
        
        for name in neuron_names:
            if name in burst_data:
                count = len(burst_data[name]['times'])
                display_name = display_names.get(name, name) if display_names else name
                print(f"{display_name}: {count} bursts detected")
        
        if cascades:
            avg_cascade_size = np.mean([len(c['propagation_chain']) for c in cascades])
            total_connections_used = sum(len(c['connections_used']) for c in cascades)
            print(f"Average cascade size: {avg_cascade_size:.1f} groups")
            print(f"Total connections used in cascades: {total_connections_used}")
            
            # 가장 활발한 cascade 표시
            longest_cascade = max(cascades, key=lambda x: len(x['propagation_chain']))
            print(f"Longest cascade: {len(longest_cascade['propagation_chain'])} groups")
            chain_names = [group for group, _ in longest_cascade['propagation_chain']]
            print(f"  Path: {' → '.join(chain_names)}")
        
        # 파일 저장
        if save_plot:
            filename = 'spike_burst_cascade.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nSpike burst cascade plot saved to '{filename}'")
        
        try:
            print("\nSpike burst cascade displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying spike burst cascade: {e}")
            
    except Exception as e:
        print(f"Spike burst cascade error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_single_neuron_detailed_analysis(voltage_monitors, spike_monitors, neuron_group_name, 
                                        neuron_index=0, analysis_window=(0*ms, 2000*ms),
                                        display_names=None, save_plot=True):
    """
    개별 뉴런의 상세한 분석 및 시각화
    Phase space, firing patterns, ISI analysis 등을 포함
    
    Parameters:
    - voltage_monitors: 전압 모니터 딕셔너리
    - spike_monitors: 스파이크 모니터 딕셔너리
    - neuron_group_name: 분석할 뉴런 그룹 이름
    - neuron_index: 분석할 뉴런 인덱스
    - analysis_window: 분석 시간 구간
    - display_names: 표시 이름 매핑
    - save_plot: 플롯 저장 여부
    """
    
    def calculate_phase_space(voltage, dt_ms):
        """Phase space (V vs dV/dt) 계산"""
        dV_dt = np.gradient(voltage, dt_ms)
        return voltage, dV_dt
    
    def detect_firing_patterns(spike_times_ms, isi_threshold=100):
        """Firing pattern 분류 (burst vs regular)"""
        if len(spike_times_ms) < 2:
            return "No spikes", [], []
        
        isis = np.diff(spike_times_ms)
        
        # Burst 감지: ISI가 임계값보다 작은 연속적인 스파이크들
        burst_indices = np.where(isis < isi_threshold)[0]
        
        bursts = []
        regular_spikes = []
        
        if len(burst_indices) > 0:
            # Burst 구간 찾기
            burst_starts = [burst_indices[0]]
            burst_ends = []
            
            for i in range(1, len(burst_indices)):
                if burst_indices[i] != burst_indices[i-1] + 1:
                    burst_ends.append(burst_indices[i-1] + 1)
                    burst_starts.append(burst_indices[i])
            burst_ends.append(burst_indices[-1] + 1)
            
            # Burst 정보 저장
            for start, end in zip(burst_starts, burst_ends):
                bursts.append({
                    'start_time': spike_times_ms[start],
                    'end_time': spike_times_ms[end],
                    'spike_count': end - start + 1,
                    'duration': spike_times_ms[end] - spike_times_ms[start]
                })
            
            # Regular spike 찾기 (burst에 속하지 않는 스파이크들)
            burst_spike_indices = set()
            for start, end in zip(burst_starts, burst_ends):
                burst_spike_indices.update(range(start, end + 1))
            
            for i, spike_time in enumerate(spike_times_ms):
                if i not in burst_spike_indices:
                    regular_spikes.append(spike_time)
            
            pattern_type = "Mixed (burst + regular)"
        else:
            pattern_type = "Regular firing"
            regular_spikes = spike_times_ms.tolist()
        
        return pattern_type, bursts, regular_spikes
    
    def analyze_membrane_dynamics(voltage, time_ms):
        """막전위 동역학 분석"""
        # 기본 통계
        v_mean = np.mean(voltage)
        v_std = np.std(voltage)
        v_min, v_max = np.min(voltage), np.max(voltage)
        
        # Threshold crossing 분석 (대략적인 threshold를 추정)
        threshold = v_mean + 2 * v_std
        crossings = np.where(np.diff(np.signbit(voltage - threshold)))[0]
        
        # Oscillation 분석 (간단한 주파수 분석)
        try:
            from scipy import signal
            freqs, psd = signal.welch(voltage, fs=1000/(time_ms[1]-time_ms[0]), nperseg=min(len(voltage)//4, 256))
            dominant_freq = freqs[np.argmax(psd[1:])] if len(psd) > 1 else 0
        except:
            dominant_freq = 0
        
        return {
            'mean': v_mean,
            'std': v_std,
            'range': (v_min, v_max),
            'threshold_crossings': len(crossings),
            'dominant_frequency': dominant_freq
        }
    
    try:
        # 데이터 확인
        if neuron_group_name not in voltage_monitors:
            print(f"No voltage monitor found for {neuron_group_name}")
            return
        
        if neuron_group_name not in spike_monitors:
            print(f"No spike monitor found for {neuron_group_name}")
            return
        
        # 전압 데이터 추출
        v_monitor = voltage_monitors[neuron_group_name]
        s_monitor = spike_monitors[neuron_group_name]
        
        if len(v_monitor.t) == 0:
            print(f"No voltage data recorded for {neuron_group_name}")
            return
        
        # 시간 범위 적용
        start_time, end_time = analysis_window
        time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
        
        time_ms = v_monitor.t[time_mask] / ms
        voltage = v_monitor.v[neuron_index][time_mask] / mV
        
        if len(time_ms) == 0:
            print(f"No data in specified time window")
            return
        
        # 스파이크 데이터 추출
        spike_times, spike_indices = get_monitor_spikes(s_monitor)
        neuron_spike_mask = spike_indices == neuron_index
        neuron_spike_times = spike_times[neuron_spike_mask]
        spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
        neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
        
        # 분석 수행
        dt_ms = time_ms[1] - time_ms[0] if len(time_ms) > 1 else 1
        voltage_phase, dv_dt_phase = calculate_phase_space(voltage, dt_ms)
        pattern_type, bursts, regular_spikes = detect_firing_patterns(neuron_spike_times_window)
        membrane_stats = analyze_membrane_dynamics(voltage, time_ms)
        
        # 시각화
        fig = plt.figure(figsize=(20, 15))
        
        display_name = display_names.get(neuron_group_name, neuron_group_name) if display_names else neuron_group_name
        fig.suptitle(f'{display_name} Neuron #{neuron_index} - Detailed Single Cell Analysis', 
                    fontsize=18, fontweight='bold')
        
        # 1. Phase Space Plot (왼쪽 상단) - 이미지와 유사한 스타일
        ax1 = plt.subplot(3, 3, (1, 4))  # 2x2 영역 차지
        
        # 2D histogram for phase space
        try:
            hist, xedges, yedges = np.histogram2d(voltage_phase, dv_dt_phase, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax1.imshow(hist.T, extent=extent, origin='lower', cmap='plasma', 
                           aspect='auto', interpolation='bilinear')
            
            # Trajectory overlay
            ax1.plot(voltage_phase, dv_dt_phase, 'white', alpha=0.3, linewidth=0.5)
            
            ax1.set_xlabel('Membrane Potential (mV)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('dV/dt (mV/ms)', fontsize=12, fontweight='bold')
            ax1.set_title('Phase Space (V vs dV/dt)', fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label('Density', rotation=270, labelpad=15)
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Phase space error: {str(e)}', 
                    transform=ax1.transAxes, ha='center', va='center')
        
        # 2. Membrane Potential Time Series (오른쪽 상단)
        ax2 = plt.subplot(3, 3, 3)
        
        ax2.plot(time_ms, voltage, 'b-', linewidth=1.5, alpha=0.8)
        
        # 스파이크 마킹
        for spike_time in neuron_spike_times_window:
            if start_time/ms <= spike_time <= end_time/ms:
                spike_idx = np.argmin(np.abs(time_ms - spike_time))
                if spike_idx < len(voltage):
                    ax2.plot(spike_time, voltage[spike_idx], 'ro', markersize=6, alpha=0.8)
        
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('V (mV)', fontsize=12)
        ax2.set_title('Membrane Potential', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(start_time/ms, end_time/ms)
        
        # 3. Firing Pattern Analysis (오른쪽 중간)
        ax3 = plt.subplot(3, 3, 6)
        
        if len(neuron_spike_times_window) > 0:
            # Raster-style 표시
            for i, spike_time in enumerate(neuron_spike_times_window):
                color = 'red' if any(b['start_time'] <= spike_time <= b['end_time'] for b in bursts) else 'blue'
                ax3.vlines(spike_time, 0, 1, colors=color, linewidth=2, alpha=0.8)
            
            # Burst 구간 표시
            for burst in bursts:
                ax3.axvspan(burst['start_time'], burst['end_time'], alpha=0.3, color='orange')
            
            ax3.set_xlabel('Time (ms)', fontsize=12)
            ax3.set_ylabel('Spikes', fontsize=12)
            ax3.set_title(f'Firing Pattern: {pattern_type}', fontsize=13, fontweight='bold')
            ax3.set_xlim(start_time/ms, end_time/ms)
            ax3.set_ylim(-0.1, 1.1)
            
            # 범례
            ax3.plot([], [], 'r-', label='Burst spikes', linewidth=3)
            ax3.plot([], [], 'b-', label='Regular spikes', linewidth=3)
            ax3.legend(fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No spikes detected', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12)
        
        # 4. ISI Analysis (왼쪽 하단)
        ax4 = plt.subplot(3, 3, 7)
        
        if len(neuron_spike_times_window) > 1:
            isis = np.diff(neuron_spike_times_window)
            
            ax4.hist(isis, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(np.mean(isis), color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {np.mean(isis):.1f}ms')
            ax4.set_xlabel('ISI (ms)', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Inter-Spike Interval Distribution', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient spikes for ISI analysis', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        
        # 5. Burst Analysis (중간 하단)
        ax5 = plt.subplot(3, 3, 8)
        
        if bursts:
            burst_counts = [b['spike_count'] for b in bursts]
            burst_durations = [b['duration'] for b in bursts]
            
            ax5.scatter(burst_durations, burst_counts, alpha=0.7, s=80, c='orange', edgecolors='black')
            ax5.set_xlabel('Burst Duration (ms)', fontsize=12)
            ax5.set_ylabel('Spikes per Burst', fontsize=12)
            ax5.set_title(f'Burst Analysis ({len(bursts)} bursts)', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # 정보 텍스트
            avg_burst_size = np.mean(burst_counts)
            avg_burst_duration = np.mean(burst_durations)
            ax5.text(0.05, 0.95, f'Avg size: {avg_burst_size:.1f}\nAvg duration: {avg_burst_duration:.1f}ms', 
                    transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'No bursts detected', transform=ax5.transAxes, 
                    ha='center', va='center', fontsize=12)
        
        # 6. Statistics Summary (오른쪽 하단)
        ax6 = plt.subplot(3, 3, 9)
        ax6.axis('off')
        
        # 통계 정보 텍스트
        stats_text = f"Single Neuron Analysis Summary\n\n"
        stats_text += f"Neuron: {display_name} #{neuron_index}\n"
        stats_text += f"Analysis window: {start_time/ms:.0f}-{end_time/ms:.0f}ms\n\n"
        
        stats_text += f"Membrane Potential:\n"
        stats_text += f"  Mean: {membrane_stats['mean']:.2f} mV\n"
        stats_text += f"  Std: {membrane_stats['std']:.2f} mV\n"
        stats_text += f"  Range: {membrane_stats['range'][0]:.1f} - {membrane_stats['range'][1]:.1f} mV\n"
        stats_text += f"  Dominant freq: {membrane_stats['dominant_frequency']:.1f} Hz\n\n"
        
        stats_text += f"Firing Activity:\n"
        stats_text += f"  Total spikes: {len(neuron_spike_times_window)}\n"
        stats_text += f"  Pattern: {pattern_type}\n"
        stats_text += f"  Bursts: {len(bursts)}\n"
        stats_text += f"  Regular spikes: {len(regular_spikes)}\n"
        
        if len(neuron_spike_times_window) > 1:
            firing_rate = len(neuron_spike_times_window) / ((end_time - start_time) / second)
            stats_text += f"  Firing rate: {firing_rate:.2f} Hz\n"
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        
        # 결과 출력
        print(f"\n=== Single Neuron Analysis: {display_name} #{neuron_index} ===")
        print(f"Firing pattern: {pattern_type}")
        print(f"Total spikes: {len(neuron_spike_times_window)}")
        print(f"Bursts detected: {len(bursts)}")
        print(f"Membrane potential range: {membrane_stats['range'][0]:.1f} - {membrane_stats['range'][1]:.1f} mV")
        
        # 파일 저장
        if save_plot:
            filename = f'single_neuron_analysis_{neuron_group_name}_{neuron_index}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Single neuron analysis saved to '{filename}'")
        
        try:
            print(f"Single neuron analysis displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying single neuron analysis: {e}")
            
    except Exception as e:
        print(f"Single neuron analysis error: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_multi_neuron_comparison(voltage_monitors, spike_monitors, neuron_group_name,
                                neuron_indices=[0, 1, 2], analysis_window=(0*ms, 2000*ms),
                                display_names=None, save_plot=True):
    """
    같은 그룹 내 여러 뉴런들의 비교 분석
    각 뉴런의 firing pattern과 특성을 비교
    
    Parameters:
    - voltage_monitors: 전압 모니터 딕셔너리
    - spike_monitors: 스파이크 모니터 딕셔너리
    - neuron_group_name: 분석할 뉴런 그룹 이름
    - neuron_indices: 비교할 뉴런 인덱스들
    - analysis_window: 분석 시간 구간
    - display_names: 표시 이름 매핑
    - save_plot: 플롯 저장 여부
    """
    
    try:
        # 데이터 확인
        if neuron_group_name not in voltage_monitors or neuron_group_name not in spike_monitors:
            print(f"No monitors found for {neuron_group_name}")
            return
        
        v_monitor = voltage_monitors[neuron_group_name]
        s_monitor = spike_monitors[neuron_group_name]
        
        start_time, end_time = analysis_window
        time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
        time_ms = v_monitor.t[time_mask] / ms
        
        if len(time_ms) == 0:
            print("No data in specified time window")
            return
        
        # 스파이크 데이터 추출
        spike_times, spike_indices = get_monitor_spikes(s_monitor)
        
        n_neurons = len(neuron_indices)
        colors = plt.cm.Set3(np.linspace(0, 1, n_neurons))
        
        # 시각화
        fig = plt.figure(figsize=(18, 4 * n_neurons))
        display_name = display_names.get(neuron_group_name, neuron_group_name) if display_names else neuron_group_name
        fig.suptitle(f'{display_name} - Multi-Neuron Comparison', fontsize=16, fontweight='bold')
        
        neuron_stats = []
        
        for i, neuron_idx in enumerate(neuron_indices):
            try:
                # 전압 데이터
                voltage = v_monitor.v[neuron_idx][time_mask] / mV
                
                # 해당 뉴런의 스파이크
                neuron_spike_mask = spike_indices == neuron_idx
                neuron_spikes = spike_times[neuron_spike_mask]
                spike_time_mask = (neuron_spikes >= start_time) & (neuron_spikes <= end_time)
                neuron_spikes_window = neuron_spikes[spike_time_mask] / ms
                
                color = colors[i]
                
                # 1. 막전위 trace (상단)
                ax1 = plt.subplot(n_neurons, 3, i*3 + 1)
                ax1.plot(time_ms, voltage, color=color, linewidth=1.5, alpha=0.8)
                
                # 스파이크 마킹
                for spike_time in neuron_spikes_window:
                    spike_idx = np.argmin(np.abs(time_ms - spike_time))
                    if spike_idx < len(voltage):
                        ax1.plot(spike_time, voltage[spike_idx], 'ro', markersize=4)
                
                ax1.set_ylabel(f'Neuron {neuron_idx}\nV (mV)', fontsize=11)
                ax1.set_title(f'Membrane Potential - Neuron {neuron_idx}', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(start_time/ms, end_time/ms)
                
                if i == n_neurons - 1:
                    ax1.set_xlabel('Time (ms)', fontsize=11)
                
                # 2. Phase space (중간)
                ax2 = plt.subplot(n_neurons, 3, i*3 + 2)
                
                if len(voltage) > 1:
                    dt_ms = time_ms[1] - time_ms[0]
                    dv_dt = np.gradient(voltage, dt_ms)
                    
                    ax2.plot(voltage, dv_dt, color=color, alpha=0.6, linewidth=1)
                    ax2.scatter(voltage[::10], dv_dt[::10], c=color, s=10, alpha=0.4)  # 샘플링해서 점 표시
                    
                    ax2.set_xlabel('V (mV)', fontsize=11)
                    ax2.set_ylabel('dV/dt (mV/ms)', fontsize=11)
                    ax2.set_title(f'Phase Space - Neuron {neuron_idx}', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                
                # 3. 스파이크 통계 (오른쪽)
                ax3 = plt.subplot(n_neurons, 3, i*3 + 3)
                
                if len(neuron_spikes_window) > 1:
                    isis = np.diff(neuron_spikes_window)
                    
                    ax3.hist(isis, bins=15, alpha=0.7, color=color, edgecolor='black')
                    ax3.axvline(np.mean(isis), color='red', linestyle='--', linewidth=2)
                    ax3.set_xlabel('ISI (ms)', fontsize=11)
                    ax3.set_ylabel('Count', fontsize=11)
                    ax3.set_title(f'ISI Distribution - Neuron {neuron_idx}', fontsize=12, fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    
                    # 통계 저장
                    firing_rate = len(neuron_spikes_window) / ((end_time - start_time) / second)
                    mean_isi = np.mean(isis)
                    cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else 0
                    
                    neuron_stats.append({
                        'index': neuron_idx,
                        'spike_count': len(neuron_spikes_window),
                        'firing_rate': firing_rate,
                        'mean_isi': mean_isi,
                        'cv_isi': cv_isi,
                        'v_mean': np.mean(voltage),
                        'v_std': np.std(voltage)
                    })
                else:
                    ax3.text(0.5, 0.5, 'No spikes', transform=ax3.transAxes, 
                            ha='center', va='center', fontsize=12)
                    
                    neuron_stats.append({
                        'index': neuron_idx,
                        'spike_count': 0,
                        'firing_rate': 0,
                        'mean_isi': 0,
                        'cv_isi': 0,
                        'v_mean': np.mean(voltage),
                        'v_std': np.std(voltage)
                    })
                
            except Exception as e:
                print(f"Error analyzing neuron {neuron_idx}: {e}")
                continue
        
        plt.tight_layout()
        
        # 통계 출력
        print(f"\n=== Multi-Neuron Comparison: {display_name} ===")
        print("Neuron | Spikes | Rate(Hz) | Mean ISI(ms) | CV ISI | V_mean(mV) | V_std(mV)")
        print("-" * 75)
        
        for stats in neuron_stats:
            print(f"{stats['index']:6d} | {stats['spike_count']:6d} | {stats['firing_rate']:8.2f} | "
                  f"{stats['mean_isi']:11.2f} | {stats['cv_isi']:6.3f} | {stats['v_mean']:10.2f} | {stats['v_std']:9.2f}")
        
        # 파일 저장
        if save_plot:
            filename = f'multi_neuron_comparison_{neuron_group_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nMulti-neuron comparison saved to '{filename}'")
        
        try:
            print("Multi-neuron comparison displayed. Close the plot window to continue...")
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying multi-neuron comparison: {e}")
            
    except Exception as e:
        print(f"Multi-neuron comparison error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_neuron_stimulus_pattern(voltage_monitors, spike_monitors, stimulus_config, 
                                neuron_group_name, neuron_index=0, 
                                analysis_window=(0*ms, 5000*ms), 
                                display_names=None, save_plot=True):
    """
    뉴런의 membrane potential과 stimulus 패턴을 함께 시각화 (A 패널 스타일)
    
    Parameters:
    - voltage_monitors: 전압 모니터 딕셔너리
    - spike_monitors: 스파이크 모니터 딕셔너리  
    - stimulus_config: 스티뮬러스 설정
    - neuron_group_name: 분석할 뉴런 그룹 이름
    - neuron_index: 분석할 뉴런 인덱스 (기본값: 0)
    - analysis_window: 분석 시간 구간
    - display_names: 표시 이름 매핑
    - save_plot: 플롯 저장 여부
    """
    
    try:
        # 데이터 유효성 확인
        if neuron_group_name not in voltage_monitors:
            print(f"No voltage monitor found for {neuron_group_name}")
            return
            
        if neuron_group_name not in spike_monitors:
            print(f"No spike monitor found for {neuron_group_name}")
            return
        
        # 전압 및 스파이크 데이터 추출
        v_monitor = voltage_monitors[neuron_group_name]
        s_monitor = spike_monitors[neuron_group_name]
        
        if len(v_monitor.t) == 0:
            print(f"No voltage data recorded for {neuron_group_name}")
            return
        
        # 시간 범위 적용
        start_time, end_time = analysis_window
        time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
        
        time_ms = v_monitor.t[time_mask] / ms
        voltage = v_monitor.v[neuron_index][time_mask] / mV
        
        if len(time_ms) == 0:
            print(f"No data in specified time window")
            return
        
        # 스파이크 데이터 추출  
        spike_times, spike_indices = get_monitor_spikes(s_monitor)
        neuron_spike_mask = spike_indices == neuron_index
        neuron_spike_times = spike_times[neuron_spike_mask]
        spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
        neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
        
        # 스티뮬러스 패턴 생성
        stimulus_pA = np.zeros_like(time_ms)
        
        if stimulus_config and stimulus_config.get('enabled', False):
            stim_start = stimulus_config.get('start_time', 0)
            stim_duration = stimulus_config.get('duration', 0)
            stim_end = stim_start + stim_duration
            
            # 스티뮬러스 구간에서 amplitude 설정 (pA)
            if neuron_group_name in stimulus_config.get('rates', {}):
                # stimulus rate를 approximate amplitude로 변환 (임의의 스케일링)
                stim_rate = stimulus_config['rates'][neuron_group_name]
                stimulus_amplitude = stim_rate * 0.025  # Hz to pA 근사 변환
            else:
                stimulus_amplitude = 20  # 기본값 20 pA
                
            # 스티뮬러스 구간에서 패턴 적용
            stim_mask = (time_ms >= stim_start) & (time_ms <= stim_end)
            stimulus_pA[stim_mask] = stimulus_amplitude
        else:
            # 스티뮬러스가 비활성화된 경우 기본 패턴
            stimulus_amplitude = 0
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                      gridspec_kw={'height_ratios': [1, 2]})
        
        # 상단: Stimulus pattern
        ax1.plot(time_ms, stimulus_pA, 'k-', linewidth=2)
        ax1.set_ylabel('자극 (pA)', fontsize=12, fontweight='bold')
        ax1.set_title('A. 뉴런 막전위와 자극 패턴', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, max(stimulus_pA) + 10 if max(stimulus_pA) > 0 else 25)
        
        # 하단: Membrane potential
        ax2.plot(time_ms, voltage, 'b-', linewidth=1.5, alpha=0.8)
        
        # 스파이크 표시
        if len(neuron_spike_times_window) > 0:
            spike_heights = []
            for spike_time in neuron_spike_times_window:
                # 각 스파이크 시간에서 가장 가까운 voltage 값 찾기
                time_idx = np.argmin(np.abs(time_ms - spike_time))
                if time_idx < len(voltage):
                    spike_heights.append(voltage[time_idx])
                else:
                    spike_heights.append(0)
            
            ax2.scatter(neuron_spike_times_window, spike_heights, 
                       color='red', s=30, marker='|', linewidth=2, alpha=0.8)
        
        display_name = display_names.get(neuron_group_name, neuron_group_name) if display_names else neuron_group_name
        ax2.set_ylabel('막전위 (mV)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('시간 (ms)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 스티뮬러스 구간 표시 (음영)
        if stimulus_config and stimulus_config.get('enabled', False):
            stim_start = stimulus_config.get('start_time', 0)
            stim_duration = stimulus_config.get('duration', 0)
            stim_end = stim_start + stim_duration
            
            ax1.axvspan(stim_start, stim_end, alpha=0.2, color='red', label='자극 구간')
            ax2.axvspan(stim_start, stim_end, alpha=0.2, color='red', label='자극 구간')
        
        # 범례 및 정보 표시
        info_text = f'{display_name} 뉴런 #{neuron_index}'
        if len(neuron_spike_times_window) > 0:
            info_text += f'\n스파이크 수: {len(neuron_spike_times_window)}개'
            
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 파일 저장
        if save_plot:
            filename = f'neuron_stimulus_pattern_{neuron_group_name}_neuron{neuron_index}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"뉴런-자극 패턴 그래프 저장됨: {filename}")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"그래프 표시 중 오류: {e}")
            plt.close()
            
        print(f"\n=== {display_name} 뉴런 #{neuron_index} 분석 완료 ===")
        print(f"분석 구간: {start_time/ms:.0f}-{end_time/ms:.0f}ms")
        print(f"스파이크 수: {len(neuron_spike_times_window)}개")
        
        if stimulus_config and stimulus_config.get('enabled', False):
            print(f"자극 구간: {stim_start}-{stim_end}ms")
            print(f"자극 강도: {stimulus_amplitude:.1f} pA")
            
    except Exception as e:
        print(f"뉴런-자극 패턴 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def plot_multi_neuron_stimulus_overview(voltage_monitors, spike_monitors, stimulus_config,
                                      target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
                                      neurons_per_group=3, analysis_window=(0*ms, 10000*ms),
                                      display_names=None, save_plot=True):
    """
    Multiple neurons from all groups with stimulus pattern overview
    
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
        
        # Create figure with stimulus at top and neurons below
        fig_height = 3 + total_neurons * 1.2  # 3 for stimulus + 1.2 per neuron
        fig, axes = plt.subplots(total_neurons + 1, 1, figsize=(16, fig_height), 
                                sharex=True, gridspec_kw={'height_ratios': [1] + [1]*total_neurons})
        
        # Extract time range
        start_time, end_time = analysis_window
        
        # Get time vector from first available monitor
        first_group = available_groups[0]
        v_monitor = voltage_monitors[first_group]
        time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
        time_ms = v_monitor.t[time_mask] / ms
        
        # Top panel: Stimulus pattern
        stimulus_pA = np.zeros_like(time_ms)
        
        if stimulus_config and stimulus_config.get('enabled', False):
            stim_start = stimulus_config.get('start_time', 0)
            stim_duration = stimulus_config.get('duration', 0)
            stim_end = stim_start + stim_duration
            
            # Use a representative stimulus amplitude
            stimulus_amplitude = 25  # pA
            stim_mask = (time_ms >= stim_start) & (time_ms <= stim_end)
            stimulus_pA[stim_mask] = stimulus_amplitude
        
        # Plot stimulus
        axes[0].plot(time_ms, stimulus_pA, 'k-', linewidth=2)
        axes[0].set_ylabel('Stimulus (pA)', fontsize=12, fontweight='bold')
        axes[0].set_title('Multi-Neuron Membrane Potential and Stimulus Pattern Overview', 
                         fontsize=16, fontweight='bold', pad=20)
        axes[0].set_ylim(-5, max(stimulus_pA) + 10 if max(stimulus_pA) > 0 else 30)
        
        # Add stimulus shading if enabled
        if stimulus_config and stimulus_config.get('enabled', False):
            axes[0].axvspan(stim_start, stim_end, alpha=0.15, color='orange', label='Stimulus Period')
            axes[0].legend(loc='upper right', fontsize=9)
        
        # Plot neurons
        ax_idx = 1
        neuron_labels = []
        
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
                ax = axes[ax_idx]
                
                # Extract voltage data
                voltage = v_monitor.v[neuron_idx][time_mask] / mV
                
                # Plot membrane potential
                ax.plot(time_ms, voltage, 'b-', linewidth=1.2, alpha=0.8)
                
                # Get spikes for this neuron
                spike_times, spike_indices = get_monitor_spikes(s_monitor)
                neuron_spike_mask = spike_indices == neuron_idx
                neuron_spike_times = spike_times[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
                neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
                
                # Mark spikes
                if len(neuron_spike_times_window) > 0:
                    spike_heights = []
                    for spike_time in neuron_spike_times_window:
                        time_idx = np.argmin(np.abs(time_ms - spike_time))
                        if time_idx < len(voltage):
                            spike_heights.append(voltage[time_idx])
                        else:
                            spike_heights.append(0)
                    
                    ax.scatter(neuron_spike_times_window, spike_heights, 
                             color='red', s=20, marker='|', linewidth=1.5, alpha=0.9)
                
                # Add stimulus shading to neuron plots
                if stimulus_config and stimulus_config.get('enabled', False):
                    ax.axvspan(stim_start, stim_end, alpha=0.1, color='orange')
                
                # Label and format
                display_name = display_names.get(group_name, group_name) if display_names else group_name
                neuron_label = f'{display_name} #{neuron_idx}'
                neuron_labels.append(neuron_label)
                
                ax.set_ylabel(f'{display_name}\n#{neuron_idx}\n(mV)', fontsize=10, fontweight='bold')
                ax.tick_params(axis='both', labelsize=9)
                
                # Add spike count info
                spike_count = len(neuron_spike_times_window)
                ax.text(0.98, 0.85, f'Spikes: {spike_count}', transform=ax.transAxes,
                       horizontalalignment='right', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax_idx += 1
        
        # Format bottom axis
        axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[-1].tick_params(axis='x', labelsize=10)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            filename = f'multi_neuron_stimulus_overview_{total_neurons}neurons.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Multi-neuron overview saved: {filename}")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Error displaying plot: {e}")
            plt.close()
            
        # Print summary
        print(f"\n=== Multi-Neuron Overview Analysis Complete ===")
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Total neurons plotted: {total_neurons} from {len(available_groups)} groups")
        print(f"Groups included: {', '.join(available_groups)}")
        
        if stimulus_config and stimulus_config.get('enabled', False):
            print(f"Stimulus period: {stim_start}-{stim_end} ms")
            
    except Exception as e:
        print(f"Error in multi-neuron stimulus overview: {e}")
        import traceback
        traceback.print_exc()

def plot_all_neurons_phase_space_overview(voltage_monitors, spike_monitors,
                                        target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
                                        analysis_window=(0*ms, 2000*ms), neurons_per_group=1,
                                        display_names=None, save_plot=True):
    """
    All neuron groups phase space analysis in single overview image
    
    Parameters:
    - voltage_monitors: voltage monitor dictionary
    - spike_monitors: spike monitor dictionary  
    - target_groups: list of neuron groups to analyze
    - analysis_window: analysis time window
    - neurons_per_group: number of neurons per group to analyze
    - display_names: display name mapping
    - save_plot: whether to save plot
    """
    
    def calculate_phase_space(voltage, dt_ms):
        """Calculate phase space (V vs dV/dt)"""
        dV_dt = np.gradient(voltage, dt_ms)
        return voltage, dV_dt
    
    def detect_firing_patterns(spike_times_ms, isi_threshold=100):
        """Classify firing patterns (burst vs regular)"""
        if len(spike_times_ms) < 2:
            return "No spikes", 0, 0
        
        isis = np.diff(spike_times_ms)
        burst_spikes = np.sum(isis < isi_threshold)
        total_spikes = len(spike_times_ms)
        burst_ratio = burst_spikes / total_spikes if total_spikes > 0 else 0
        
        if burst_ratio > 0.3:
            return "Burst", total_spikes, burst_ratio
        elif total_spikes > 5:
            return "Regular", total_spikes, 0
        else:
            return "Sparse", total_spikes, 0
    
    try:
        # Filter available groups
        available_groups = []
        
        for group_name in target_groups:
            if (group_name in voltage_monitors and 
                group_name in spike_monitors and
                len(voltage_monitors[group_name].t) > 0):
                available_groups.append(group_name)
        
        if not available_groups:
            print("No available neuron groups with voltage data")
            return
        
        # Calculate grid layout
        n_groups = len(available_groups)
        cols = min(4, n_groups)  # Maximum 4 columns
        rows = (n_groups + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Analysis parameters
        start_time, end_time = analysis_window
        
        # Process each group
        for idx, group_name in enumerate(available_groups):
            ax = axes[idx]
            
            # Get monitors
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            
            # Select neuron (first available)
            neuron_idx = 0
            
            # Extract voltage data
            time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
            time_ms = v_monitor.t[time_mask] / ms
            voltage = v_monitor.v[neuron_idx][time_mask] / mV
            
            if len(time_ms) < 2:
                ax.text(0.5, 0.5, f'{group_name}\nNo data', 
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            # Calculate phase space
            dt_ms = np.mean(np.diff(time_ms))
            voltage_phase, dV_dt = calculate_phase_space(voltage, dt_ms)
            
            # Create 2D histogram for phase space
            try:
                counts, xedges, yedges = np.histogram2d(voltage_phase, dV_dt, bins=50, density=True)
                
                # Plot phase space heatmap
                im = ax.imshow(counts.T, origin='lower', aspect='auto', cmap='viridis', alpha=0.8,
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                
                # Add trajectory line
                ax.plot(voltage_phase, dV_dt, 'white', alpha=0.3, linewidth=0.5)
                
            except Exception as e:
                # Fallback to simple scatter plot
                ax.scatter(voltage_phase, dV_dt, c=np.arange(len(voltage_phase)), 
                          cmap='viridis', s=1, alpha=0.6)
            
            # Get spike data
            spike_times, spike_indices = get_monitor_spikes(s_monitor)
            neuron_spike_mask = spike_indices == neuron_idx
            neuron_spike_times = spike_times[neuron_spike_mask]
            spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
            neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
            
            # Analyze firing pattern
            firing_pattern, spike_count, burst_ratio = detect_firing_patterns(neuron_spike_times_window)
            
            # Calculate firing rate
            window_duration = (end_time - start_time) / ms / 1000  # convert to seconds
            firing_rate = spike_count / window_duration if window_duration > 0 else 0
            
            # Format labels
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            
            # Set labels and title
            ax.set_xlabel('Membrane Potential (mV)', fontsize=10)
            ax.set_ylabel('dV/dt (mV/ms)', fontsize=10)
            ax.set_title(f'{display_name} Phase Space\n#{neuron_idx}', fontsize=12, fontweight='bold')
            
            # Add statistics text
            stats_text = f'Pattern: {firing_pattern}\nSpikes: {spike_count}\nRate: {firing_rate:.1f} Hz'
            if burst_ratio > 0:
                stats_text += f'\nBurst: {burst_ratio:.2f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Grid and formatting
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=9)
        
        # Hide unused subplots
        for idx in range(len(available_groups), len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle('Phase Space Analysis Overview - All Neuron Groups', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Layout
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            filename = f'phase_space_overview_all_neurons.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Phase space overview saved: {filename}")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Error displaying plot: {e}")
            plt.close()
            
        # Print summary
        print(f"\n=== Phase Space Analysis Overview Complete ===")
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Groups analyzed: {len(available_groups)}")
        
        # Print individual group results
        for group_name in available_groups:
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            
            spike_times, spike_indices = get_monitor_spikes(s_monitor)
            neuron_spike_mask = spike_indices == 0  # first neuron
            neuron_spike_times = spike_times[neuron_spike_mask]
            spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
            spike_count = np.sum(spike_time_mask)
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"  {display_name}: {spike_count} spikes")
            
        print("\n=== Phase Space Interpretation Guide ===")
        print("• Phase Space shows V (membrane potential) vs dV/dt (voltage change rate)")
        print("• Spiral patterns → Converging to stable state (Regular spiking)")
        print("• Limit cycles → Periodic oscillations (Burst firing)")  
        print("• Dense areas (bright colors) → States where neuron spends more time")
        print("• Trajectory direction → Shows voltage dynamics flow")
        print("• Center clustering → Around resting potential")
        
    except Exception as e:
        print(f"Error in phase space overview: {e}")
        import traceback
        traceback.print_exc()

def plot_phase_space_only_overview(voltage_monitors, spike_monitors,
                                   target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
                                   analysis_window=(0*ms, 2000*ms), display_names=None, save_plot=True):
    """
    Clean phase space plots only - all neuron groups in single image
    
    Parameters:
    - voltage_monitors: voltage monitor dictionary
    - spike_monitors: spike monitor dictionary  
    - target_groups: list of neuron groups to analyze
    - analysis_window: analysis time window
    - display_names: display name mapping
    - save_plot: whether to save plot
    """
    
    def calculate_phase_space(voltage, dt_ms):
        """Calculate phase space (V vs dV/dt)"""
        dV_dt = np.gradient(voltage, dt_ms)
        return voltage, dV_dt
    
    try:
        # Filter available groups with detailed debugging
        available_groups = []
        
        print(f"Checking voltage monitors for groups: {target_groups}")
        print(f"Available voltage monitors: {list(voltage_monitors.keys())}")
        print(f"Available spike monitors: {list(spike_monitors.keys())}")
        
        for group_name in target_groups:
            has_voltage = group_name in voltage_monitors
            has_spike = group_name in spike_monitors
            has_data = has_voltage and len(voltage_monitors[group_name].t) > 0 if has_voltage else False
            
            print(f"  {group_name}: voltage={has_voltage}, spike={has_spike}, data={has_data}")
            
            if has_voltage and has_spike and has_data:
                available_groups.append(group_name)
        
        print(f"Final available groups for phase space: {available_groups}")
        
        if not available_groups:
            print("No available neuron groups with voltage data")
            return
        
        # Calculate grid layout (4 columns max for clean look)
        n_groups = len(available_groups)
        cols = min(4, n_groups)
        rows = (n_groups + cols - 1) // cols
        
        # Create figure with clean layout
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Analysis parameters
        start_time, end_time = analysis_window
        
        # Process each group - clean phase space only
        for idx, group_name in enumerate(available_groups):
            ax = axes[idx]
            
            # Get monitors
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            
            # Extract voltage data for first neuron
            neuron_idx = 0
            time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
            time_ms = v_monitor.t[time_mask] / ms
            voltage = v_monitor.v[neuron_idx][time_mask] / mV
            
            if len(time_ms) < 2:
                ax.text(0.5, 0.5, f'{group_name}\nNo Data', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Calculate phase space
            dt_ms = np.mean(np.diff(time_ms))
            voltage_phase, dV_dt = calculate_phase_space(voltage, dt_ms)
            
            # Create clean 2D histogram phase space
            try:
                # Calculate 2D histogram
                counts, xedges, yedges = np.histogram2d(voltage_phase, dV_dt, bins=40, density=True)
                
                # Apply smoothing to the histogram
                from scipy.ndimage import gaussian_filter
                counts_smooth = gaussian_filter(counts, sigma=0.8)
                
                # Plot phase space heatmap with viridis colormap
                im = ax.imshow(counts_smooth.T, origin='lower', aspect='auto', 
                              cmap='viridis', alpha=0.9,
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                
                # Add subtle trajectory line
                ax.plot(voltage_phase, dV_dt, 'white', alpha=0.4, linewidth=0.8)
                
            except Exception as e:
                # Fallback to scatter plot
                ax.scatter(voltage_phase, dV_dt, c=np.arange(len(voltage_phase)), 
                          cmap='viridis', s=0.8, alpha=0.7)
            
            # Get spike count for this neuron
            spike_times, spike_indices = get_monitor_spikes(s_monitor)
            neuron_spike_mask = spike_indices == neuron_idx
            neuron_spike_times = spike_times[neuron_spike_mask]
            spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
            spike_count = np.sum(spike_time_mask)
            
            # Clean labeling
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            
            # Set labels and title
            ax.set_xlabel('V (mV)', fontsize=10, fontweight='bold')
            ax.set_ylabel('dV/dt (mV/ms)', fontsize=10, fontweight='bold')
            ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
            
            # Add minimal spike count info
            ax.text(0.95, 0.95, f'{spike_count}', transform=ax.transAxes,
                   horizontalalignment='right', verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Clean formatting
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(True, alpha=0.2)
        
        # Hide unused subplots
        for idx in range(len(available_groups), len(axes)):
            axes[idx].set_visible(False)
        
        # Clean overall title
        fig.suptitle('Phase Space Analysis - Neural Dynamics Overview', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Tight layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save plot
        if save_plot:
            filename = f'phase_space_clean_overview.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Phase space clean overview saved: {filename}")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Error displaying plot: {e}")
            plt.close()
            
        # Minimal summary
        print(f"\n=== Phase Space Clean Overview Complete ===")
        print(f"Analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms | Groups: {len(available_groups)}")
        print("Phase Space: V (membrane potential) vs dV/dt (voltage rate)")
        
    except Exception as e:
        print(f"Error in phase space clean overview: {e}")
        import traceback
        traceback.print_exc()