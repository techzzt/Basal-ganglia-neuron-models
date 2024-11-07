import matplotlib.pyplot as plt
import os
from brian2 import *
from datetime import datetime

def plot_raster(spike_monitors):
    """각 뉴런 그룹의 개별 raster plot 생성 및 저장"""
    try:
        n_plots = len(spike_monitors)
        
        # 그래프 생성
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # 각 뉴런 그룹의 spike를 plot
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            axes[i].scatter(monitor.t/ms, monitor.i, s=1)
            axes[i].set_title(f'{name} Raster Plot')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(0, 1000)
        
        axes[-1].set_xlabel('Time (ms)')
        plt.subplots_adjust(hspace=0.5)
        
        # 저장 디렉토리 생성
        save_dir = 'results/raster_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 현재 시간을 파일명에 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'raster_plot_{timestamp}.png')
        
        # 파일 저장
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Raster plot saved: {filename}")
        
        # 그래프 표시
        plt.show()
        
    except Exception as e:
        print(f"Raster plot 생성 중 오류 발생: {str(e)}")