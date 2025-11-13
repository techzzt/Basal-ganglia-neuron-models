#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import *
import numpy as np

class Thalamus:
    def __init__(self, mu_max=1000*Hz, mu_snr_max=2000*Hz, filter_tau=50*ms, window_ms=200.0, 
                 cortex_baselines=None, mu_min=10*Hz, snr_min=20*Hz):

        self.mu_max = mu_max
        self.mu_snr_max = mu_snr_max
        self.mu_min = mu_min  
        self.snr_min = snr_min 
        self.cortex_baselines = cortex_baselines if cortex_baselines is not None else {}
        self.current_lambda_cortex = {}
        
        self.filter_tau = float(filter_tau / ms)
        self.smoothed_snr_rate = None
        self.smoothing_enabled = True
        
        self.window_ms = float(window_ms)
        self.rate_history = []
        self.time_history = []

    # Smooth SNr rate using exponential filter    
    def smooth_snr_rate(self, snr_rate_raw, dt_ms):
        if not self.smoothing_enabled:
            return snr_rate_raw
            
        if self.smoothed_snr_rate is None:
            self.smoothed_snr_rate = float(snr_rate_raw / Hz)
            return snr_rate_raw
        
        dt = float(dt_ms)
        tau = self.filter_tau
        alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        
        rate_raw_hz = float(snr_rate_raw / Hz)
        rate_smooth_prev = self.smoothed_snr_rate
        
        rate_smooth_new = rate_smooth_prev * (1.0 - alpha) + rate_raw_hz * alpha
        self.smoothed_snr_rate = rate_smooth_new
        
        return rate_smooth_new * Hz
    
    # Smooth SNr rate using sliding window average    
    def smooth_snr_rate_window(self, snr_rate_raw, current_time_ms):
        if not self.smoothing_enabled:
            return snr_rate_raw
            
        rate_hz = float(snr_rate_raw / Hz)
        self.rate_history.append(rate_hz)
        self.time_history.append(current_time_ms)
        
        cutoff_time = current_time_ms - self.window_ms
        while len(self.time_history) > 0 and self.time_history[0] < cutoff_time:
            self.time_history.pop(0)
            self.rate_history.pop(0)
        
        if len(self.rate_history) == 0:
            return snr_rate_raw
        
        avg_rate = np.mean(self.rate_history)
        return avg_rate * Hz
        
    def calculate_thalamus_rate(self, snr_rate, baseline=None):

        snr_min_hz = float(self.snr_min / Hz)
        snr_max_hz = float(self.mu_snr_max / Hz)
        snr_rate_hz = float(snr_rate / Hz)
        
        if snr_max_hz <= snr_min_hz:
            inhibition_index = 1.0 if snr_rate_hz >= snr_max_hz else 0.0
        else:
            inhibition_index = (snr_rate_hz - snr_min_hz) / (snr_max_hz - snr_min_hz)
        
        inhibition_index = min(1.0, max(0.0, inhibition_index))
        
        if baseline is not None:
            group_baseline = baseline
        else:
            group_baseline = self.mu_max

        thalamus_rate = group_baseline - (group_baseline - self.mu_min) * inhibition_index
        
        return max(self.mu_min, thalamus_rate)
    
    def update_cortex_lambda(self, cortex_groups, snr_rate_raw, dt_ms=None, current_time_ms=None):

        if dt_ms is not None:
            snr_rate_filtered = self.smooth_snr_rate(snr_rate_raw, dt_ms)
        elif current_time_ms is not None:
            snr_rate_filtered = self.smooth_snr_rate_window(snr_rate_raw, current_time_ms)
        else:
            snr_rate_filtered = snr_rate_raw

        for group_name, group in cortex_groups.items():
            if group_name.startswith('Cortex_'):
                baseline = self.cortex_baselines.get(group_name, self.mu_max)
                thalamus_rate = self.calculate_thalamus_rate(snr_rate_filtered, baseline=baseline)
                
                self.current_lambda_cortex[group_name] = thalamus_rate
                
                if hasattr(group, 'rate'):
                    group.rate = thalamus_rate
                elif hasattr(group, 'firing_rate'):
                    group.firing_rate = thalamus_rate
    
    def get_current_lambda(self):
        return self.current_lambda_cortex.copy()
    
    def get_thalamus_parameters(self):
        return {
            'mu_max': self.mu_max,
            'mu_snr_max': self.mu_snr_max,
            'mu_min': self.mu_min,
            'snr_min': self.snr_min
        }

