# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import pickle
import h5py
import numpy as np
import json
import os
from datetime import datetime
from brian2 import ms, second

import gzip

# save simulation results
class SimulationDataStorage:
    
    def __init__(self, storage_format='hdf5'):
        """
        Args:
            storage_format: 'hdf5', 'pickle', 'netcdf' 
        """
        self.storage_format = storage_format
        
    def save_simulation_results(self, results, config_file, analysis_start_time, analysis_end_time, 
                              output_file=None, include_voltage=False):
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if 'normal' in config_file.lower():
                output_file = f"normal_results_{timestamp}.{self.storage_format}"
            elif 'dop' in config_file.lower():
                output_file = f"pd_results_{timestamp}.{self.storage_format}"
            else:
                output_file = f"simulation_results_{timestamp}.{self.storage_format}"
        
        if self.storage_format == 'hdf5':
            return self._save_hdf5(results, config_file, analysis_start_time, analysis_end_time, 
                                 output_file, include_voltage)
        elif self.storage_format == 'pickle':
            return self._save_pickle(results, config_file, analysis_start_time, analysis_end_time, 
                                   output_file, include_voltage)
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
    
    def _save_hdf5(self, results, config_file, analysis_start_time, analysis_end_time, 
                   output_file, include_voltage):

        try:
            with h5py.File(output_file, 'w') as f:
                meta_group = f.create_group('metadata')
                meta_group.attrs['config_file'] = os.path.basename(config_file)
                meta_group.attrs['start_time_ms'] = int(analysis_start_time/ms)
                meta_group.attrs['end_time_ms'] = int(analysis_end_time/ms)
                meta_group.attrs['default_bin_size_ms'] = 10
                meta_group.attrs['created_at'] = datetime.now().isoformat()
                meta_group.attrs['brian2_version'] = '2.5.2' 

                spike_group = f.create_group('spike_monitors')
                for group_name, monitor in results['spike_monitors'].items():
                    if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
                        group_data = spike_group.create_group(group_name)
                        group_data.create_dataset('t_ms', data=np.array(monitor.t / ms, dtype=float), 
                                                compression='gzip', compression_opts=9)
                        group_data.create_dataset('i', data=np.array(monitor.i, dtype=int), 
                                                compression='gzip', compression_opts=9)
                        group_data.attrs['N'] = int(monitor.source.N)
                        group_data.attrs['total_spikes'] = len(monitor.t)
                
                if include_voltage and 'voltage_monitors' in results:
                    voltage_group = f.create_group('voltage_monitors')
                    for group_name, monitor in results['voltage_monitors'].items():
                        if hasattr(monitor, 't') and hasattr(monitor, 'v') and len(monitor.t) > 0:
                            group_data = voltage_group.create_group(group_name)
                            group_data.create_dataset('t_ms', data=np.array(monitor.t / ms, dtype=float), 
                                                    compression='gzip', compression_opts=9)
                            group_data.create_dataset('v_mV', data=np.array(monitor.v / mV, dtype=float), 
                                                    compression='gzip', compression_opts=9)
                            group_data.attrs['N'] = int(monitor.source.N)
                            group_data.attrs['sampling_rate_hz'] = 1000.0 / (monitor.t[1] - monitor.t[0]) if len(monitor.t) > 1 else 0

                f.create_dataset('group_names', data=list(results['spike_monitors'].keys()), 
                               dtype=h5py.special_dtype(vlen=str))
            
            print(f"Results saved to '{output_file}' (HDF5 format)")
            return output_file
            
        except Exception as e:
            print(f"Error saving HDF5 results: {e}")
            return None
    
    def _save_pickle(self, results, config_file, analysis_start_time, analysis_end_time, 
                    output_file, include_voltage):

        try:            
            save_data = {
                'metadata': {
                    'config_file': os.path.basename(config_file),
                    'start_time_ms': int(analysis_start_time/ms),
                    'end_time_ms': int(analysis_end_time/ms),
                    'default_bin_size_ms': 10,
                    'created_at': datetime.now().isoformat(),
                    'storage_format': 'pickle_compressed'
                },
                'groups': list(results['spike_monitors'].keys())
            }
            
            for group_name, monitor in results['spike_monitors'].items():
                if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
                    save_data[f'spike_monitors_{group_name}'] = {
                        't_ms': np.array(monitor.t / ms, dtype=float),
                        'i': np.array(monitor.i, dtype=int),
                        'N': int(monitor.source.N),
                        'total_spikes': len(monitor.t)
                    }
            
            if include_voltage and 'voltage_monitors' in results:
                for group_name, monitor in results['voltage_monitors'].items():
                    if hasattr(monitor, 't') and hasattr(monitor, 'v') and len(monitor.t) > 0:
                        save_data[f'voltage_monitors_{group_name}'] = {
                            't_ms': np.array(monitor.t / ms, dtype=float),
                            'v_mV': np.array(monitor.v / mV, dtype=float),
                            'N': int(monitor.source.N),
                            'sampling_rate_hz': 1000.0 / (monitor.t[1] - monitor.t[0]) if len(monitor.t) > 1 else 0
                        }
            
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Results saved to '{output_file}' (Compressed Pickle format)")
            return output_file
            
        except Exception as e:
            print(f"Error saving pickle results: {e}")
            return None
    
    def load_simulation_results(self, file_path):

        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            return self._load_hdf5(file_path)
        elif file_path.endswith('.pkl') or file_path.endswith('.pkl.gz'):
            return self._load_pickle(file_path)
        elif self.storage_format == 'hdf5':
            return self._load_hdf5(file_path)
        elif self.storage_format == 'pickle':
            return self._load_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _load_hdf5(self, file_path):

        try:
            with h5py.File(file_path, 'r') as f:
                metadata = dict(f['metadata'].attrs)
                spike_monitors = {}
                if 'spike_monitors' in f:
                    for group_name in f['spike_monitors'].keys():
                        group_data = f['spike_monitors'][group_name]
                        spike_monitors[group_name] = self._create_mock_spike_monitor(
                            group_data['t_ms'][:], group_data['i'][:], group_data.attrs['N']
                        )
                
                voltage_monitors = {}
                if 'voltage_monitors' in f:
                    for group_name in f['voltage_monitors'].keys():
                        group_data = f['voltage_monitors'][group_name]
                        voltage_monitors[group_name] = self._create_mock_voltage_monitor(
                            group_data['t_ms'][:], group_data['v_mV'][:], group_data.attrs['N']
                        )
                
                return {
                    'metadata': metadata,
                    'spike_monitors': spike_monitors,
                    'voltage_monitors': voltage_monitors
                }
                
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            return None
    
    def _load_pickle(self, file_path):
        try:
            
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            spike_monitors = {}
            for group_name in data['groups']:
                spike_data = data[f'spike_monitors_{group_name}']
                spike_monitors[group_name] = self._create_mock_spike_monitor(
                    spike_data['t_ms'], spike_data['i'], spike_data['N']
                )
            
            voltage_monitors = {}
            for key, value in data.items():
                if key.startswith('voltage_monitors_'):
                    group_name = key.replace('voltage_monitors_', '')
                    voltage_monitors[group_name] = self._create_mock_voltage_monitor(
                        value['t_ms'], value['v_mV'], value['N']
                    )
            
            metadata = data.get('metadata', data.get('meta', {}))
            
            return {
                'metadata': metadata,
                'spike_monitors': spike_monitors,
                'voltage_monitors': voltage_monitors
            }
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None
    
    def _create_mock_spike_monitor(self, t_ms, i, N):
        class MockSpikeMonitor:
            def __init__(self, t_ms, i, N):
                self.t = t_ms * ms
                self.i = i
                self.source = type('obj', (object,), {'N': N})()
        
        return MockSpikeMonitor(t_ms, i, N)
    
    def _create_mock_voltage_monitor(self, t_ms, v_mV, N):
        class MockVoltageMonitor:
            def __init__(self, t_ms, v_mV, N):
                self.t = t_ms * ms
                self.v = v_mV * mV
                self.source = type('obj', (object,), {'N': N})()
        
        return MockVoltageMonitor(t_ms, v_mV, N)
    
    def get_file_info(self, file_path):
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                print(f"File: {file_path}")
                print(f"Groups: {list(f.keys())}")
                if 'metadata' in f:
                    print(f"Metadata: {dict(f['metadata'].attrs)}")
                if 'spike_monitors' in f:
                    print(f"Spike monitor groups: {list(f['spike_monitors'].keys())}")
        else:
            data = self.load_simulation_results(file_path)
            if data:
                print(f"File: {file_path}")
                print(f"Metadata: {data['metadata']}")
                print(f"Spike monitor groups: {list(data['spike_monitors'].keys())}")

def save_simulation_results(results, config_file, analysis_start_time, analysis_end_time, 
                          output_file=None, storage_format='hdf5', include_voltage=False):
    storage = SimulationDataStorage(storage_format)
    return storage.save_simulation_results(results, config_file, analysis_start_time, 
                                         analysis_end_time, output_file, include_voltage)

def load_simulation_results(file_path):
    storage = SimulationDataStorage()
    return storage.load_simulation_results(file_path)





