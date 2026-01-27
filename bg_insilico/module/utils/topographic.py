#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class TopographicRegion:
    MOTOR = 0
    ASSOCIATIVE = 1
    LIMBIC = 2
    
    NAMES = ['motor', 'associative', 'limbic']
    
    @staticmethod
    def get_name(region_id: int) -> str:
        return TopographicRegion.NAMES[region_id]


class TopographicMapper:
    
    def __init__(self, overlap_ratio: float = 0.3, distant_ratio: float = 0.1):

        self.overlap_ratio = overlap_ratio
        self.distant_ratio = distant_ratio
        self.region_map: Dict[str, Dict[int, np.ndarray]] = {} 
        self._topo_cache: Dict[Tuple[str, str, float, bool, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._distance_cache: Dict[Tuple[str, str, int, float, Optional[int], int, int], Tuple[np.ndarray, np.ndarray]] = {}
        
    def divide_population(self, group_name: str, total_neurons: int) -> Dict[int, np.ndarray]:

        neurons_per_region = total_neurons // 3
        remainder = total_neurons % 3
        
        regions: Dict[int, np.ndarray] = {}
        start_idx = 0
        
        for region_id in range(3):
            n_neurons = neurons_per_region + (1 if region_id < remainder else 0)
            regions[region_id] = np.arange(start_idx, start_idx + n_neurons)
            start_idx += n_neurons
        
        self.region_map[group_name] = regions
        return regions
    
    def get_region_indices(self, group_name: str, region_id: int) -> np.ndarray:
        if group_name not in self.region_map:
            raise ValueError(f"Group {group_name} not registered")
        if region_id not in self.region_map[group_name]:
             raise ValueError(f"Region {region_id} not in group {group_name}")
        return self.region_map[group_name][region_id]

    def _get_connection_ratio(self, pre_region: int, post_region: int, is_feedback: bool) -> float:
        
        if is_feedback:
            return 1.0 if pre_region == post_region else 0.0
        
        region_distance = abs(pre_region - post_region)
        
        if region_distance == 0:
            return 1.0
        elif region_distance == 1:
            return self.overlap_ratio
        elif region_distance == 2:
            return self.distant_ratio
        
        return 0.0 
    
    def create_topographic_connection_indices(self,
                                             pre_group: str,
                                             post_group: str,
                                             original_connectivity: float,
                                             is_feedback: bool = False,
                                             seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        try:
            pre_total = sum(len(indices) for indices in self.region_map[pre_group].values())
            post_total = sum(len(indices) for indices in self.region_map[post_group].values())
        except KeyError as e:
            raise ValueError(f"Group {e} not registered in region map.")

        cache_key = (pre_group, post_group, round(original_connectivity, 6), bool(is_feedback), pre_total, post_total)
        if cache_key in self._topo_cache:
            i_cached, j_cached = self._topo_cache[cache_key]
            return i_cached.copy(), j_cached.copy()

        total_target = int(round(max(0.0, original_connectivity) * pre_total * post_total))
        if total_target <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        blocks = []
        region_normalization = {}
        for pre_region in range(3):
            pre_idx = self.get_region_indices(pre_group, pre_region)
            pre_size = len(pre_idx)
            if pre_size == 0:
                continue
            
            region_weight_sum = 0.0
            for post_region in range(3):
                post_idx = self.get_region_indices(post_group, post_region)
                topo_ratio = self._get_connection_ratio(pre_region, post_region, is_feedback)
                if topo_ratio <= 0.0:
                    continue
                n_pairs = pre_size * len(post_idx)
                if n_pairs == 0:
                    continue
                weight = topo_ratio * len(post_idx)
                region_weight_sum += weight
                blocks.append((pre_idx, post_idx, weight, n_pairs, pre_region))
            
            if region_weight_sum > 0:
                region_normalization[pre_region] = region_weight_sum

        if not blocks or not region_normalization:
            return np.array([], dtype=int), np.array([], dtype=int)

        alloc = []
        pre_region_alloc = {r: int(round(original_connectivity * len(self.get_region_indices(pre_group, r)) * post_total)) for r in range(3) if r in region_normalization}
        total_alloc = sum(pre_region_alloc.values())
        
        if total_alloc != total_target:
            diff = total_target - total_alloc
            if diff != 0:
                for r in sorted(pre_region_alloc.keys(), key=lambda x: -pre_region_alloc[x]):
                    if diff == 0:
                        break
                    adjust = 1 if diff > 0 else -1
                    pre_region_alloc[r] += adjust
                    diff -= adjust
        
        for (pre_idx, post_idx, weight, n_pairs, pre_region) in blocks:
            region_target = pre_region_alloc.get(pre_region, 0)
            if region_target <= 0 or region_normalization[pre_region] <= 0:
                alloc.append(0)
                continue
            n_block = int(round(region_target * (weight / region_normalization[pre_region])))
            n_block = min(max(0, n_block), n_pairs)
            alloc.append(n_block)
        
        total_allocated = sum(alloc)
        if total_allocated != total_target:
            diff = total_target - total_allocated
            if diff != 0:
                order = np.argsort([-w for (_pre, _post, w, _npairs, _r) in blocks])
                for k in order:
                    if diff == 0:
                        break
                    pre_idx, post_idx, weight, n_pairs, pre_region = blocks[k]
                    gap = n_pairs - alloc[k]
                    if gap <= 0:
                        continue
                    add = 1 if diff > 0 else -1
                    add = max(-alloc[k], min(gap, add))
                    alloc[k] += add
                    diff -= add

        all_i, all_j = [], []
        for k, (pre_idx, post_idx, weight, n_pairs, pre_region) in enumerate(blocks):
            n_block = alloc[k]
            if n_block <= 0:
                continue
            i_conn, j_conn = self._create_connections_vectorized(pre_idx, post_idx, n_block)
            all_i.extend(i_conn)
            all_j.extend(j_conn)

        i_arr = np.array(all_i, dtype=int)
        j_arr = np.array(all_j, dtype=int)
        self._topo_cache[cache_key] = (i_arr, j_arr)
        return i_arr.copy(), j_arr.copy()
    
    def _create_connections_vectorized(self, 
                                     pre_idx: np.ndarray, 
                                     post_idx: np.ndarray, 
                                     expected_connections: int) -> Tuple[np.ndarray, np.ndarray]:
        n_pre = len(pre_idx)
        n_post = len(post_idx)
        
        total_possible = n_pre * n_post
        
        if expected_connections >= total_possible:
            i_indices_local = np.repeat(np.arange(n_pre), n_post)
            j_indices_local = np.tile(np.arange(n_post), n_pre)
        else:
            step_size = total_possible / expected_connections
            connection_indices = []
            for i in range(expected_connections):
                idx = int(i * step_size) % total_possible
                connection_indices.append(idx)
            
            connection_indices = np.array(connection_indices)
            i_indices_local = connection_indices // n_post
            j_indices_local = connection_indices % n_post

        i_indices = pre_idx[i_indices_local]
        j_indices = post_idx[j_indices_local]
        
        return i_indices, j_indices
    
    def create_distance_based_connections(self,
                                         pre_group: str,
                                         post_group: str,
                                         total_connections: int,
                                         distance_decay: float = 0.1,
                                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:

        if seed is not None:
            np.random.seed(seed)

        try:
            pre_total = sum(len(indices) for indices in self.region_map[pre_group].values())
            post_total = sum(len(indices) for indices in self.region_map[post_group].values())
        except KeyError:
            raise ValueError(f"Group {pre_group} or {post_group} not registered in region map")

        cache_key = (pre_group, post_group, int(total_connections), round(distance_decay, 6), seed, pre_total, post_total)
        if cache_key in self._distance_cache:
            i_cached, j_cached = self._distance_cache[cache_key]
            return i_cached.copy(), j_cached.copy()
        
        pre_indices_all = np.concatenate([self.region_map[pre_group][i] for i in range(3)])
        post_indices_all = np.concatenate([self.region_map[post_group][i] for i in range(3)])
        
        i_conn = []
        j_conn = []
        
        for _ in range(total_connections):
            pre_idx_local = np.random.randint(0, len(pre_indices_all))
            pre_idx_global = pre_indices_all[pre_idx_local]
            
            distances = np.abs(post_indices_all - pre_idx_global * (post_total / pre_total))
            max_distance = np.max(distances)
            
            if max_distance > 0:
                probabilities = np.exp(-distance_decay * distances / max_distance)
                probabilities /= probabilities.sum()
            else:
                probabilities = np.ones(len(post_indices_all)) / len(post_indices_all)
            
            post_idx_local = np.random.choice(len(post_indices_all), p=probabilities)
            post_idx_global = post_indices_all[post_idx_local]
            
            i_conn.append(pre_idx_global)
            j_conn.append(post_idx_global)
        
        i_arr = np.array(i_conn, dtype=int)
        j_arr = np.array(j_conn, dtype=int)
        self._distance_cache[cache_key] = (i_arr, j_arr)
        return i_arr.copy(), j_arr.copy()

    def create_index_based_connections(self,
                                       pre_group: str,
                                       post_group: str,
                                       total_connections: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            pre_indices_all = np.concatenate([self.region_map[pre_group][i] for i in range(3)])
            post_indices_all = np.concatenate([self.region_map[post_group][i] for i in range(3)])
        except KeyError:
            raise ValueError(f"Group {pre_group} or {post_group} not registered in region map")

        n_pre = len(pre_indices_all)
        n_post = len(post_indices_all)
        total_possible = n_pre * n_post
        if total_connections >= total_possible:
            i_local = np.repeat(np.arange(n_pre), n_post)
            j_local = np.tile(np.arange(n_post), n_pre)
        else:
            step_size = total_possible / max(1, total_connections)
            idxs = (np.floor(np.arange(total_connections) * step_size).astype(int)) % total_possible
            i_local = idxs // n_post
            j_local = idxs % n_post
        return pre_indices_all[i_local], post_indices_all[j_local]
    
    def print_topology_summary(self):
        print("\n" + "="*60)
        print("Topographic Organization Summary")
        print("="*60)
        
        for group_name in sorted(self.region_map.keys()):
            print(f"\n[Group: {group_name}]")
            regions = self.region_map[group_name]
            total_n = sum(len(v) for v in regions.values())
            print(f"  Total Neurons: {total_n}")
            for region_id in range(3):
                indices = regions[region_id]
                region_name = TopographicRegion.get_name(region_id)
                print(f"  {region_name:12s}: indices {indices[0] if len(indices)>0 else 'N/A'}-{indices[-1] if len(indices)>0 else 'N/A'} (n={len(indices)})")
        
        print("\n[Connection Rules]")
        print(f"  Forward (ctx→str, str→gpe, gpe→stn):")
        print(f"    - Within-region (Distance 0): 100% of Base P")
        print(f"    - Adjacent regions (Distance 1): {self.overlap_ratio*100:.0f}% of Base P (Overlap)")
        print(f"    - Distant regions (Distance 2): {self.distant_ratio*100:.0f}% of Base P (Weak)")
        print(f"  Feedback (stn→gpe):")
        print(f"    - Strict within-region only: 100% of Base P if same region, 0% otherwise")
        print("="*60 + "\n")


def identify_connection_type(pre_name: str, post_name: str) -> Optional[str]:

    pre_lower = pre_name.lower()
    post_lower = post_name.lower()
    
    if 'ctx' in pre_lower or 'cortex' in pre_lower:
        if any(x in post_lower for x in ['msn', 'fsn', 'striatum', 'str']):
            return 'forward'
    
    if any(x in pre_lower for x in ['msn', 'fsn', 'striatum', 'str']):
        if 'gpe' in post_lower:
            return 'forward'
    
    if 'gpe' in pre_lower:
        if 'stn' in post_lower:
            return 'forward'
    
    if 'stn' in pre_lower:
        if 'gpe' in post_lower:
            return 'feedback'
    
    return None


def get_topographic_groups() -> List[str]:

    return [
        'CTX', 'Cortex', 
        'MSND1', 'MSND2', 'FSN', 
        'GPeT1', 'GPeTA',  
        'STN', 'STN_PVminus', 'STN_PVplus'
    ]


def should_apply_topographic(group_name: str) -> bool:
    topo_groups = get_topographic_groups()
    group_upper = group_name.upper()
    
    for pattern in topo_groups:
        if pattern.upper() in group_upper:
            return True
    
    return False