"""
Optimized UMI clustering algorithms for large datasets.

Provides efficient implementations of edit distance-based UMI clustering
with performance optimizations for production use.
"""

from typing import List, Set, Dict, Tuple
import numpy as np
from collections import defaultdict, Counter


class OptimizedUMICluster:
    """Optimized UMI clustering with efficient edit distance calculation."""
    
    def __init__(self, max_edit_distance: int = 1):
        """Initialize clusterer with maximum edit distance."""
        self.max_edit_distance = max_edit_distance
        
    def hamming_distance(self, seq1: str, seq2: str) -> int:
        """Calculate Hamming distance between equal-length sequences."""
        if len(seq1) != len(seq2):
            return max(len(seq1), len(seq2))  # Large distance for unequal lengths
        
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    
    def levenshtein_distance(self, seq1: str, seq2: str) -> int:
        """Calculate Levenshtein edit distance using dynamic programming."""
        if len(seq1) == len(seq2):
            return self.hamming_distance(seq1, seq2)
        
        m, n = len(seq1), len(seq2)
        
        # Create distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[m][n]
    
    def cluster_umis_greedy(self, umis: List[str]) -> List[List[str]]:
        """Cluster UMIs using greedy algorithm.
        
        This is faster but may not find optimal clustering.
        Suitable for most practical applications.
        
        Args:
            umis: List of UMI sequences
            
        Returns:
            List of clusters, each containing UMI sequences
        """
        if not umis:
            return []
        
        # Count occurrences for tie-breaking
        umi_counts = Counter(umis)
        unique_umis = list(set(umis))
        
        # Sort by frequency (descending) for better clustering
        unique_umis.sort(key=lambda x: (-umi_counts[x], x))
        
        clusters = []
        used = set()
        
        for umi in unique_umis:
            if umi in used:
                continue
            
            # Start new cluster with this UMI
            cluster = [umi]
            used.add(umi)
            
            # Find all UMIs within edit distance
            for other_umi in unique_umis:
                if (other_umi not in used and 
                    self.hamming_distance(umi, other_umi) <= self.max_edit_distance):
                    cluster.append(other_umi)
                    used.add(other_umi)
            
            clusters.append(cluster)
        
        return clusters
    
    def cluster_umis_connected_components(self, umis: List[str]) -> List[List[str]]:
        """Cluster UMIs using connected components algorithm.
        
        This finds optimal clustering by treating UMIs as graph nodes
        and edit distances as edges.
        
        Args:
            umis: List of UMI sequences
            
        Returns:
            List of clusters, each containing UMI sequences
        """
        if not umis:
            return []
        
        unique_umis = list(set(umis))
        n = len(unique_umis)
        
        if n == 1:
            return [unique_umis]
        
        # Build adjacency graph
        graph = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                umi1, umi2 = unique_umis[i], unique_umis[j]
                if self.hamming_distance(umi1, umi2) <= self.max_edit_distance:
                    graph[umi1].add(umi2)
                    graph[umi2].add(umi1)
        
        # Find connected components using DFS
        visited = set()
        clusters = []
        
        def dfs(umi: str, cluster: List[str]) -> None:
            if umi in visited:
                return
            
            visited.add(umi)
            cluster.append(umi)
            
            for neighbor in graph[umi]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for umi in unique_umis:
            if umi not in visited:
                cluster = []
                dfs(umi, cluster)
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def cluster_umis_fast(self, umis: List[str]) -> List[List[str]]:
        """Fast UMI clustering with optimizations for large datasets.
        
        Uses various optimizations:
        - Early termination for identical sequences
        - Length-based pre-filtering
        - Optimized distance calculation
        
        Args:
            umis: List of UMI sequences
            
        Returns:
            List of clusters, each containing UMI sequences
        """
        if not umis:
            return []
        
        # Group by length first (optimization)
        length_groups = defaultdict(list)
        for umi in set(umis):  # Remove duplicates
            length_groups[len(umi)].append(umi)
        
        all_clusters = []
        
        # Process each length group separately
        for length, umi_group in length_groups.items():
            if len(umi_group) == 1:
                all_clusters.append(umi_group)
                continue
            
            # For same-length sequences, use greedy clustering
            clusters = self.cluster_umis_greedy(umi_group)
            all_clusters.extend(clusters)
        
        # If max_edit_distance > 0, we need to check cross-length clustering
        if self.max_edit_distance > 0 and len(length_groups) > 1:
            # Merge clusters that are within edit distance across length groups
            merged_clusters = self._merge_cross_length_clusters(all_clusters)
            return merged_clusters
        
        return all_clusters
    
    def _merge_cross_length_clusters(self, clusters: List[List[str]]) -> List[List[str]]:
        """Merge clusters that should be connected across different lengths."""
        if len(clusters) <= 1:
            return clusters
        
        # Build cluster graph
        cluster_graph = defaultdict(set)
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i >= j:
                    continue
                
                # Check if any UMIs in cluster1 are close to any in cluster2
                should_merge = False
                for umi1 in cluster1:
                    for umi2 in cluster2:
                        if self.levenshtein_distance(umi1, umi2) <= self.max_edit_distance:
                            should_merge = True
                            break
                    if should_merge:
                        break
                
                if should_merge:
                    cluster_graph[i].add(j)
                    cluster_graph[j].add(i)
        
        # Find connected components of clusters
        visited = set()
        merged_clusters = []
        
        def dfs_clusters(cluster_idx: int, merged_cluster: List[str]) -> None:
            if cluster_idx in visited:
                return
            
            visited.add(cluster_idx)
            merged_cluster.extend(clusters[cluster_idx])
            
            for neighbor_idx in cluster_graph[cluster_idx]:
                if neighbor_idx not in visited:
                    dfs_clusters(neighbor_idx, merged_cluster)
        
        for i in range(len(clusters)):
            if i not in visited:
                merged_cluster = []
                dfs_clusters(i, merged_cluster)
                if merged_cluster:
                    merged_clusters.append(merged_cluster)
        
        return merged_clusters
    
    def benchmark_clustering_methods(
        self, 
        umis: List[str], 
        methods: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark different clustering methods for performance comparison.
        
        Args:
            umis: List of UMI sequences to cluster
            methods: List of method names to benchmark
            
        Returns:
            Dictionary with timing and accuracy results for each method
        """
        import time
        
        if methods is None:
            methods = ['greedy', 'connected_components', 'fast']
        
        results = {}
        
        for method in methods:
            start_time = time.perf_counter()
            
            if method == 'greedy':
                clusters = self.cluster_umis_greedy(umis)
            elif method == 'connected_components':
                clusters = self.cluster_umis_connected_components(umis)
            elif method == 'fast':
                clusters = self.cluster_umis_fast(umis)
            else:
                continue
            
            end_time = time.perf_counter()
            
            results[method] = {
                'runtime_seconds': end_time - start_time,
                'num_clusters': len(clusters),
                'num_input_umis': len(umis),
                'num_unique_umis': len(set(umis)),
                'average_cluster_size': np.mean([len(cluster) for cluster in clusters]),
                'max_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0
            }
        
        return results