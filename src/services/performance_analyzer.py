"""
OpenSearch Performance Analyzer integration service.
Provides access to detailed performance metrics and monitoring data.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..core.config import get_config

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Available Performance Analyzer metric types."""
    CPU_UTILIZATION = "CPU_Utilization"
    MEMORY_UTILIZATION = "Memory_Utilization"
    DISK_UTILIZATION = "Disk_Utilization"
    NETWORK_UTILIZATION = "Network_Utilization"
    GC_INFO = "GC_Info"
    THREAD_POOL = "ThreadPool"
    SHARD_STATS = "Shard_Stats"
    INDEX_STATS = "Index_Stats"
    CLUSTER_MANAGER_METRICS = "ClusterManager_Metrics"
    CACHE_STATS = "Cache_Stats"
    CIRCUIT_BREAKER = "CircuitBreaker_Stats"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_type: str
    node_id: Optional[str]
    value: float
    unit: str
    metadata: Dict[str, Any]


@dataclass
class NodeMetrics:
    """Node-level performance metrics."""
    node_id: str
    node_name: str
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    heap_used: float
    heap_max: float
    gc_time: float
    thread_pool_stats: Dict[str, Any]


@dataclass
class ClusterMetrics:
    """Cluster-level performance metrics."""
    cluster_name: str
    cluster_status: str
    number_of_nodes: int
    active_primary_shards: int
    active_shards: int
    relocating_shards: int
    initializing_shards: int
    unassigned_shards: int
    task_max_waiting_time: float


class PerformanceAnalyzerClient:
    """Client for OpenSearch Performance Analyzer API."""
    
    def __init__(self):
        self.config = get_config()
        self.pa_config = self.config.performance_analyzer
        self.base_url = self.pa_config.base_url
        self.timeout = self.pa_config.timeout
        self.enabled = self.pa_config.enabled
        
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        if self.enabled:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            
    async def health_check(self) -> bool:
        """Check if Performance Analyzer is available."""
        if not self.enabled:
            return False
            
        try:
            async with self._session.get(f"{self.base_url}/_opendistro/_performanceanalyzer/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Performance Analyzer health check failed: {e}")
            return False
    
    async def get_metrics(self, 
                         metric_types: List[MetricType], 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         node_ids: Optional[List[str]] = None) -> List[PerformanceMetric]:
        """
        Get performance metrics from Performance Analyzer.
        
        Args:
            metric_types: List of metric types to retrieve
            start_time: Start time for metrics (default: 1 hour ago)
            end_time: End time for metrics (default: now)
            node_ids: Specific node IDs to query (default: all nodes)
            
        Returns:
            List of performance metrics
        """
        if not self.enabled or not self._session:
            return []
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
            
        metrics = []
        
        for metric_type in metric_types:
            try:
                url = f"{self.base_url}/_opendistro/_performanceanalyzer/metrics"
                params = {
                    "metrics": metric_type.value,
                    "starttime": int(start_time.timestamp() * 1000),
                    "endtime": int(end_time.timestamp() * 1000)
                }
                
                if node_ids:
                    params["nodes"] = ",".join(node_ids)
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed_metrics = self._parse_metrics(data, metric_type.value)
                        metrics.extend(parsed_metrics)
                    else:
                        logger.warning(f"Failed to get {metric_type.value} metrics: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error getting {metric_type.value} metrics: {e}")
                
        return metrics
    
    async def get_node_metrics(self, node_id: Optional[str] = None) -> List[NodeMetrics]:
        """Get comprehensive node performance metrics."""
        if not self.enabled:
            return []
            
        node_metrics = []
        
        try:
            # Get current node metrics
            metric_types = [
                MetricType.CPU_UTILIZATION,
                MetricType.MEMORY_UTILIZATION, 
                MetricType.DISK_UTILIZATION,
                MetricType.GC_INFO,
                MetricType.THREAD_POOL
            ]
            
            metrics = await self.get_metrics(
                metric_types=metric_types,
                start_time=datetime.now() - timedelta(minutes=5),
                node_ids=[node_id] if node_id else None
            )
            
            # Group metrics by node
            nodes_data = {}
            for metric in metrics:
                if metric.node_id not in nodes_data:
                    nodes_data[metric.node_id] = {}
                nodes_data[metric.node_id][metric.metric_type] = metric
            
            # Create NodeMetrics objects
            for node_id, data in nodes_data.items():
                node_metric = NodeMetrics(
                    node_id=node_id,
                    node_name=data.get("node_name", f"node-{node_id}"),
                    cpu_utilization=data.get(MetricType.CPU_UTILIZATION.value, {}).get("value", 0.0),
                    memory_utilization=data.get(MetricType.MEMORY_UTILIZATION.value, {}).get("value", 0.0),
                    disk_utilization=data.get(MetricType.DISK_UTILIZATION.value, {}).get("value", 0.0),
                    heap_used=data.get(MetricType.MEMORY_UTILIZATION.value, {}).get("metadata", {}).get("heap_used", 0.0),
                    heap_max=data.get(MetricType.MEMORY_UTILIZATION.value, {}).get("metadata", {}).get("heap_max", 0.0),
                    gc_time=data.get(MetricType.GC_INFO.value, {}).get("value", 0.0),
                    thread_pool_stats=data.get(MetricType.THREAD_POOL.value, {}).get("metadata", {})
                )
                node_metrics.append(node_metric)
                
        except Exception as e:
            logger.error(f"Error getting node metrics: {e}")
            
        return node_metrics
    
    async def get_cluster_metrics(self) -> Optional[ClusterMetrics]:
        """Get cluster-level performance metrics."""
        if not self.enabled:
            return None
            
        try:
            metrics = await self.get_metrics(
                metric_types=[MetricType.CLUSTER_MANAGER_METRICS],
                start_time=datetime.now() - timedelta(minutes=1)
            )
            
            if not metrics:
                return None
                
            latest_metric = max(metrics, key=lambda m: m.timestamp)
            metadata = latest_metric.metadata
            
            return ClusterMetrics(
                cluster_name=metadata.get("cluster_name", "unknown"),
                cluster_status=metadata.get("cluster_status", "unknown"), 
                number_of_nodes=metadata.get("number_of_nodes", 0),
                active_primary_shards=metadata.get("active_primary_shards", 0),
                active_shards=metadata.get("active_shards", 0),
                relocating_shards=metadata.get("relocating_shards", 0),
                initializing_shards=metadata.get("initializing_shards", 0),
                unassigned_shards=metadata.get("unassigned_shards", 0),
                task_max_waiting_time=metadata.get("task_max_waiting_time", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting cluster metrics: {e}")
            return None
    
    async def get_index_performance(self, index_name: str) -> Dict[str, Any]:
        """Get performance metrics for specific index."""
        if not self.enabled:
            return {}
            
        try:
            metrics = await self.get_metrics(
                metric_types=[MetricType.INDEX_STATS, MetricType.SHARD_STATS],
                start_time=datetime.now() - timedelta(minutes=5)
            )
            
            index_metrics = {}
            for metric in metrics:
                if metric.metadata.get("index_name") == index_name:
                    index_metrics[metric.metric_type] = {
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "metadata": metric.metadata
                    }
                    
            return index_metrics
            
        except Exception as e:
            logger.error(f"Error getting index performance for {index_name}: {e}")
            return {}
    
    def _parse_metrics(self, data: Dict, metric_type: str) -> List[PerformanceMetric]:
        """Parse raw Performance Analyzer response into PerformanceMetric objects."""
        metrics = []
        
        try:
            # PA response format may vary, this is a general parser
            if isinstance(data, dict) and "data" in data:
                for record in data["data"]:
                    timestamp = datetime.fromtimestamp(record.get("timestamp", 0) / 1000)
                    
                    metric = PerformanceMetric(
                        timestamp=timestamp,
                        metric_type=metric_type,
                        node_id=record.get("node_id"),
                        value=float(record.get("value", 0)),
                        unit=record.get("unit", ""),
                        metadata=record.get("metadata", {})
                    )
                    metrics.append(metric)
                    
        except Exception as e:
            logger.error(f"Error parsing metrics for {metric_type}: {e}")
            
        return metrics


class PerformanceAnalyzerService:
    """High-level service for Performance Analyzer operations."""
    
    def __init__(self):
        self.config = get_config()
        self._client_cache = {}
        
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system performance overview."""
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                return {"error": "Performance Analyzer not available"}
            
            # Get all key metrics
            node_metrics = await client.get_node_metrics()
            cluster_metrics = await client.get_cluster_metrics()
            
            # Calculate aggregate statistics
            overview = {
                "timestamp": datetime.now().isoformat(),
                "cluster": cluster_metrics.__dict__ if cluster_metrics else {},
                "nodes": {
                    "count": len(node_metrics),
                    "avg_cpu": sum(n.cpu_utilization for n in node_metrics) / len(node_metrics) if node_metrics else 0,
                    "avg_memory": sum(n.memory_utilization for n in node_metrics) / len(node_metrics) if node_metrics else 0,
                    "total_heap_used": sum(n.heap_used for n in node_metrics),
                    "total_heap_max": sum(n.heap_max for n in node_metrics),
                    "details": [n.__dict__ for n in node_metrics]
                },
                "performance_analyzer": {
                    "enabled": client.enabled,
                    "healthy": await client.health_check()
                }
            }
            
            return overview
    
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                return alerts
                
            node_metrics = await client.get_node_metrics()
            
            # Check node-level alerts
            for node in node_metrics:
                if node.cpu_utilization > 80:
                    alerts.append({
                        "type": "HIGH_CPU",
                        "severity": "warning" if node.cpu_utilization < 90 else "critical",
                        "node_id": node.node_id,
                        "value": node.cpu_utilization,
                        "message": f"High CPU utilization on {node.node_name}: {node.cpu_utilization:.1f}%"
                    })
                
                if node.memory_utilization > 85:
                    alerts.append({
                        "type": "HIGH_MEMORY",
                        "severity": "warning" if node.memory_utilization < 95 else "critical", 
                        "node_id": node.node_id,
                        "value": node.memory_utilization,
                        "message": f"High memory utilization on {node.node_name}: {node.memory_utilization:.1f}%"
                    })
                
                if node.heap_used > 0 and node.heap_max > 0:
                    heap_usage = (node.heap_used / node.heap_max) * 100
                    if heap_usage > 80:
                        alerts.append({
                            "type": "HIGH_HEAP", 
                            "severity": "warning" if heap_usage < 90 else "critical",
                            "node_id": node.node_id,
                            "value": heap_usage,
                            "message": f"High heap usage on {node.node_name}: {heap_usage:.1f}%"
                        })
            
            # Check cluster-level alerts
            cluster_metrics = await client.get_cluster_metrics()
            if cluster_metrics:
                if cluster_metrics.cluster_status == "red":
                    alerts.append({
                        "type": "CLUSTER_RED",
                        "severity": "critical",
                        "message": "Cluster status is RED"
                    })
                elif cluster_metrics.cluster_status == "yellow":
                    alerts.append({
                        "type": "CLUSTER_YELLOW", 
                        "severity": "warning",
                        "message": "Cluster status is YELLOW"
                    })
                
                if cluster_metrics.unassigned_shards > 0:
                    alerts.append({
                        "type": "UNASSIGNED_SHARDS",
                        "severity": "warning",
                        "value": cluster_metrics.unassigned_shards,
                        "message": f"{cluster_metrics.unassigned_shards} unassigned shards"
                    })
        
        return alerts
    
    async def get_index_performance_summary(self, index_name: str) -> Dict[str, Any]:
        """Get performance summary for specific index."""
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                return {"error": "Performance Analyzer not available"}
            
            index_perf = await client.get_index_performance(index_name)
            
            summary = {
                "index_name": index_name,
                "timestamp": datetime.now().isoformat(),
                "metrics": index_perf,
                "status": "healthy" if index_perf else "no_data"
            }
            
            return summary 