"""
Performance Analyzer API endpoints for monitoring OpenSearch cluster performance.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

from ...core.shared.security import get_current_user, User
from ...services.performance_analyzer import (
    PerformanceAnalyzerService, 
    PerformanceAnalyzerClient,
    MetricType
)

router = APIRouter()
logger = logging.getLogger(__name__)


class PerformanceOverviewResponse(BaseModel):
    """Performance overview response model."""
    timestamp: str = Field(..., description="Response timestamp")
    cluster: Dict[str, Any] = Field(..., description="Cluster metrics")
    nodes: Dict[str, Any] = Field(..., description="Node metrics")
    performance_analyzer: Dict[str, Any] = Field(..., description="PA status")


class PerformanceAlertResponse(BaseModel):
    """Performance alert response model."""
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    node_id: Optional[str] = Field(None, description="Node ID if node-specific")
    value: Optional[float] = Field(None, description="Metric value")
    timestamp: str = Field(..., description="Alert timestamp")


class IndexPerformanceResponse(BaseModel):
    """Index performance response model."""
    index_name: str = Field(..., description="Index name")
    timestamp: str = Field(..., description="Response timestamp")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    status: str = Field(..., description="Performance status")


@router.get("/", response_model=PerformanceOverviewResponse)
async def get_performance_overview(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive performance overview of the OpenSearch cluster.
    
    Returns:
        System performance overview including cluster and node metrics
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for performance monitoring"
            )
        
        service = PerformanceAnalyzerService()
        overview = await service.get_system_overview()
        
        return PerformanceOverviewResponse(
            timestamp=overview["timestamp"],
            cluster=overview["cluster"],
            nodes=overview["nodes"],
            performance_analyzer=overview["performance_analyzer"]
        )
        
    except Exception as e:
        logger.error(f"Error getting performance overview: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get performance overview: {str(e)}"
        )


@router.get("/alerts", response_model=List[PerformanceAlertResponse])
async def get_performance_alerts(
    current_user: User = Depends(get_current_user)
):
    """
    Get current performance alerts based on thresholds.
    
    Returns:
        List of active performance alerts
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for performance alerts"
            )
        
        service = PerformanceAnalyzerService()
        alerts = await service.get_performance_alerts()
        
        return [
            PerformanceAlertResponse(
                type=alert["type"],
                severity=alert["severity"],
                message=alert["message"],
                node_id=alert.get("node_id"),
                value=alert.get("value"),
                timestamp=datetime.now().isoformat()
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting performance alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance alerts: {str(e)}"
        )


@router.get("/nodes")
async def get_node_performance(
    node_id: Optional[str] = Query(None, description="Specific node ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed performance metrics for cluster nodes.
    
    Args:
        node_id: Optional specific node ID to query
        
    Returns:
        Node performance metrics
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for node metrics"
            )
        
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                raise HTTPException(
                    status_code=503,
                    detail="Performance Analyzer is not available"
                )
            
            node_metrics = await client.get_node_metrics(node_id)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "node_name": node.node_name,
                        "cpu_utilization": node.cpu_utilization,
                        "memory_utilization": node.memory_utilization,
                        "disk_utilization": node.disk_utilization,
                        "heap_used_mb": node.heap_used / (1024 * 1024) if node.heap_used else 0,
                        "heap_max_mb": node.heap_max / (1024 * 1024) if node.heap_max else 0,
                        "heap_usage_percent": (node.heap_used / node.heap_max * 100) if node.heap_max > 0 else 0,
                        "gc_time_ms": node.gc_time,
                        "thread_pool_stats": node.thread_pool_stats
                    }
                    for node in node_metrics
                ]
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get node performance: {str(e)}"
        )


@router.get("/cluster")
async def get_cluster_performance(
    current_user: User = Depends(get_current_user)
):
    """
    Get cluster-level performance metrics.
    
    Returns:
        Cluster performance metrics
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for cluster metrics"
            )
        
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                raise HTTPException(
                    status_code=503,
                    detail="Performance Analyzer is not available"
                )
            
            cluster_metrics = await client.get_cluster_metrics()
            
            if not cluster_metrics:
                raise HTTPException(
                    status_code=404,
                    detail="Cluster metrics not available"
                )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cluster_name": cluster_metrics.cluster_name,
                "cluster_status": cluster_metrics.cluster_status,
                "number_of_nodes": cluster_metrics.number_of_nodes,
                "active_primary_shards": cluster_metrics.active_primary_shards,
                "active_shards": cluster_metrics.active_shards,
                "relocating_shards": cluster_metrics.relocating_shards,
                "initializing_shards": cluster_metrics.initializing_shards,
                "unassigned_shards": cluster_metrics.unassigned_shards,
                "task_max_waiting_time_ms": cluster_metrics.task_max_waiting_time
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cluster performance: {str(e)}"
        )


@router.get("/index/{index_name}", response_model=IndexPerformanceResponse)
async def get_index_performance(
    index_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get performance metrics for a specific index.
    
    Args:
        index_name: Name of the index to analyze
        
    Returns:
        Index performance metrics
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for index metrics"
            )
        
        service = PerformanceAnalyzerService()
        summary = await service.get_index_performance_summary(index_name)
        
        return IndexPerformanceResponse(
            index_name=summary["index_name"],
            timestamp=summary["timestamp"],
            metrics=summary["metrics"],
            status=summary["status"]
        )
        
    except Exception as e:
        logger.error(f"Error getting index performance for {index_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index performance: {str(e)}"
        )


@router.get("/metrics")
async def get_raw_metrics(
    metric_types: List[str] = Query(..., description="Metric types to retrieve"),
    hours_back: int = Query(1, ge=1, le=24, description="Hours of data to retrieve"),
    node_ids: Optional[List[str]] = Query(None, description="Specific node IDs"),
    current_user: User = Depends(get_current_user)
):
    """
    Get raw Performance Analyzer metrics.
    
    Args:
        metric_types: List of metric types to retrieve
        hours_back: Number of hours of historical data
        node_ids: Optional list of specific node IDs
        
    Returns:
        Raw performance metrics
    """
    try:
        # Проверка прав администратора
        if "admin" not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail="Admin access required for raw metrics"
            )
        
        # Проверка типов метрик
        valid_metrics = {
            "cpu", "memory", "disk", "heap", "gc", "indexing", 
            "search", "cache", "thread_pool", "network"
        }
        
        invalid_metrics = set(metric_types) - valid_metrics
        if invalid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric types: {list(invalid_metrics)}"
            )
        
        # Преобразование строковых типов метрик в enum
        try:
            metric_enum_list = [PerformanceMetricType(metric) for metric in metric_types]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {str(e)}"
            )
        
        # Преобразование метрик в сериализуемый формат
        processed_metrics = [
            {
                "metric_type": metric.metric_type.value,
                "node_id": metric.node_id,
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.value,
                "unit": metric.unit
            }
            for metric in raw_metrics
        ]
        
        async with PerformanceAnalyzerClient() as client:
            if not await client.health_check():
                raise HTTPException(
                    status_code=503,
                    detail="Performance Analyzer is not available"
                )
            
            # Convert string metric types to enum
            metric_enums = [MetricType(mt) for mt in metric_types]
            
            metrics = await client.get_metrics(
                metric_types=metric_enums,
                start_time=datetime.now() - timedelta(hours=hours_back),
                node_ids=node_ids
            )
            
            # Convert metrics to serializable format
            serialized_metrics = []
            for metric in metrics:
                serialized_metrics.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "metric_type": metric.metric_type,
                    "node_id": metric.node_id,
                    "value": metric.value,
                    "unit": metric.unit,
                    "metadata": metric.metadata
                })
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_metrics": len(serialized_metrics),
                "time_range_hours": hours_back,
                "metrics": serialized_metrics
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raw metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get raw metrics: {str(e)}"
        )


@router.get("/health")
async def get_performance_analyzer_health():
    """
    Check Performance Analyzer health status.
    
    Returns:
        Performance Analyzer health status
    """
    try:
        async with PerformanceAnalyzerClient() as client:
            is_healthy = await client.health_check()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "performance_analyzer": {
                    "enabled": client.enabled,
                    "healthy": is_healthy,
                    "base_url": client.base_url,
                    "timeout": client.timeout
                },
                "status": "healthy" if is_healthy else "unhealthy"
            }
        
    except Exception as e:
        logger.error(f"Error checking PA health: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_analyzer": {
                "enabled": False,
                "healthy": False,
                "error": str(e)
            },
            "status": "error"
        } 