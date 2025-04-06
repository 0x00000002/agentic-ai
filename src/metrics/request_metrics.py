# src/metrics/request_metrics.py

from typing import Dict, Any, List, Optional, Union, Set
import uuid
import json
from datetime import datetime, timedelta, timezone
import time
import os


class RequestMetricsService:
    """
    Service for tracking metrics about agent and tool usage during requests.
    Enables performance optimization and usage analysis.
    """
    
    def __init__(self, storage_path: str = "data/metrics/request_metrics.json"):
        """Initialize the request metrics service."""
        self._metrics_data = {}  # Request metrics data
        self._storage_path = storage_path
        self._ensure_storage_dir()
        self._load_metrics()
    
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
    
    def _load_metrics(self):
        """Load metrics from storage if available."""
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, 'r') as f:
                    self._metrics_data = json.load(f)
            except Exception as e:
                print(f"Error loading metrics data: {e}")
                self._metrics_data = {
                    "requests": {},
                    "agent_usage": {},
                    "tool_usage": {},
                    "model_usage": {}
                }
        else:
            self._metrics_data = {
                "requests": {},      # Request-level metrics
                "agent_usage": {},   # Agent-level metrics
                "tool_usage": {},    # Tool-level metrics  
                "model_usage": {}    # Model-level metrics
            }
    
    def start_request_tracking(self, 
                             request_id: Optional[str] = None, 
                             prompt: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a request.
        
        Args:
            request_id: Optional ID for the request (generated if not provided)
            prompt: The user prompt
            metadata: Additional request metadata
            
        Returns:
            Request ID for tracking
        """
        if not request_id:
            request_id = str(uuid.uuid4())
            
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if "requests" not in self._metrics_data:
            self._metrics_data["requests"] = {}
            
        self._metrics_data["requests"][request_id] = {
            "request_id": request_id,
            "start_time": timestamp,
            "end_time": None,
            "duration_ms": None,
            "prompt": prompt[:1000] if prompt else None,  # Truncate long prompts
            "prompt_tokens": len(prompt.split()) if prompt else 0,
            "metadata": metadata or {},
            "agents_used": [],
            "tools_used": [],
            "models_used": [],
            "success": None,
            "error": None
        }
        
        return request_id
    
    def end_request_tracking(self, 
                           request_id: str, 
                           success: bool = True,
                           error: Optional[str] = None) -> None:
        """
        End tracking for a request.
        
        Args:
            request_id: ID of the request
            success: Whether the request was successful
            error: Error message if unsuccessful
        """
        if request_id not in self._metrics_data["requests"]:
            return
            
        request_data = self._metrics_data["requests"][request_id]
        request_data["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate duration if start_time is available
        if request_data.get("start_time"):
            start = datetime.fromisoformat(request_data["start_time"])
            end = datetime.fromisoformat(request_data["end_time"])
            duration_ms = int((end - start).total_seconds() * 1000)
            request_data["duration_ms"] = duration_ms
        
        request_data["success"] = success
        request_data["error"] = error
        
        # Save metrics to disk
        self._save_metrics()
    
    def track_agent_usage(self, 
                        request_id: str, 
                        agent_id: str,
                        confidence: float = 1.0,
                        duration_ms: Optional[int] = None,
                        success: bool = True,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track usage of an agent during a request.
        
        Args:
            request_id: ID of the request
            agent_id: ID of the agent used
            confidence: Confidence score for agent selection
            duration_ms: Time taken by the agent in milliseconds
            success: Whether the agent was successful
            metadata: Additional metadata about agent usage
        """
        if request_id not in self._metrics_data["requests"]:
            return
            
        # Update request data
        if agent_id not in self._metrics_data["requests"][request_id]["agents_used"]:
            self._metrics_data["requests"][request_id]["agents_used"].append(agent_id)
        
        # Update agent usage stats
        if "agent_usage" not in self._metrics_data:
            self._metrics_data["agent_usage"] = {}
            
        if agent_id not in self._metrics_data["agent_usage"]:
            self._metrics_data["agent_usage"][agent_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_duration_ms": 0,
                "avg_confidence": 0,
                "last_used": None,
                "avg_duration_ms": 0
            }
            
        # Update agent metrics
        agent_data = self._metrics_data["agent_usage"][agent_id]
        agent_data["total_requests"] += 1
        if success:
            agent_data["successful_requests"] += 1
        
        if duration_ms:
            agent_data["total_duration_ms"] += duration_ms
            agent_data["avg_duration_ms"] = agent_data["total_duration_ms"] / agent_data["total_requests"]
            
        # Update average confidence
        current_avg = agent_data["avg_confidence"]
        total_requests = agent_data["total_requests"]
        agent_data["avg_confidence"] = ((current_avg * (total_requests - 1)) + confidence) / total_requests
        
        agent_data["last_used"] = datetime.now(timezone.utc).isoformat()
        
        # Save metrics to disk
        self._save_metrics()
    
    def track_tool_usage(self, 
                       request_id: str, 
                       tool_id: str,
                       duration_ms: Optional[int] = None,
                       success: bool = True,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track usage of a tool during a request.
        
        Args:
            request_id: ID of the request
            tool_id: ID of the tool used
            duration_ms: Time taken by the tool in milliseconds
            success: Whether the tool was successful
            metadata: Additional metadata about tool usage
        """
        if request_id not in self._metrics_data["requests"]:
            return
            
        # Update request data
        if tool_id not in self._metrics_data["requests"][request_id]["tools_used"]:
            self._metrics_data["requests"][request_id]["tools_used"].append(tool_id)
        
        # Update tool usage stats
        if "tool_usage" not in self._metrics_data:
            self._metrics_data["tool_usage"] = {}
            
        if tool_id not in self._metrics_data["tool_usage"]:
            self._metrics_data["tool_usage"][tool_id] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_duration_ms": 0,
                "last_used": None,
                "avg_duration_ms": 0
            }
            
        # Update tool metrics
        tool_data = self._metrics_data["tool_usage"][tool_id]
        tool_data["total_calls"] += 1
        if success:
            tool_data["successful_calls"] += 1
        
        if duration_ms:
            tool_data["total_duration_ms"] += duration_ms
            tool_data["avg_duration_ms"] = tool_data["total_duration_ms"] / tool_data["total_calls"]
            
        tool_data["last_used"] = datetime.now(timezone.utc).isoformat()
        
        # Save metrics to disk
        self._save_metrics()
    
    def track_model_usage(self, 
                        request_id: str, 
                        model_id: str,
                        tokens_in: int = 0,
                        tokens_out: int = 0,
                        duration_ms: Optional[int] = None,
                        success: bool = True,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track usage of a model during a request.
        
        Args:
            request_id: ID of the request
            model_id: ID of the model used
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            duration_ms: Time taken for the model response
            success: Whether the model call was successful
            metadata: Additional metadata about model usage
        """
        if request_id not in self._metrics_data["requests"]:
            return
            
        # Update request data
        if model_id not in self._metrics_data["requests"][request_id]["models_used"]:
            self._metrics_data["requests"][request_id]["models_used"].append(model_id)
        
        # Update model usage stats
        if "model_usage" not in self._metrics_data:
            self._metrics_data["model_usage"] = {}
            
        if model_id not in self._metrics_data["model_usage"]:
            self._metrics_data["model_usage"][model_id] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_duration_ms": 0,
                "last_used": None,
                "avg_duration_ms": 0
            }
            
        # Update model metrics
        model_data = self._metrics_data["model_usage"][model_id]
        model_data["total_calls"] += 1
        if success:
            model_data["successful_calls"] += 1
        
        model_data["total_tokens_in"] += tokens_in
        model_data["total_tokens_out"] += tokens_out
        
        if duration_ms:
            model_data["total_duration_ms"] += duration_ms
            model_data["avg_duration_ms"] = model_data["total_duration_ms"] / model_data["total_calls"]
            
        model_data["last_used"] = datetime.now(timezone.utc).isoformat()
        
        # Save metrics to disk
        self._save_metrics()
    
    def get_agent_metrics(self, 
                        agent_id: Optional[str] = None, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get metrics for agent usage.
        
        Args:
            agent_id: ID of agent to get metrics for (or None for all)
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary of agent metrics
        """
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = end_time - timedelta(days=30)
            
        # Convert to ISO format strings for comparison
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        # Filter requests within time range
        filtered_requests = {
            req_id: req_data for req_id, req_data in self._metrics_data.get("requests", {}).items()
            if req_data.get("start_time", "") >= start_iso and req_data.get("start_time", "") <= end_iso
        }
        
        # Collect agent IDs of interest
        agent_ids = [agent_id] if agent_id else list(self._metrics_data.get("agent_usage", {}).keys())
        
        # Compute metrics for each agent
        agent_metrics = {}
        
        for aid in agent_ids:
            if aid not in self._metrics_data.get("agent_usage", {}):
                continue
                
            # Get base metrics
            base_metrics = self._metrics_data["agent_usage"][aid].copy()
            
            # Count requests in time period
            period_requests = 0
            period_successful = 0
            for req_id, req_data in filtered_requests.items():
                if aid in req_data.get("agents_used", []):
                    period_requests += 1
                    if req_data.get("success", False):
                        period_successful += 1
            
            # Add period-specific metrics
            base_metrics["period_requests"] = period_requests
            base_metrics["period_successful"] = period_successful
            base_metrics["period_success_rate"] = period_successful / period_requests if period_requests > 0 else 0
            
            agent_metrics[aid] = base_metrics
        
        return agent_metrics
    
    def get_tool_metrics(self, 
                       tool_id: Optional[str] = None, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get metrics for tool usage.
        
        Args:
            tool_id: ID of tool to get metrics for (or None for all)
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary of tool metrics
        """
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = end_time - timedelta(days=30)
            
        # Convert to ISO format strings for comparison
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        # Filter requests within time range
        filtered_requests = {
            req_id: req_data for req_id, req_data in self._metrics_data.get("requests", {}).items()
            if req_data.get("start_time", "") >= start_iso and req_data.get("start_time", "") <= end_iso
        }
        
        # Collect tool IDs of interest
        tool_ids = [tool_id] if tool_id else list(self._metrics_data.get("tool_usage", {}).keys())
        
        # Compute metrics for each tool
        tool_metrics = {}
        
        for tid in tool_ids:
            if tid not in self._metrics_data.get("tool_usage", {}):
                continue
                
            # Get base metrics
            base_metrics = self._metrics_data["tool_usage"][tid].copy()
            
            # Count calls in time period
            period_calls = 0
            period_successful = 0
            for req_id, req_data in filtered_requests.items():
                if tid in req_data.get("tools_used", []):
                    period_calls += 1
                    if req_data.get("success", False):
                        period_successful += 1
            
            # Add period-specific metrics
            base_metrics["period_calls"] = period_calls
            base_metrics["period_successful"] = period_successful
            base_metrics["period_success_rate"] = period_successful / period_calls if period_calls > 0 else 0
            
            tool_metrics[tid] = base_metrics
        
        return tool_metrics
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            with open(self._storage_path, 'w') as f:
                json.dump(self._metrics_data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics data: {e}")