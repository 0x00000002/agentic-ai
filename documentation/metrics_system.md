# Metrics Tracking System

This document provides an overview of the metrics tracking system implemented in the Agentic-AI framework. The system allows tracking and analysis of agent and tool usage, performance metrics, and request processing.

## Overview

The metrics tracking system consists of the following components:

1. **RequestMetricsService**: Core service for tracking and storing metrics about requests, agents, tools, and models.
2. **MetricsDashboard**: Visualization and reporting tool for analyzing metrics data.
3. **Metrics CLI**: Command-line interface for querying and visualizing metrics.

The system tracks:

- Request processing details and duration
- Agent usage and performance
- Tool usage and performance
- Model usage and token counts

## Installation and Setup

The metrics system is built into the Agentic-AI framework and does not require separate installation. All metrics are automatically stored in:

```
data/metrics/request_metrics.json
```

This location can be customized by providing a different path when initializing the `RequestMetricsService`.

## Usage

### 1. Programmatic Usage

#### Tracking Request Metrics

```python
from src.metrics.request_metrics import RequestMetricsService

# Initialize the metrics service
metrics_service = RequestMetricsService()

# Start tracking a request
request_id = metrics_service.start_request_tracking(
    prompt="User query here",
    metadata={"user_id": "user123"}
)

# Later, end tracking when request is complete
metrics_service.end_request_tracking(
    request_id=request_id,
    success=True  # or False if there was an error
)
```

#### Tracking Agent Usage

```python
# Track when an agent is used
metrics_service.track_agent_usage(
    request_id=request_id,
    agent_id="example_agent",
    confidence=0.85,
    duration_ms=152,
    success=True
)
```

#### Tracking Tool Usage

```python
# Track when a tool is used
metrics_service.track_tool_usage(
    request_id=request_id,
    tool_id="example_tool",
    duration_ms=75,
    success=True
)
```

#### Tracking Model Usage

```python
# Track when a model is used
metrics_service.track_model_usage(
    request_id=request_id,
    model_id="gpt-4",
    tokens_in=250,
    tokens_out=50,
    duration_ms=1200,
    success=True
)
```

#### Getting Metrics

```python
# Get agent metrics
agent_metrics = metrics_service.get_agent_metrics(
    agent_id="example_agent",  # Optional, get all if not specified
    start_time=datetime.now() - timedelta(days=30)
)

# Get tool metrics
tool_metrics = metrics_service.get_tool_metrics(
    tool_id="example_tool",  # Optional, get all if not specified
    start_time=datetime.now() - timedelta(days=30)
)
```

#### Visualization

```python
from src.metrics.dashboard import MetricsDashboard

# Initialize the dashboard
dashboard = MetricsDashboard()

# Generate plots
dashboard.plot_agent_usage(top_n=10, days=30)
dashboard.plot_tool_usage(top_n=10, days=30)

# Generate a performance report
report = dashboard.generate_performance_report(days=30)
```

### 2. Command-Line Interface

The metrics system includes a command-line interface for easy analysis of metrics data.

#### Basic Usage

```bash
# Show summary of metrics
python src/metrics/metrics_cli.py summary

# Show agent metrics
python src/metrics/metrics_cli.py agents

# Show metrics for a specific agent
python src/metrics/metrics_cli.py agents --agent="orchestrator"

# Show tool metrics
python src/metrics/metrics_cli.py tools

# Show details for a specific request
python src/metrics/metrics_cli.py request REQUEST_ID

# Generate a performance report
python src/metrics/metrics_cli.py report
```

#### Advanced Usage

```bash
# Generate and save a plot of agent metrics
python src/metrics/metrics_cli.py agents --plot --output=agent_metrics.png

# Generate and save a plot of tool metrics
python src/metrics/metrics_cli.py tools --plot --output=tool_metrics.png

# Generate a JSON performance report
python src/metrics/metrics_cli.py report --output=performance_report.json

# Show metrics for the last 7 days
python src/metrics/metrics_cli.py summary --days=7
```

## Integration with Orchestrator

The metrics system is fully integrated with the Orchestrator agent. When the Orchestrator processes a request, it:

1. Automatically generates a request ID if not provided
2. Tracks the start and end of each request
3. Tracks which agents and tools are used
4. Records model usage and selection
5. Tracks success or failure of each component
6. Includes the request ID and metrics metadata in the response

## Integration with Tool Registry

The metrics system is also integrated with the Tool Registry. When a tool is executed:

1. The execution time is tracked
2. The success or failure is recorded
3. The tool usage is associated with the current request ID

## Extending the Metrics System

### Adding New Metrics

To add new metrics to track:

1. Add new fields to the appropriate dictionaries in `RequestMetricsService`
2. Add methods to update and retrieve the new metrics
3. Update visualization methods in `MetricsDashboard` if needed

### Integrating with Custom Agents

To integrate the metrics system with your custom agent:

```python
from src.metrics.request_metrics import RequestMetricsService

class MyCustomAgent:
    def process_request(self, request):
        # Extract request_id if it exists
        request_id = request.get("request_id")

        # Initialize metrics service
        metrics_service = RequestMetricsService()

        # Record start time
        start_time = time.time()

        try:
            # Process the request
            result = self._do_processing(request)

            # Track usage
            metrics_service.track_agent_usage(
                request_id=request_id,
                agent_id=self.agent_id,
                duration_ms=int((time.time() - start_time) * 1000),
                success=True
            )

            return result
        except Exception as e:
            # Track failure
            metrics_service.track_agent_usage(
                request_id=request_id,
                agent_id=self.agent_id,
                duration_ms=int((time.time() - start_time) * 1000),
                success=False,
                metadata={"error": str(e)}
            )
            raise
```

## Troubleshooting

### Common Issues

1. **Missing metrics data**: Ensure the data directory exists and is writable.
2. **Request ID not tracked**: Make sure the request_id is passed correctly between components.
3. **CLI import errors**: Run the CLI script from the project root directory.

### Debugging

To debug metrics tracking, you can:

1. Inspect the metrics JSON file directly: `data/metrics/request_metrics.json`
2. Add detailed metadata to track more information
3. Use the CLI's request command to inspect specific requests: `python src/metrics/metrics_cli.py request REQUEST_ID`

## Best Practices

1. Always include request_id when passing requests between components
2. Use consistent agent_id and tool_id values for accurate tracking
3. Track both successes and failures for complete analytics
4. Include relevant metadata for easier debugging and analysis
5. Regularly review metrics to identify performance issues and opportunities for optimization
