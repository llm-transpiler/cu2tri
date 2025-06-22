# NVIDIA GPU Management System

A sophisticated GPU resource management and task scheduling system designed for kernel testing and performance evaluation.

## Features

- **Dual Task Types**: Support for both functional and performance tasks
- **Smart Scheduling**: Automatic GPU resource allocation and task scheduling
- **GPU Type Detection**: Automatic detection of H100, L20, RTX 6000 Ada, and other GPU types
- **Memory Management**: Intelligent memory usage monitoring and allocation
- **Environment Control**: Support for `CUDA_VISIBLE_DEVICES` environment variable
- **Task Queue Management**: FIFO queuing with priority handling and timeout management
- **Comprehensive Logging**: Detailed logging with separate log directories for each task

## Architecture

### Core Components

1. **GPUManager**: Main orchestrator for GPU resources and task scheduling
2. **TaskQueue**: Queue management for functional and performance tasks
3. **GPUInfo**: GPU information detection and specification management
4. **GPUTaskExecutor**: Specialized executor for kernel testing tasks

### Task Types

- **Functional Tasks**: Multiple tasks can run simultaneously on the same GPU if memory allows (< 2/3 usage)
- **Performance Tasks**: Require exclusive GPU access for accurate benchmarking

## Quick Start

### Basic Usage

```python
import asyncio
from server.xpu.nvgpu import GPUManager, TaskType
from eval_.kernelbench_c import create_correctness_task

async def main():
    # Create and start GPU manager
    async with GPUManager() as gpu_manager:
        
        # Submit a functional task
        task_id = await gpu_manager.submit_task(
            task_type=TaskType.FUNCTIONAL,
            name="Matrix Multiplication Test",
            description="Compare Triton vs PyTorch correctness",
            execute_func=create_correctness_task,
            args=("path/to/kernel/dir", "triton_vs_torch", "test_name"),
            estimated_memory_mb=2000
        )
        
        # Monitor task progress
        while True:
            status = await gpu_manager.get_task_status(task_id)
            if status['status'] in ['completed', 'failed']:
                break
            await asyncio.sleep(1)

asyncio.run(main())
```

### API Server

Start the REST API server:

```bash
cd server/xpu/nvgpu
python api_server.py
```

API endpoints:
- `GET /status` - System status
- `GET /gpus` - GPU information
- `GET /queue` - Task queue status
- `POST /tasks/submit` - Submit new task
- `GET /tasks/{task_id}` - Get task status
- `DELETE /tasks/{task_id}` - Cancel task

### Example API Usage

```bash
# Submit a correctness task
curl -X POST "http://localhost:8080/tasks/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "name": "Matrix Multiplication Correctness",
    "description": "Compare Triton vs PyTorch",
    "kernel_dir": "cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_",
    "comparison_type": "triton_vs_torch",
    "estimated_memory_mb": 2000
  }'

# Check task status
curl "http://localhost:8080/tasks/{task_id}"

# Get GPU status
curl "http://localhost:8080/gpus"
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the system
- Example: `CUDA_VISIBLE_DEVICES=0,1` to use only GPUs 0 and 1

### GPU Manager Configuration

```python
gpu_manager = GPUManager(
    refresh_interval_seconds=10,    # GPU status refresh interval
    max_task_wait_minutes=30,       # Maximum task wait time
    log_dir="server/logs/xpu/nvgpu" # Log directory
)
```

## Supported GPU Types

| GPU Type | Memory | Compute Capability | Tensor Cores | FP32 TFLOPS |
|----------|--------|-------------------|--------------|-------------|
| H100     | 80GB   | 9.0               | Yes          | 51.0        |
| L20      | 48GB   | 8.6               | Yes          | 59.7        |
| RTX 6000 Ada | 48GB | 8.9             | Yes          | 91.1        |

## Task Execution Flow

1. **Task Submission**: Tasks are submitted to appropriate queues based on type
2. **Resource Allocation**: GPU manager monitors GPU availability and memory usage
3. **Task Scheduling**: Tasks are assigned to suitable GPUs based on requirements
4. **Execution**: Tasks run with isolated CUDA environments
5. **Completion**: Results are logged and resources are freed

## Kernel Testing Integration

The system integrates with the KernelBench-C evaluation framework to provide:

- **Correctness Comparison**: Compare results between Triton, CUDA, and PyTorch implementations
- **Performance Benchmarking**: Measure execution times with statistical analysis
- **Comprehensive Testing**: Full evaluation including all comparison types

### Supported Comparisons

- `triton_vs_torch`: Triton kernel vs PyTorch reference
- `triton_vs_cuda`: Triton kernel vs CUDA implementation  
- `cuda_vs_torch`: CUDA implementation vs PyTorch reference
- `comprehensive`: All comparisons combined

## Logging

Logs are organized hierarchically:

```
server/logs/xpu/nvgpu/
├── correctness/
│   └── {task_id}/
├── performance/
│   └── {task_id}/
└── comprehensive/
    └── {task_id}/
```

Each task gets its own log directory with detailed execution information.

## Best Practices

1. **Memory Estimation**: Provide accurate memory estimates for better scheduling
2. **Task Naming**: Use descriptive names for easier monitoring
3. **Timeout Setting**: Set appropriate timeouts based on expected execution time
4. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` to control GPU availability
5. **Resource Monitoring**: Monitor GPU usage to optimize task scheduling

## Error Handling

The system provides comprehensive error handling:

- **Task Timeout**: Tasks exceeding wait time are automatically cancelled
- **GPU Failures**: Failed GPU operations are logged and tasks are rescheduled
- **Memory Overflow**: Tasks are rejected if insufficient memory is available
- **Execution Errors**: Detailed error information is captured and logged

## Performance Considerations

- **Functional Tasks**: Can share GPU resources if memory allows
- **Performance Tasks**: Get exclusive GPU access for accurate measurements
- **Queue Priority**: Performance tasks have higher priority when GPUs become available
- **Memory Threshold**: Functional tasks limited to 67% of GPU memory to prevent conflicts

## Monitoring and Debugging

Use the API endpoints or direct GPU manager methods to monitor:

- GPU utilization and memory usage
- Task queue lengths and wait times
- Running task distribution across GPUs
- System health and error rates

For debugging, check the logs in `server/logs/xpu/nvgpu/` for detailed execution traces. 