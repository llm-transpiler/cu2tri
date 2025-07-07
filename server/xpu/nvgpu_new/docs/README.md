# GPU Management System

A GPU resource management system with task-based scheduling supporting exclusive and shared GPU access.

## Overview

This system provides efficient GPU resource management with the following key features:

### Key Features

- **Task-Based GPU Access**: Tasks declare whether they need exclusive or shared GPU access
- **Dynamic Scheduling**: GPUs automatically schedule tasks based on their requirements
- **Direct GPU Assignment**: Tasks can be bound to specific GPU IDs (based on CUDA_VISIBLE_DEVICES)
- **Simple Load Balancing**: Automatic GPU selection based on load when no GPU is specified
- **Per-GPU Management**: Each GPU has its own dedicated manager and task queue
- **CUDA Error Protection**: Automatic cooldown mechanism to protect GPUs from error cascades
- **Health Monitoring**: Lightweight, isolated GPU health checks with minimal memory footprint

### Task Types

1. **Exclusive Tasks (独占GPU)**: Require exclusive GPU access, no other tasks can run simultaneously
2. **Shared Tasks (共享GPU)**: Can run in parallel with other shared tasks (up to configured limit)

## Quick Start

### 1. Installation

Ensure you have the required dependencies:
```bash
pip install fastapi uvicorn torch pydantic
```

### 2. Start the Server

```bash
# Basic start
python start_server.py

# Exclude certain GPUs
python start_server.py --exclude-gpus 0,1

# Configure max parallel shared tasks
python start_server.py --max-parallel 8

# Custom configuration
python start_server.py --port 8080 --max-parallel 6 --log-level DEBUG
```

### 3. Check System Status

```bash
curl http://localhost:8080/status
```

## API Usage

### Submit an Exclusive Task

```python
import requests

# Exclusive task - requires full GPU
response = requests.post("http://localhost:8080/tasks/submit", json={
    "task_type": "exclusive",
    "name": "Training Task",
    "description": "Model training requiring full GPU",
    "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
    "function_name": "train_model",
    "kwargs": {"epochs": 10}
})

task_id = response.json()["task_id"]
```

### Submit a Shared Task

```python
# Shared task - can run with other shared tasks
response = requests.post("http://localhost:8080/tasks/submit", json={
    "task_type": "shared",
    "name": "Inference Task",
    "description": "Model inference task",
    "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
    "function_name": "run_inference",
    "kwargs": {"batch_size": 32},
    "gpu_id": 1  # Optional: specify GPU 1
})
```

### Check Task Status

```python
response = requests.get(f"http://localhost:8080/tasks/{task_id}")
status = response.json()

print(f"Status: {status['status']}")
print(f"GPU: {status['gpu_id']}")
print(f"Task Type: {status['task_type']}")
print(f"Duration: {status['execution_time_seconds']:.2f}s")
```

### Check System Status

```python
response = requests.get("http://localhost:8080/status")
system_status = response.json()

stats = system_status['system_stats']
print(f"Exclusive tasks queued: {stats['total_exclusive_queued']}")
print(f"Shared tasks queued: {stats['total_shared_queued']}")
print(f"Running exclusive: {stats['total_running_exclusive']}")
print(f"Running shared: {stats['total_running_shared']}")
```

## CUDA Error Protection & Health Monitoring

### CUDA Error Cooldown Mechanism

The system includes a built-in CUDA error cooldown mechanism to protect GPUs from cascading errors:

**How it works:**
1. When a CUDA error occurs during task execution, the system automatically detects it
2. The affected GPU enters a cooldown period (default: 30 seconds)
3. During cooldown, no new tasks are scheduled to that GPU
4. After cooldown expires, the system performs health check to verify GPU recovery
5. If health check passes, the GPU becomes available again
6. If health check fails, the GPU enters another cooldown period

**Error Detection:**
The system automatically detects various CUDA errors including:
- Out of memory errors
- Driver errors
- Kernel launch failures
- Device assert errors
- Memory access violations

### GPU Health Monitoring

The system performs lightweight health checks on each GPU:

**Health Check Features:**
- **Complete Isolation**: Uses subprocess execution to avoid memory leaks
- **Minimal Operations**: Only performs simple tensor additions to verify GPU functionality
- **Zero Memory Footprint**: Guaranteed cleanup with no persistent CUDA contexts
- **Error-Driven**: Only runs after CUDA errors during cooldown recovery
- **Task-Safe**: Only runs when GPU has no active tasks
- **Fast Detection**: Quickly identifies GPU hardware or driver issues

**Health Check Process:**
1. Triggered only after cooldown period expires
2. Ensures GPU has no running tasks before starting
3. Creates isolated subprocess with clean environment
4. Performs minimal tensor operations (addition, matrix multiplication)
5. Verifies computational correctness
6. Forces complete memory cleanup
7. Reports health status without leaving any GPU memory residue
8. If check fails, extends cooldown period

### Configuration Options

```bash
# Start server with custom protection settings
python start_server.py --error-cooldown 60

# Available options:
# --error-cooldown: CUDA error cooldown period in seconds (default: 30)
```

### Monitoring Protection Status

```python
# Check system protection status
response = requests.get("http://localhost:8080/status")
status = response.json()

print(f"Available GPUs: {status['available_gpus']}")
print(f"GPUs in cooldown: {status['cooldown_gpus']}")
print(f"Unhealthy GPUs: {status['unhealthy_gpus']}")

# Check individual GPU status
for gpu_id, gpu_status in status['gpus'].items():
    print(f"GPU-{gpu_id}:")
    print(f"  Available: {gpu_status['is_available']}")
    print(f"  In cooldown: {gpu_status['is_in_cooldown']}")
    print(f"  Last health check passed: {gpu_status['last_health_check_passed']}")
    print(f"  Error count: {gpu_status['cuda_error_count']}")
    print(f"  Last health check: {gpu_status['last_health_check']}")
    print(f"  Cooldown reason: {gpu_status['cooldown_reason']}")
```

## Configuration

### Command Line Options

```bash
python start_server.py --help
```

Key options:
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--exclude-gpus`: Comma-separated GPU IDs to exclude
- `--max-parallel`: Max parallel shared tasks per GPU (default: 4)
- `--error-cooldown`: CUDA error cooldown period in seconds (default: 30)
- `--refresh-interval`: GPU manager refresh interval in seconds (default: 30)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Configuration File

See `config.yaml` for detailed configuration options:

```yaml
task_dispatcher:
  # Exclude specific GPUs
  excluded_gpu_ids: [0]
  
  # Max parallel shared tasks per GPU
  max_parallel_task_num: 4
```

## GPU ID Mapping

The system uses CUDA device IDs which correspond to the `CUDA_VISIBLE_DEVICES` environment variable:

- If `CUDA_VISIBLE_DEVICES=0,1,2,3` (or not set), GPU IDs are 0, 1, 2, 3
- If `CUDA_VISIBLE_DEVICES=2,3`, GPU IDs are 0, 1 (mapped to physical GPUs 2, 3)
- The mapping is handled automatically by the system

## Task Scheduling Logic

### GPU Task Queue Behavior

Each GPU manages its own task queue with the following scheduling rules:

1. **If an exclusive task is running**: Wait for it to complete
2. **If no tasks are running**: 
   - Start an exclusive task if one is waiting, OR
   - Start shared tasks (up to max_parallel_task_num)
3. **If shared tasks are running**:
   - Continue running shared tasks up to the limit
   - Wait for all shared tasks to complete before starting an exclusive task

### Load Balancing

When no GPU is specified, the system selects the GPU with the lowest load score based on:
- Memory usage (50% weight)
- GPU utilization (30% weight)
- Temperature (20% weight)
- Queue sizes and running tasks

## Examples

### Running Examples

```bash
# Start the server
python start_server.py

# In another terminal, run examples
python examples/task_example.py

# Test CUDA error protection and health monitoring
python examples/error_handling_example.py
```

### CUDA Error Protection Examples

```python
# Example: Task that might cause CUDA errors
async def risky_gpu_task():
    """Task that might cause CUDA errors"""
    import torch
    
    # This might cause out of memory error
    try:
        huge_tensor = torch.randn(100000, 100000, device='cuda')
        result = torch.mm(huge_tensor, huge_tensor)
        return {'status': 'success', 'result': result.sum().item()}
    except RuntimeError as e:
        # CUDA errors are automatically detected and handled
        raise e

# Submit task that might trigger cooldown
response = requests.post("http://localhost:8080/tasks/submit", json={
    "task_type": "exclusive",
    "name": "Risky Task",
    "description": "Task that might cause CUDA errors",
    "module_path": "your_module",
    "function_name": "risky_gpu_task"
})

# Check system status to see cooldown effect
response = requests.get("http://localhost:8080/status")
status = response.json()

if status['cooldown_gpus'] > 0:
    print("Some GPUs are in cooldown mode due to CUDA errors")
    for gpu_id, gpu_status in status['gpus'].items():
        if gpu_status['is_in_cooldown']:
            print(f"GPU-{gpu_id} in cooldown, expires: {gpu_status['cooldown_expires_at']}")
```

### Health Monitoring Examples

```python
# Check GPU health status
response = requests.get("http://localhost:8080/status")
status = response.json()

print("=== GPU Health Status ===")
for gpu_id, gpu_status in status['gpus'].items():
    print(f"GPU-{gpu_id}:")
    print(f"  Health Status: {'Healthy' if gpu_status['last_health_check_passed'] else 'Unhealthy'}")
    print(f"  Last Health Check: {gpu_status['last_health_check']}")
    print(f"  Health Error: {gpu_status['last_health_error'] or 'None'}")
    print(f"  Memory Usage: {gpu_status['memory_usage_ratio']:.1%}")
    print(f"  Temperature: {gpu_status['temperature_c']}°C")
    print(f"  Available: {'Yes' if gpu_status['is_available'] else 'No'}")
    print(f"  Cooldown: {'Yes' if gpu_status['is_in_cooldown'] else 'No'}")
    print()

# System-wide health statistics
print("=== System Health Summary ===")
print(f"Total GPUs: {status['total_gpus']}")
print(f"Healthy GPUs: {status['total_gpus'] - status['unhealthy_gpus']}")
print(f"GPUs in Cooldown: {status['cooldown_gpus']}")
print(f"Configuration:")
print(f"  Error Cooldown: {status['configuration']['error_cooldown_seconds']} seconds")
```

### Custom Task Creation

```python
# Exclusive task example
async def train_model(epochs: int = 10) -> dict:
    """Training task requiring exclusive GPU access"""
    import torch
    
    # This task needs the full GPU
    model = create_large_model()
    dataset = load_dataset()
    
    for epoch in range(epochs):
        # Training logic
        pass
    
    return {'status': 'completed', 'epochs': epochs}

# Shared task example  
async def run_inference(batch_size: int = 32) -> dict:
    """Inference task that can share GPU"""
    import torch
    
    # This task uses limited GPU resources
    model = load_model()
    results = model.predict(batch)
    
    return {'predictions': results}
```

## API Endpoints

### Task Management
- `POST /tasks/submit` - Submit a new task
- `GET /tasks/{task_id}` - Get task status
- `DELETE /tasks/{task_id}` - Cancel a task
- `GET /tasks` - Get system-wide task statistics

### System Monitoring
- `GET /status` - Complete system status
- `GET /gpus` - GPU status information
- `GET /gpus/available` - Available GPU IDs
- `GET /task-types` - Available task types
- `GET /health` - Health check

### Service Info
- `GET /` - Service information

## Architecture

### Core Components

1. **API Server**: HTTP REST API for task submission and monitoring
2. **Task Dispatcher**: Manages GPU selection and task distribution
3. **GPU Managers**: Per-GPU resource management
4. **GPU Task Queues**: Per-GPU task scheduling with exclusive/shared task handling

### Task Flow

1. Client submits task via REST API with task type (exclusive/shared)
2. Dispatcher selects GPU (specified or auto-selected based on load)
3. Task is queued in the selected GPU's appropriate queue
4. GPU manager executes tasks based on scheduling rules:
   - **Exclusive tasks**: Run alone with full GPU access
   - **Shared tasks**: Run in parallel up to max_parallel_task_num
5. Results are returned when task completes

## Use Cases

### Exclusive Task Use Cases
- Model training
- Performance benchmarking
- Memory-intensive computations
- Tasks requiring consistent GPU performance

### Shared Task Use Cases
- Model inference
- Batch processing
- Development and testing
- Multiple small computations

## Performance Considerations

- **Exclusive Tasks**: Guarantee full GPU resources but may underutilize if not fully using GPU
- **Shared Tasks**: Better GPU utilization but tasks may compete for resources
- **Task Isolation**: Each task runs with `CUDA_VISIBLE_DEVICES` set to its assigned GPU
- **Queue Management**: Exclusive tasks have priority once all shared tasks complete

## Troubleshooting

### Common Issues

1. **No GPUs Available**
   - Check excluded GPU list
   - Verify nvidia-smi output
   - Check GPU memory usage (< 80% required)

2. **Tasks Not Starting**
   - Check GPU availability (memory < 80%, temp < 85°C)
   - Verify the specified GPU exists and isn't excluded
   - Check task queue status via `/status` endpoint

3. **Connection Errors**
   - Ensure server is running
   - Check port configuration
   - Verify firewall settings

### Debug Mode

```bash
python start_server.py --log-level DEBUG
```

This provides detailed logging for:
- Task dispatch decisions
- GPU status changes
- Queue operations
- Task execution details

## License

This project follows the same license as the parent project. 