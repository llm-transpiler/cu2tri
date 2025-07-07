# Quick Start Guide - GPU Management System

## 1. Start the Server

### Basic Usage
```bash
python start_server.py
```

### Custom Configuration
```bash
# Exclude specific GPUs and set max parallel tasks
python start_server.py --exclude-gpus 0 --max-parallel 8

# Debug mode with custom port
python start_server.py --port 8090 --log-level DEBUG
```

## 2. Submit Tasks

### Exclusive Task (独占GPU)
```python
import requests

# Submit a task that needs exclusive GPU access
response = requests.post("http://localhost:8080/tasks/submit", json={
    "task_type": "exclusive",
    "name": "Training Task",
    "description": "Model training",
    "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
    "function_name": "simple_gpu_task",
    "kwargs": {"duration": 10.0}
})

task_id = response.json()["task_id"]
print(f"Exclusive task submitted: {task_id}")
```

### Shared Task (共享GPU)
```python
# Submit a task that can share GPU with others
response = requests.post("http://localhost:8080/tasks/submit", json={
    "task_type": "shared",
    "name": "Inference Task",
    "description": "Model inference",
    "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
    "function_name": "simple_gpu_task",
    "kwargs": {"duration": 5.0},
    "gpu_id": 1  # Optional: specify GPU
})
```

## 3. Check Status

### Task Status
```python
response = requests.get(f"http://localhost:8080/tasks/{task_id}")
status = response.json()

print(f"Status: {status['status']}")
print(f"Task Type: {status['task_type']}")
print(f"GPU: {status['gpu_id']}")
print(f"Execution Time: {status['execution_time_seconds']:.2f}s")
```

### System Status
```python
response = requests.get("http://localhost:8080/status")
system_status = response.json()

# Check task statistics
stats = system_status['system_stats']
print(f"Exclusive tasks running: {stats['total_running_exclusive']}")
print(f"Shared tasks running: {stats['total_running_shared']}")

# Check GPU details
for gpu_id, gpu_info in system_status['gpus'].items():
    queue = gpu_info['queue']
    print(f"\nGPU {gpu_id}:")
    print(f"  Exclusive queue: {queue['exclusive_queue_size']}")
    print(f"  Shared queue: {queue['shared_queue_size']}")
    print(f"  Running: {1 if queue['running_exclusive'] else 0} exclusive, "
          f"{queue['running_shared_count']} shared")
```

## 4. Run Examples

```bash
# Start server
python start_server.py

# In another terminal, run the task example
python examples/task_example.py
```

## Key Concepts

### Task Types

1. **Exclusive Tasks** (`task_type: "exclusive"`):
   - Require full GPU access
   - No other tasks can run simultaneously
   - Good for: training, benchmarking, memory-intensive work

2. **Shared Tasks** (`task_type: "shared"`):
   - Can run in parallel with other shared tasks
   - Up to `max_parallel_task_num` can run together
   - Good for: inference, small computations, development

### Task Scheduling Rules

1. If an **exclusive task** is running → wait for it to complete
2. If **no tasks** are running → can start either exclusive or shared tasks
3. If **shared tasks** are running:
   - Can add more shared tasks (up to limit)
   - Exclusive tasks must wait for all shared tasks to complete

### GPU Assignment

- **Automatic**: System selects GPU with lowest load
- **Manual**: Specify `gpu_id` in task submission

### GPU IDs

- Based on CUDA_VISIBLE_DEVICES
- If CUDA_VISIBLE_DEVICES="2,3", then GPU IDs are 0,1

## Common Commands

```bash
# Check available task types
curl http://localhost:8080/task-types

# Get available GPUs
curl http://localhost:8080/gpus/available

# Cancel a task
curl -X DELETE http://localhost:8080/tasks/{task_id}

# Health check
curl http://localhost:8080/health
```

## Tips

1. **For training workloads**: Use exclusive tasks to ensure consistent performance
2. **For inference**: Use shared tasks to maximize GPU utilization
3. **Mixed workloads**: Exclusive tasks will wait for shared tasks to complete
4. **Debugging**: Use `--log-level DEBUG` to see detailed scheduling decisions 