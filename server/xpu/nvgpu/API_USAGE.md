# GPU Management API Usage Guide

This API now supports **arbitrary task submission** with **GPU ID specification**, not just predefined kernel testing tasks.

## API Endpoints

### 1. General Task Submission (`/tasks/submit`)

Submit any Python function as a task with optional GPU specification:

```bash
curl -X POST "http://localhost:8081/tasks/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "name": "Simple CPU Task",
    "description": "A basic CPU task for testing",
    "module_path": "server.xpu.nvgpu.example_tasks",
    "function_name": "simple_cpu_task",
    "args": ["Hello World", 5],
    "kwargs": {},
    "max_wait_time_minutes": 10,
    "preferred_gpu_id": 2,
    "allow_fallback": true
  }'
```

### 2. Kernel Testing (Convenience Endpoint) (`/tasks/submit-kernel`)

For backward compatibility with kernel testing, now with GPU specification:

```bash
curl -X POST "http://localhost:8081/tasks/submit-kernel" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "performance",
    "name": "Kernel Performance Test",
    "description": "Test kernel performance",
    "kernel_dir": "/path/to/kernel",
    "model_name": "test_model",
    "max_wait_time_minutes": 30,
    "preferred_gpu_id": 0,
    "allow_fallback": false
  }'
```

## GPU Specification Parameters

- **preferred_gpu_id**: (Optional) Specify which GPU you want to use (0, 1, 2, etc.)
- **allow_fallback**: (Default: false) Whether to allow the task to run on other GPUs if the preferred one is unavailable
- **require_same_gpu_type**: (Default: true) When fallback is enabled, require the fallback GPU to have the same type (e.g., H100, L20) as the preferred GPU

## Task Types

- **functional**: Tasks that can share GPU resources with other functional tasks
- **performance**: Tasks that require exclusive access to a single GPU

## GPU Selection Logic

The system uses a **two-pass scheduling algorithm** with **GPU type matching**:

1. **First Pass**: Schedule tasks that specify a preferred GPU ID
   - Tasks with `preferred_gpu_id` get priority on their preferred GPU
   - Higher chance of getting the exact GPU you requested

2. **Second Pass**: Schedule remaining tasks using fallback logic with type matching
   - Tasks without GPU preference can use any available GPU
   - Tasks with `allow_fallback=true` can use alternative GPUs if preferred is busy
   - **GPU Type Matching**: If `require_same_gpu_type=true`, fallback GPUs must have the same type as the preferred GPU
     - Example: If you prefer GPU 0 (H100) but it's busy, the system will only consider other H100 GPUs for fallback
     - This ensures consistent performance characteristics across GPU types

## Example Tasks with GPU Specification

### Specify Exact GPU (No Fallback)

```python
# Force task to run on GPU 1 only
{
  "task_type": "performance",
  "module_path": "server.xpu.nvgpu.example_tasks",
  "function_name": "gpu_compute_task",
  "args": [1000],
  "kwargs": {},
  "preferred_gpu_id": 1,
  "allow_fallback": false
}
```

### Prefer GPU with Fallback (Same Type Only)

```python
# Prefer GPU 0, but only allow fallback to other H100 GPUs
{
  "task_type": "functional",
  "module_path": "server.xpu.nvgpu.example_tasks",
  "function_name": "memory_intensive_task",
  "args": [],
  "kwargs": {"memory_gb": 2.0},
  "preferred_gpu_id": 0,
  "allow_fallback": true,
  "require_same_gpu_type": true  # Only use other H100s if GPU 0 is busy
}
```

### Prefer GPU with Fallback (Any Type)

```python
# Prefer GPU 0, but allow fallback to any available GPU type
{
  "task_type": "functional",
  "module_path": "server.xpu.nvgpu.example_tasks",
  "function_name": "simple_cpu_task",
  "args": ["Processing", 5],
  "kwargs": {},
  "preferred_gpu_id": 0,
  "allow_fallback": true,
  "require_same_gpu_type": false  # Allow fallback to different GPU types
}
```

### No GPU Preference

```python
# Let system choose any available GPU
{
  "task_type": "functional",
  "module_path": "server.xpu.nvgpu.example_tasks",
  "function_name": "async_cpu_task",
  "args": ["Processing", 5],
  "kwargs": {}
  # preferred_gpu_id not specified - any GPU is fine
}
```

## GPU Status Checking

Before submitting tasks, you can check GPU availability:

```bash
# Check all GPU status
curl "http://localhost:8081/gpus"

# Example response:
{
  "0": {
    "device_id": 0,
    "name": "NVIDIA H100 PCIe",
    "memory_usage_ratio": 0.1,
    "available_for_functional": true,
    "available_for_performance": true
  },
  "1": {
    "device_id": 1,
    "name": "NVIDIA H100 PCIe", 
    "memory_usage_ratio": 0.8,
    "available_for_functional": false,
    "available_for_performance": false
  }
}
```

## Best Practices

1. **Performance Tasks**: Use specific GPU ID with `allow_fallback=false` for consistent performance
2. **Functional Tasks**: Use GPU preference with `allow_fallback=true` for better resource utilization
3. **Load Balancing**: Don't always use GPU 0 - distribute tasks across available GPUs
4. **Memory-Intensive Tasks**: Check GPU memory status before submitting large tasks

## Task Status Monitoring

Task status now includes GPU assignment information:

```bash
curl "http://localhost:8081/tasks/{task_id}"

# Response includes:
{
  "task_id": "...",
  "gpu_id": 2,  # Which GPU is actually running the task
  "status": "running",
  "preferred_gpu_id": 1,  # Which GPU was originally requested
  ...
}
```

This flexible GPU specification system gives you fine-grained control over GPU resource allocation while maintaining automatic fallback capabilities! 