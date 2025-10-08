# NVGPU Server Design Architecture

[English documentation for code implementation details]

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [GPU Mode System](#gpu-mode-system)
3. [Task Type System](#task-type-system)
4. [Scheduler Design](#scheduler-design)
5. [Complete Workflow](#complete-workflow)
6. [Design Rationale](#design-rationale)
7. [Code Reference](#code-reference)

---

## Core Concepts

### Critical Distinction: GPU Mode vs Task Type

**These are TWO INDEPENDENT concepts:**

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  GPU Mode (exclusive/shared)                                 │
│  ↓                                                            │
│  Controls HOW MANY tasks can run concurrently on a GPU       │
│  ↓                                                            │
│  Set by: Administrator via API                               │
│  ↓                                                            │
│  Affects: Task scheduling and GPU resource allocation        │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Task Type (functional/performance)                          │
│  ↓                                                            │
│  Labels the PURPOSE of a task                                │
│  ↓                                                            │
│  Set by: User when submitting task                           │
│  ↓                                                            │
│  Affects: NOTHING in scheduling logic (just metadata)        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Point
**GPU mode switching is MANUAL, not automatic!**

- Mode does NOT change based on what tasks are running
- Administrator explicitly calls API to change mode
- Running tasks continue unaffected
- New tasks follow the new mode rules

---

## GPU Mode System

### Definition

```python
# models.py Line 8-10
class GPUMode(str, Enum):
    EXCLUSIVE = "exclusive"  # Only 1 task at a time
    SHARED = "shared"        # Multiple tasks allowed
```

### Mode Properties

#### 1. EXCLUSIVE Mode

**Purpose:** Ensure a task has complete GPU access

**Behavior:**
- Maximum 1 task running at any time
- Other tasks must wait in queue
- No resource sharing

**Use Cases:**
- Performance benchmarking (need stable environment)
- Memory-intensive tasks (need all VRAM)
- Tasks requiring predictable performance

**Code Implementation:**
```python
# models.py Line 23-36
def can_accept_task(self) -> bool:
    """Check if GPU can accept a new task."""
    if self.status != GPUStatus.ONLINE:
        return False
    
    if self.mode == GPUMode.EXCLUSIVE:
        # EXCLUSIVE: Only accept if NO tasks running
        return len(self.running_tasks) == 0  # ← KEY LOGIC
    
    # SHARED mode checks below
    if len(self.running_tasks) >= self.max_concurrent_tasks:
        return False
    
    return self.current_memory_usage < self.memory_threshold
```

#### 2. SHARED Mode

**Purpose:** Maximize GPU utilization

**Behavior:**
- Multiple tasks can run concurrently
- Controlled by two limits:
  - `max_concurrent_tasks`: e.g., 3 tasks
  - `memory_threshold`: e.g., 75% VRAM usage

**Use Cases:**
- Functional tests (lightweight)
- Parallel execution for throughput
- Development/testing workflows

**Code Implementation:**
```python
# models.py Line 31-36
# ... (continuing from above)
    
    # SHARED mode constraints
    if len(self.running_tasks) >= self.max_concurrent_tasks:
        return False  # ← Constraint 1: Task count limit
    
    return self.current_memory_usage < self.memory_threshold  # ← Constraint 2: Memory limit
```

### Mode Switching Mechanism

#### When/How Mode Changes

**Mode changes ONLY when administrator explicitly calls API:**

```python
# Example API call
import requests
response = requests.put(
    "http://localhost:8080/gpus/0/mode",
    json={"mode": "exclusive"}
)
```

**Implementation:**
```python
# gpu_manager.py Line 100-108
def set_gpu_mode(self, gpu_id: int, mode: GPUMode) -> bool:
    """Set GPU mode.
    
    KEY BEHAVIOR:
    - Only changes the 'mode' attribute
    - Does NOT stop/interrupt running tasks
    - New scheduling decisions use the new mode immediately
    """
    with self.lock:
        if gpu_id not in self.gpus:
            return False
        self.gpus[gpu_id].mode = mode
        logger.info(f"GPU {gpu_id} mode set to {mode.value}")
        return True
```

**Critical Design Decision:**
```
Q: What happens to running tasks when mode switches?
A: NOTHING! They continue running.

Reason: Safe, non-disruptive mode switching
```

#### Scenario 1: Shared → Exclusive (While Tasks Running)

```
Timeline:
─────────────────────────────────────────────────────────────
10:00  │ GPU 0: mode=SHARED, running_tasks=[T1, T2, T3]
       │
10:01  │ Admin calls: set_gpu_mode(0, "exclusive")
       │ ✓ Mode changed immediately
       │ ✓ T1, T2, T3 continue running (NOT interrupted)
       │
10:02  │ User submits: Task T4
       │ Scheduler checks: can_accept_task()
       │   → mode == EXCLUSIVE
       │   → len(running_tasks) == 3 > 0
       │   → return False
       │ ✓ T4 goes to pending queue
       │
10:05  │ T1 completes
       │ running_tasks=[T2, T3]
       │ Scheduler checks: can_accept_task()
       │   → len(running_tasks) == 2 > 0
       │   → return False
       │ ✓ T4 still pending
       │
10:08  │ T2 completes
       │ running_tasks=[T3]
       │ Still 1 task → T4 still pending
       │
10:10  │ T3 completes
       │ running_tasks=[]
       │ Scheduler checks: can_accept_task()
       │   → len(running_tasks) == 0
       │   → return True ✓
       │ ✓ T4 starts execution!
─────────────────────────────────────────────────────────────
```

**Code Flow:**
```python
# scheduler.py Line 28-43
def _schedule_round(self):
    """Run one round of task scheduling."""
    while True:
        task = self.task_queue.pop_pending_task()
        if task is None:
            break
        
        # Determine target GPU
        gpu_id = task.gpu_id if task.gpu_id is not None else None
        
        # Find available GPU (respects current mode)
        available_gpu = self.gpu_manager.find_available_gpu(gpu_id)
        
        if available_gpu is None:
            # GPU not ready (e.g., exclusive mode with tasks running)
            # Put task back in queue
            self.task_queue.global_queue.appendleft(task)
            logger.debug(f"Task {task.task_id[:8]} waiting for GPU")
            break  # Try again in next round (1 second later)
        
        # Assign task to GPU
        self.task_queue.assign_task_to_gpu(task.task_id, available_gpu)
        self.gpu_manager.add_task(available_gpu, task.task_id)
        
        # Run task in background thread
        threading.Thread(
            target=self.task_runner.run_task,
            args=(task, available_gpu),
            daemon=True
        ).start()
```

#### Scenario 2: Exclusive → Shared (While Task Running)

```
Timeline:
─────────────────────────────────────────────────────────────
10:00  │ GPU 0: mode=EXCLUSIVE, running_tasks=[T1]
       │
10:01  │ Admin calls: set_gpu_mode(0, "shared", max_tasks=3)
       │ ✓ Mode changed immediately
       │ ✓ T1 continues running
       │
10:02  │ User submits: Task T2
       │ Scheduler checks: can_accept_task()
       │   → mode == SHARED
       │   → len(running_tasks) == 1 < 3 ✓
       │   → memory_usage < 0.75 ✓
       │   → return True ✓
       │ ✓ T2 starts immediately (runs alongside T1)
       │
10:03  │ User submits: Task T3
       │ running_tasks=[T1, T2]
       │ can_accept_task() → True ✓
       │ ✓ T3 starts (runs alongside T1, T2)
       │
10:04  │ User submits: Task T4
       │ running_tasks=[T1, T2, T3]
       │ can_accept_task()
       │   → len(running_tasks) == 3 == max_tasks
       │   → return False
       │ ✓ T4 goes to pending
       │
10:05  │ T1 completes
       │ running_tasks=[T2, T3]
       │ can_accept_task() → True ✓
       │ ✓ T4 starts
─────────────────────────────────────────────────────────────
```

#### Scenario 3: Exclusive Task Submitted to Shared GPU

**Important: Task type does NOT determine GPU behavior!**

```
Timeline:
─────────────────────────────────────────────────────────────
10:00  │ GPU 0: mode=SHARED, running_tasks=[]
       │
10:01  │ User submits: Task T1 (type=performance)
       │ Note: Task type is just a label!
       │ Scheduler checks: can_accept_task()
       │   → mode == SHARED
       │   → len(running_tasks) == 0 < 3 ✓
       │   → return True ✓
       │ ✓ T1 starts
       │
10:02  │ User submits: Task T2 (type=functional)
       │ Scheduler checks: can_accept_task()
       │   → mode == SHARED (GPU is still in shared mode!)
       │   → len(running_tasks) == 1 < 3 ✓
       │   → return True ✓
       │ ✓ T2 starts (runs alongside T1)
       │
       │ Both tasks run concurrently because GPU mode = SHARED
─────────────────────────────────────────────────────────────
```

**If user wants exclusive execution, they must:**
```python
# Step 1: Set GPU to exclusive mode
client.set_gpu_mode(gpu_id=0, mode="exclusive")

# Step 2: Submit task
task_id = client.submit_task(
    script_path="benchmark.py",
    task_type="performance",  # This is just metadata
    gpu_id=0
)
```

---

## Task Type System

### Definition

```python
# models.py Line 13-15
class TaskType(str, Enum):
    FUNCTIONAL = "functional"    # For correctness testing
    PERFORMANCE = "performance"  # For benchmarking
```

### Purpose

**Task type is METADATA ONLY. It does NOT affect scheduling.**

```
┌────────────────────────────────────────────────────────┐
│ Task Type Usage:                                       │
│                                                         │
│ ✓ Categorization in logs/statistics                   │
│ ✓ UI/dashboard display                                │
│ ✓ Filtering in task queries                           │
│ ✓ User documentation/organization                     │
│                                                         │
│ ✗ NOT used in scheduling decisions                    │
│ ✗ NOT used in resource allocation                     │
│ ✗ NOT used in GPU mode determination                  │
└────────────────────────────────────────────────────────┘
```

### Task Structure

```python
# models.py Line 48-78
class Task:
    """Task model."""
    task_id: str                    # Unique identifier
    task_type: TaskType             # functional/performance (metadata)
    script_path: str                # Python script to run
    args: List[str]                 # Command line arguments
    env: Dict[str, str]             # Environment variables
    gpu_id: Optional[int]           # Preferred GPU (None = any GPU)
    work_dir: Optional[str]         # Working directory
    
    status: TaskStatus              # pending/running/completed/failed/cancelled
    
    submit_time: float              # When task was submitted
    start_time: Optional[float]     # When task started running
    end_time: Optional[float]       # When task finished
    
    exit_code: Optional[int]        # Process exit code
    log_file: Optional[str]         # Task log file path
    stdout_size: int                # Size of stdout
    stderr_size: int                # Size of stderr
```

**Key Point:**
```python
# Scheduler does NOT use task.task_type for any decisions
# It only uses:
# - task.gpu_id (preferred GPU)
# - GPU's mode (exclusive/shared)
# - GPU's resource constraints (max_tasks, memory)
```

---

## Scheduler Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NVGPU Scheduler                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Every 1 second
                          ▼
         ┌────────────────────────────────┐
         │  1. Check Global Pending Queue │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  2. For Each Pending Task:     │
         │     - Get preferred GPU        │
         │     - Find available GPU       │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  3. Check GPU.can_accept_task()│
         │     (uses current GPU mode)    │
         └────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         ┌─────────┐           ┌──────────┐
         │   YES   │           │    NO    │
         └─────────┘           └──────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Assign to GPU    │    │ Put back in queue│
    │ Start execution  │    │ Wait next round  │
    └──────────────────┘    └──────────────────┘
```

### Complete Scheduling Logic

```python
# scheduler.py Line 17-56
class Scheduler:
    """Task scheduler - runs in background thread."""
    
    def __init__(self, task_queue, gpu_manager, task_runner):
        self.task_queue = task_queue
        self.gpu_manager = gpu_manager
        self.task_runner = task_runner
        self.running = False
        self.thread = None
    
    def start(self):
        """Start scheduler thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Scheduler started")
    
    def _run(self):
        """Main scheduler loop - runs every 1 second."""
        while self.running:
            try:
                self._schedule_round()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            time.sleep(1)  # Check every 1 second
    
    def _schedule_round(self):
        """Run one round of task scheduling.
        
        Process:
        1. Pop task from global pending queue
        2. Find available GPU (respects current mode)
        3. If GPU available: assign and run
        4. If GPU not available: put back, wait next round
        """
        while True:
            # Step 1: Get next pending task
            task = self.task_queue.pop_pending_task()
            if task is None:
                break  # No more pending tasks
            
            # Step 2: Determine GPU preference
            gpu_id = task.gpu_id if task.gpu_id is not None else None
            
            # Step 3: Find available GPU
            # This function checks GPU mode and constraints
            available_gpu = self.gpu_manager.find_available_gpu(gpu_id)
            
            if available_gpu is None:
                # No GPU available right now
                # Put task back in queue for next round
                self.task_queue.global_queue.appendleft(task)
                logger.debug(f"Task {task.task_id[:8]} waiting for GPU")
                break  # Stop this round, try again in 1 second
            
            # Step 4: Assign task to GPU
            self.task_queue.assign_task_to_gpu(task.task_id, available_gpu)
            self.gpu_manager.add_task(available_gpu, task.task_id)
            
            # Step 5: Run task in background thread
            threading.Thread(
                target=self.task_runner.run_task,
                args=(task, available_gpu),
                daemon=True
            ).start()
            
            logger.info(f"Task {task.task_id[:8]} assigned to GPU {available_gpu}")
```

### GPU Selection Logic

```python
# gpu_manager.py Line 209-246
def find_available_gpu(self, preferred_gpu: Optional[int] = None) -> Optional[int]:
    """Find an available GPU for task assignment.
    
    Process:
    1. Update GPU memory usage (real-time)
    2. Check if in severe error state (pause all scheduling)
    3. If preferred GPU specified: check if it can accept task
    4. Otherwise: find any GPU that can accept task
    
    Returns:
        GPU ID if available, None otherwise
    """
    # Check severe error state
    if self.severe_error_active:
        return None  # All scheduling paused
    
    # Try preferred GPU first
    if preferred_gpu is not None and preferred_gpu in self.gpus:
        # Update memory before checking
        self._update_gpu_memory(preferred_gpu)
        
        with self.lock:
            gpu = self.gpus.get(preferred_gpu)
            if gpu and gpu.can_accept_task():  # ← Uses current mode!
                return preferred_gpu
        return None  # Preferred GPU not available
    
    # Find any available GPU
    # First update all GPU memory usages
    with self.lock:
        gpu_ids = list(self.gpus.keys())
    
    for gpu_id in gpu_ids:
        self._update_gpu_memory(gpu_id)
    
    # Now check which GPUs can accept tasks
    with self.lock:
        for gpu_id, gpu in self.gpus.items():
            if gpu.can_accept_task():  # ← Uses current mode!
                return gpu_id
    
    return None  # No GPU available
```

### Task Execution

```python
# task_runner.py Line 38-154 (simplified)
class TaskRunner:
    """Execute tasks as subprocesses."""
    
    def run_task(self, task: Task, gpu_id: int) -> bool:
        """Run a task on specified GPU.
        
        Steps:
        1. Update task status to running
        2. Setup CUDA_VISIBLE_DEVICES environment
        3. Setup stdout/stderr capture files
        4. Launch Python subprocess
        5. Monitor execution (with timeout)
        6. Capture output and exit code
        7. Update task status
        8. Remove task from GPU
        9. Check for GPU errors
        """
        try:
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            # Prepare command
            cmd = [sys.executable, task.script_path] + task.args
            
            # Setup environment (critical for GPU isolation)
            env = os.environ.copy()
            env.update(task.env or {})
            
            # Set CUDA_VISIBLE_DEVICES to isolate GPU
            cuda_id = self._get_cuda_id(gpu_id)
            env['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
            
            # Setup log files
            log_dir = NVGPU_ROOT / "logs" / "tasks"
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"{task.task_id}.stdout"
            stderr_path = log_dir / f"{task.task_id}.stderr"
            
            # Run subprocess
            with open(stdout_path, 'w') as stdout_f, \
                 open(stderr_path, 'w') as stderr_f:
                
                process = subprocess.Popen(
                    cmd,
                    cwd=task.work_dir or os.getcwd(),
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True
                )
                
                # Wait for completion (with timeout)
                try:
                    exit_code = process.wait(timeout=config.max_task_duration)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Task {task.task_id[:8]} timed out")
                    process.kill()
                    exit_code = -1
            
            # Update task status
            task.end_time = time.time()
            task.exit_code = exit_code
            
            if exit_code == 0:
                task.status = TaskStatus.COMPLETED
                logger.info(f"Task {task.task_id[:8]} completed")
            else:
                task.status = TaskStatus.FAILED
                logger.warning(f"Task {task.task_id[:8]} failed")
            
            return exit_code == 0
            
        finally:
            # Always remove task from GPU (crucial for correct scheduling)
            self.gpu_manager.remove_task(gpu_id, task.task_id)
```

---

## Complete Workflow

### Example: Submitting Tasks to Shared GPU

```python
# Client code
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# Setup: Configure GPU 0 as shared with max 3 concurrent tasks
client.set_gpu_mode(gpu_id=0, mode="shared")
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=3)

# Submit 5 tasks
task_ids = []
for i in range(5):
    task_id = client.submit_task(
        script_path=f"test_{i}.py",
        task_type="functional",  # Just metadata
        gpu_id=0
    )
    task_ids.append(task_id)
```

**Server-side execution:**

```
Time    Event                          GPU State                Scheduler Action
─────────────────────────────────────────────────────────────────────────────────
10:00   5 tasks submitted              running=[], pending=[T0,T1,T2,T3,T4]
        
10:01   Scheduler round 1              
        - Pop T0                       can_accept_task()        Assign T0 → GPU 0
          check: 0 < 3 ✓               running=[T0], pending=[T1,T2,T3,T4]
          
        - Pop T1                       can_accept_task()        Assign T1 → GPU 0
          check: 1 < 3 ✓               running=[T0,T1], pending=[T2,T3,T4]
          
        - Pop T2                       can_accept_task()        Assign T2 → GPU 0
          check: 2 < 3 ✓               running=[T0,T1,T2], pending=[T3,T4]
          
        - Pop T3                       can_accept_task()        Put T3 back
          check: 3 < 3 ✗ BLOCKED       running=[T0,T1,T2], pending=[T3,T4]
          
        Round ends (GPU at capacity)

10:02   Scheduler round 2              
        - Pop T3                       can_accept_task()        T3 still blocked
          check: 3 < 3 ✗               running=[T0,T1,T2], pending=[T3,T4]
          
        Round ends

10:05   T0 completes                   running=[T1,T2], pending=[T3,T4]
        GPU removes T0

10:06   Scheduler round 3              
        - Pop T3                       can_accept_task()        Assign T3 → GPU 0
          check: 2 < 3 ✓               running=[T1,T2,T3], pending=[T4]

10:07   T1 completes                   running=[T2,T3], pending=[T4]

10:08   Scheduler round 4              
        - Pop T4                       can_accept_task()        Assign T4 → GPU 0
          check: 2 < 3 ✓               running=[T2,T3,T4], pending=[]

10:10   All tasks complete             running=[], pending=[]
```

### Example: Mode Switch During Execution

```python
# Setup: GPU in shared mode with tasks running
client.set_gpu_mode(gpu_id=0, mode="shared")

# Submit 3 tasks that run for 60 seconds each
for i in range(3):
    client.submit_task("long_task.py", args=["--duration", "60"], gpu_id=0)

# Wait for tasks to start
time.sleep(5)

# NOW: Switch to exclusive mode
client.set_gpu_mode(gpu_id=0, mode="exclusive")

# Submit another task
task_id = client.submit_task("quick_task.py", gpu_id=0)
```

**Server-side execution:**

```
Time    Event                          GPU State                Scheduler Behavior
────────────────────────────────────────────────────────────────────────────────────
10:00   3 tasks submitted              mode=SHARED
                                       running=[], pending=[T0,T1,T2]

10:01   Scheduler assigns all 3        running=[T0,T1,T2], pending=[]
        (shared mode allows it)

10:05   Admin: set_gpu_mode(0, "exclusive")
        ✓ Mode changed                 mode=EXCLUSIVE (changed!)
        ✓ T0, T1, T2 keep running      running=[T0,T1,T2]  (unchanged!)

10:06   T3 submitted                   running=[T0,T1,T2], pending=[T3]

10:07   Scheduler round:
        - Pop T3
        - can_accept_task()?
          → mode == EXCLUSIVE
          → len(running) == 3 > 0
          → return False ✗
        - Put T3 back                  T3 remains pending

10:08   [Rounds continue, T3 still pending]

10:30   T0 completes                   running=[T1,T2], pending=[T3]

10:31   Scheduler round:
        - Pop T3
        - can_accept_task()?
          → len(running) == 2 > 0
          → return False ✗
        - Put T3 back                  T3 still pending

10:45   T1 completes                   running=[T2], pending=[T3]

10:46   Scheduler round:
        - Pop T3
        - can_accept_task()?
          → len(running) == 1 > 0
          → return False ✗              T3 still pending

11:00   T2 completes                   running=[], pending=[T3]

11:01   Scheduler round:
        - Pop T3
        - can_accept_task()?
          → mode == EXCLUSIVE
          → len(running) == 0
          → return True ✓
        - Assign T3 → GPU 0            running=[T3], pending=[]
        
11:05   T3 completes                   running=[], pending=[]
```

---

## Design Rationale

### Why Separate GPU Mode from Task Type?

**Flexibility and Control:**

```
┌─────────────────────────────────────────────────────────────┐
│ Scenario 1: Performance Testing                             │
│                                                               │
│ Need: Run benchmark in isolation                            │
│ Solution:                                                    │
│   1. Admin: set_gpu_mode(0, "exclusive")                    │
│   2. User: submit_task(..., type="performance")             │
│                                                               │
│ Result: Task gets exclusive GPU access                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Scenario 2: Batch Functional Testing                        │
│                                                               │
│ Need: Run 100 functional tests quickly                      │
│ Solution:                                                    │
│   1. Admin: set_gpu_mode(0, "shared", max_tasks=10)         │
│   2. User: submit 100 tasks (type="functional")             │
│                                                               │
│ Result: Up to 10 tests run concurrently                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Scenario 3: Mixed Workload                                  │
│                                                               │
│ Need: Allow both test types to share GPU                    │
│ Solution:                                                    │
│   1. Admin: set_gpu_mode(0, "shared")                       │
│   2. Users submit various task types                        │
│                                                               │
│ Result: All tasks treated equally, scheduled by FIFO        │
└─────────────────────────────────────────────────────────────┘
```

### Why Manual Mode Switching?

**Predictability and Safety:**

1. **No surprises:** Admins control GPU behavior explicitly
2. **No interruptions:** Running tasks never affected by mode changes
3. **Clear states:** GPU mode is always known and deterministic
4. **Safety:** No automatic decisions that might disrupt workflows

**Alternative (rejected) designs:**

```python
# ❌ BAD: Automatic mode switching based on task type
if task.task_type == "performance":
    gpu.mode = "exclusive"  # Dangerous! What about other running tasks?

# ❌ BAD: Task type overrides GPU mode
if task.task_type == "performance":
    ignore_gpu_mode = True  # Confusing! Mode becomes meaningless
```

### Why Running Tasks Unaffected by Mode Changes?

**Graceful Degradation:**

```
Philosophy: "Don't break what's working"

When mode switches:
- New tasks follow new rules
- Old tasks finish naturally
- No abrupt terminations
- No resource conflicts

This ensures:
✓ Data consistency (tasks complete normally)
✓ No wasted computation
✓ Predictable behavior
✓ User trust
```

### Why FIFO Scheduling?

**Simplicity and Fairness:**

- Easy to understand and predict
- Fair: first submitted, first executed
- No starvation: every task eventually runs
- Sufficient for most GPU testing workloads

**Future enhancements could add:**
- Priority queues (for urgent tasks)
- Preemption (for high-priority tasks)
- Resource-aware scheduling (memory-based)

---

## Code Reference

### Key Files and Functions

```
server/nvgpu/
├── models.py
│   ├── GPUMode (Line 8-10)           # exclusive/shared definition
│   ├── TaskType (Line 13-15)         # functional/performance definition
│   ├── GPU.can_accept_task() (23-36) # Core scheduling logic
│   └── Task (48-78)                  # Task data structure
│
├── gpu_manager.py
│   ├── set_gpu_mode() (100-108)      # Mode switching
│   └── find_available_gpu() (209-246)# GPU selection
│
├── scheduler.py
│   ├── _run() (28-33)                # Main loop (1 second interval)
│   └── _schedule_round() (35-56)     # Task assignment logic
│
├── task_runner.py
│   └── run_task() (38-154)           # Task execution
│
├── task_queue.py
│   ├── submit_task() (23-48)         # Task submission
│   ├── pop_pending_task() (50-56)    # Dequeue for scheduling
│   └── assign_task_to_gpu() (58-73)  # Move to GPU queue
│
└── api_server.py
    ├── POST /tasks (90-120)          # Submit task endpoint
    ├── PUT /gpus/{id}/mode (221-234) # Change mode endpoint
    └── PUT /gpus/{id}/max_tasks (236-249) # Set concurrency limit
```

### Critical Logic Summary

```python
# The three pillars of scheduling:

# 1. GPU Mode Check (models.py)
def can_accept_task(self) -> bool:
    if self.mode == GPUMode.EXCLUSIVE:
        return len(self.running_tasks) == 0
    # ... shared mode checks

# 2. GPU Selection (gpu_manager.py)
def find_available_gpu(self, preferred_gpu=None) -> Optional[int]:
    for gpu_id, gpu in self.gpus.items():
        if gpu.can_accept_task():  # ← Uses mode
            return gpu_id
    return None

# 3. Scheduling Loop (scheduler.py)
def _schedule_round(self):
    task = self.task_queue.pop_pending_task()
    available_gpu = self.gpu_manager.find_available_gpu(task.gpu_id)
    if available_gpu:
        # Assign and run
    else:
        # Put back in queue, wait next round
```

---

## Summary

### Key Takeaways

1. **GPU Mode is Manual**
   - Set by administrator via API
   - Does NOT change automatically
   - Does NOT depend on task type

2. **Task Type is Metadata**
   - Labels task purpose
   - Does NOT affect scheduling
   - Does NOT determine GPU mode

3. **Mode Switching is Safe**
   - Running tasks continue unaffected
   - New tasks follow new mode rules
   - No interruptions or conflicts

4. **Scheduling is Simple**
   - FIFO order
   - Respects current GPU mode
   - Checks every 1 second
   - Waits in queue if GPU unavailable

5. **Design Philosophy**
   - Separation of concerns (mode vs task type)
   - Explicit control (manual mode switching)
   - Graceful degradation (no task interruption)
   - Predictable behavior (clear rules)

### Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│ I want to...                     | What to do               │
├─────────────────────────────────────────────────────────────┤
│ Run task in isolation            | Set GPU to exclusive     │
│ Run multiple tasks concurrently  | Set GPU to shared        │
│ Change GPU behavior              | Call set_gpu_mode API    │
│ Label task purpose               | Set task_type param      │
│ Prevent overloading GPU          | Set max_concurrent_tasks │
│ Control memory usage             | Set memory_threshold     │
│ Submit task to specific GPU      | Set gpu_id param         │
│ Wait for GPU to be free          | Just submit; auto-queued │
└─────────────────────────────────────────────────────────────┘
```

---

## Document Version

- **Version:** 1.0
- **Date:** 2025-10-08
- **Author:** NVGPU Server Team
- **Status:** Complete

