from server.nvgpu.client import NVGPUClient

client = NVGPUClient(base_url="http://localhost:8080")

tasks = client.list_tasks()  # get all tasks

target_status = {"running", "pending", "queued"}

for t in tasks:
    if t.get("task_type") == "performance" and t.get("status") in target_status:
        task_id = t["task_id"]
        print(f"Force cancelling performance task: {task_id}")
        client.cancel_task(task_id, force=True)