#!/bin/bash
# Example API test script for NVGPU server

BASE_URL="http://localhost:8080"

echo "=== NVGPU Server API Test ==="
echo ""

# Health check
echo "1. Health check"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# List GPUs
echo "2. List GPUs"
curl -s "$BASE_URL/gpus" | python3 -m json.tool
echo ""

# Submit a functional task
echo "3. Submit functional task"
TASK_RESPONSE=$(curl -s -X POST "$BASE_URL/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "test_scripts/simple_functional_test.py",
    "work_dir": ".",
    "args": []
  }')
echo "$TASK_RESPONSE" | python3 -m json.tool
TASK_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
echo "Task ID: $TASK_ID"
echo ""

# Wait a bit
echo "4. Waiting 2 seconds..."
sleep 2
echo ""

# Check task status
echo "5. Check task status"
curl -s "$BASE_URL/tasks/$TASK_ID" | python3 -m json.tool
echo ""

# Get statistics
echo "6. Get statistics"
curl -s "$BASE_URL/stats" | python3 -m json.tool
echo ""

# List all tasks
echo "7. List all tasks"
curl -s "$BASE_URL/tasks" | python3 -m json.tool
echo ""

echo "=== Test complete ==="

