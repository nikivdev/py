#!/bin/bash

# CPU Cleanup Script
# Identifies and optionally kills high-CPU processes

THRESHOLD=30  # CPU percentage threshold

echo "=== High CPU Processes (>${THRESHOLD}%) ==="
echo ""

# Get high CPU processes (excluding kernel_task and Activity Monitor)
ps -Ao pid,pcpu,comm -r | awk -v thresh="$THRESHOLD" '
NR==1 {print; next}
$2 > thresh && $3 !~ /kernel_task/ && $3 !~ /Activity Monitor/ && $3 !~ /WindowServer/ {
    print
}' | head -20

echo ""
echo "=== Safe to kill ==="
echo "1) workerd     - Cloudflare Workers (can restart)"
echo "2) debugserver - Debug processes (safe to kill)"
echo "3) find        - Runaway file searches"
echo ""
echo "=== DO NOT kill ==="
echo "- kernel_task (system thermal management)"
echo "- WindowServer (will log you out)"
echo ""

read -p "Kill all workerd processes? (y/n): " kill_workerd
if [[ "$kill_workerd" == "y" ]]; then
    pkill -9 workerd && echo "Killed workerd" || echo "No workerd found"
fi

read -p "Kill all debugserver processes? (y/n): " kill_debug
if [[ "$kill_debug" == "y" ]]; then
    pkill -9 debugserver && echo "Killed debugserver" || echo "No debugserver found"
fi

read -p "Kill all find processes? (y/n): " kill_find
if [[ "$kill_find" == "y" ]]; then
    pkill -9 find && echo "Killed find" || echo "No find found"
fi

echo ""
echo "=== Current top CPU after cleanup ==="
ps -Ao pid,pcpu,comm -r | head -10
