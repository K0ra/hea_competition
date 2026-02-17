#!/bin/bash
# Check remote server hardware specifications

echo "============================================"
echo "SERVER HARDWARE SPECIFICATIONS"
echo "============================================"

echo ""
echo "=== CPU Information ==="
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket"
echo ""
echo "CPU Cores (logical): $(nproc)"

echo ""
echo "=== Memory Information ==="
free -h

echo ""
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap --format=csv
    echo ""
    echo "Number of GPUs: $(nvidia-smi --list-gpus | wc -l)"
else
    echo "No NVIDIA GPU detected (CPU-only machine)"
fi

echo ""
echo "=== Disk Space ==="
df -h /home

echo ""
echo "=== System Architecture ==="
uname -m

echo ""
echo "=== PyTorch Detection ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('Running on CPU')
    print(f'CPU threads available: {torch.get_num_threads()}')
" 2>/dev/null || echo "PyTorch not installed yet"

echo ""
echo "============================================"
echo "OPTIMIZATION RECOMMENDATIONS"
echo "============================================"
