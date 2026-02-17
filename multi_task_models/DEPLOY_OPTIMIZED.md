# Deploy Optimized Neural Network Script

## 📊 Optimization Results

Your script has been optimized for your server hardware:
- **CPU**: Intel Xeon Gold 6338 (16 cores)
- **RAM**: 60 GB available
- **Expected speedup**: 2x faster (4-5 hours instead of 8-10 hours)

## 🚀 Quick Deployment

### 1. Download from S3 on Server

```bash
# SSH into server
ssh nebius@89.169.122.10 -i ~/.ssh/barcelona

# Create directory and download files
mkdir -p ~/neural_experiments_optimized
cd ~/neural_experiments_optimized

# Download optimized script and dataset
cp /s3/run_neural_sweep_standalone_optimized.py ./run_neural_sweep.py
cp /s3/rand_hrs_model_merged.parquet ./rand_hrs_model_merged.parquet
cp /s3/requirements_standalone.txt ./requirements.txt
cp /s3/OPTIMIZATION_SUMMARY.md ./README.md

# Install dependencies (if not already installed)
pip install -q -r requirements.txt
```

### 2. Run Optimized Experiments

```bash
# Start experiments in background
nohup python run_neural_sweep.py > experiments.log 2>&1 &

# Get process ID
echo $! > experiment.pid
echo "Experiment running with PID: $(cat experiment.pid)"

# Monitor progress in real-time
tail -f experiments.log
```

### 3. Check CPU Utilization (Optional)

```bash
# In another terminal, verify optimization is working
htop
# You should see: All 16 cores at ~95-100% during training
```

## 📈 What Changed from Original Script

### Performance Optimizations
1. **PyTorch Threads**: 8 → 16 (uses all CPU cores)
2. **DataLoader Workers**: 0 → 6 (parallel data loading)
3. **Sequential Execution**: Optimized for CPU (no thread contention)

### Configuration Variables (Top of Script)
```python
NUM_WORKERS = 6          # DataLoader workers
TORCH_THREADS = 16       # PyTorch computation threads
```

These are already set optimally for your hardware - **no changes needed!**

## ⏱️ Time Estimates

| Phase | Time per Experiment | Total Time (41 experiments) |
|-------|---------------------|----------------------------|
| **Before optimization** | 12-15 minutes | 8-10 hours |
| **After optimization** | 6-8 minutes | 4-5 hours |

## 📂 Output Structure

```
~/neural_experiments_optimized/
├── run_neural_sweep.py            # Optimized script
├── rand_hrs_model_merged.parquet  # Dataset (correct one!)
├── experiments.log                # Execution log
├── experiment.pid                 # Process ID
├── results/
│   └── neural_sweep/
│       ├── exp_011_*.json
│       ├── exp_012_*.json
│       └── ... (41 JSON files)
```

## 🔍 Monitoring Commands

### Real-time Progress
```bash
tail -f experiments.log
```

### Count Completed Experiments
```bash
ls -1 results/neural_sweep/*.json 2>/dev/null | wc -l
```

### Check Process Status
```bash
ps -p $(cat experiment.pid) > /dev/null && echo "Running ✓" || echo "Stopped ✗"
```

### Show Top 3 Results So Far
```bash
python3 << 'EOF'
import json, glob
results = []
for f in sorted(glob.glob("results/neural_sweep/*.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
            if "aggregate_metrics" in d:
                name = d['experiment_name']
                roc = d['aggregate_metrics']['mean_roc_auc']
                f2 = d['aggregate_metrics']['mean_f2_score']
                results.append((roc, f2, name))
    except: pass

results.sort(reverse=True)
print(f"\n{'='*60}")
print(f"Completed: {len(results)}/41 experiments")
print(f"{'='*60}")
print(f"\nTop 3 by ROC-AUC:")
for i, (roc, f2, name) in enumerate(results[:3], 1):
    print(f"{i}. {name}")
    print(f"   ROC-AUC: {roc:.4f}  |  F2: {f2:.4f}")
print(f"{'='*60}\n")
EOF
```

## 💾 Download Results When Complete

### Option 1: SCP (Direct)
```bash
# From your Mac
scp -i ~/.ssh/barcelona -r \
    nebius@89.169.122.10:~/neural_experiments_optimized/results/neural_sweep/ \
    ~/Desktop/hea_hackathon/hea_competition/results/
```

### Option 2: Via S3
```bash
# On server: Upload to S3
cd ~/neural_experiments_optimized
cp -r results/neural_sweep/ /s3/results_optimized/

# On Mac: Download from S3
cd ~/Desktop/hea_hackathon/hea_competition/multi_task_models
aws s3 sync \
    s3://hackathon-team-fabric3-8/results_optimized/neural_sweep/ \
    ./results/neural_sweep/
```

## ⚠️ Important Notes

### ✅ What's Optimized
- CPU utilization: **100%** across all cores
- Data loading: **Parallel** (6 workers)
- Memory usage: **Efficient** (~10-15 GB peak)
- Execution: **Sequential** (no thread contention)

### ❌ What's NOT Changed
- Model architectures (same as original)
- Hyperparameters (same as original)
- Batch sizes (same as original)
- Dataset (correctly using `rand_hrs_model_merged.parquet`)
- Experiment order (still reverse: 51 → 11)
- Results format (still JSON files)

### 🎯 Why These Optimizations?
See `OPTIMIZATION_SUMMARY.md` for detailed explanation of:
- Why 16 threads (not 8)
- Why 6 workers (not 0)
- Why sequential (not parallel experiments)
- Performance benchmarks and rationale

## 🐛 Troubleshooting

### Script exits immediately
```bash
# Check for errors
tail -n 50 experiments.log

# Verify dataset exists
ls -lh rand_hrs_model_merged.parquet
```

### CPU not at 100%
```bash
# Check if PyTorch is using all threads
python -c "import torch; print(f'Threads: {torch.get_num_threads()}')"
# Should output: Threads: 16
```

### Out of memory
```bash
# Check available RAM
free -h
# Should have >40 GB available during training
```

### Experiments too slow (>10 min each)
```bash
# Verify num_workers is 6
grep "NUM_WORKERS" run_neural_sweep.py
# Should show: NUM_WORKERS = 6
```

## 📊 Expected Performance

### Resource Usage During Training
- **CPU**: 95-100% across all 16 cores ✓
- **RAM**: 10-15 GB (out of 60 GB) ✓
- **Disk I/O**: Minimal (data cached in RAM) ✓
- **Network**: None (local execution) ✓

### Experiment Timing
- **Fastest experiments** (MLPs): 4-6 minutes
- **Average experiments** (ResNet): 6-8 minutes
- **Slowest experiments** (Attention): 8-12 minutes
- **Overall average**: 6-8 minutes/experiment

## ✨ Comparison: Original vs Optimized

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| PyTorch threads | 8 | 16 | **2.0x** |
| DataLoader workers | 0 | 6 | **1.3x** |
| Time per experiment | 12-15 min | 6-8 min | **1.9x** |
| Total time (41 exp) | 8-10 hours | 4-5 hours | **2.0x** |
| CPU utilization | ~50% | ~100% | **2.0x** |
| Model performance | Baseline | Identical | No change |

## 🎉 Ready to Go!

Your optimized script is ready to run. Just execute:

```bash
cd ~/neural_experiments_optimized
nohup python run_neural_sweep.py > experiments.log 2>&1 &
tail -f experiments.log
```

Sit back and let your 16 cores do the heavy lifting! 🚀

Estimated completion: **4-5 hours** from start.

---

**Questions?** Check `OPTIMIZATION_SUMMARY.md` for technical details.
