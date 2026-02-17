# Remote Server Execution Instructions

## Files Uploaded to S3

Your S3 bucket (`hackathon-team-fabric3-8`) now contains:
- `fam_aggregated_filtered.pkl` (6.9 MB) - The dataset
- `run_neural_sweep_standalone.py` (52 KB) - The standalone experiment script

## Server Details

- **IP**: 89.169.122.10
- **SSH Key**: `~/.ssh/barcelona`
- **User**: `nebius`
- **S3 Mount**: `/s3` (bucket already mounted on server)

## Step-by-Step Execution Guide

### 1. SSH into the Server

```bash
ssh nebius@89.169.122.10 -i ~/.ssh/barcelona
```

### 2. Navigate to S3 Mount and Copy Files

```bash
# Files should be accessible at /s3
ls /s3

# Copy files to working directory
mkdir -p ~/neural_experiments
cp /s3/fam_aggregated_filtered.pkl ~/neural_experiments/
cp /s3/run_neural_sweep_standalone.py ~/neural_experiments/
cd ~/neural_experiments
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install torch numpy pandas scikit-learn imbalanced-learn

# Or if using conda:
# conda install pytorch numpy pandas scikit-learn -c pytorch
# pip install imbalanced-learn
```

### 4. Update Dataset Path in Script

```bash
# Edit the script to set the correct dataset path
nano run_neural_sweep_standalone.py

# Change line 32 from:
# DATASET_PATH = "/path/to/fam_aggregated_filtered.pkl"
# To:
# DATASET_PATH = "/home/nebius/neural_experiments/fam_aggregated_filtered.pkl"

# Or use sed to automatically update:
sed -i 's|DATASET_PATH = "/path/to/fam_aggregated_filtered.pkl"|DATASET_PATH = "/home/nebius/neural_experiments/fam_aggregated_filtered.pkl"|' run_neural_sweep_standalone.py
```

### 5. Run the Experiments

```bash
# Make executable (if not already)
chmod +x run_neural_sweep_standalone.py

# Run with output logging
python run_neural_sweep_standalone.py 2>&1 | tee neural_experiments.log

# Or run in background with nohup (recommended for long runs):
nohup python run_neural_sweep_standalone.py > neural_experiments.log 2>&1 &

# Get the process ID
echo $!
```

### 6. Monitor Progress

```bash
# Watch the log file in real-time
tail -f neural_experiments.log

# Or use less to navigate
less +F neural_experiments.log

# Check GPU usage (if using CUDA)
nvidia-smi

# Count completed experiments
ls results/neural_sweep/*.json | wc -l
```

### 7. Retrieve Results

After completion, copy results back to S3:

```bash
# Copy results to S3
aws --endpoint-url=https://storage.eu-north1.nebius.cloud:443 \
    s3 cp results/neural_sweep/ s3://hackathon-team-fabric3-8/results/ --recursive

# Or if /s3 is mounted, copy directly
cp -r results/neural_sweep/ /s3/results/
```

## Experiment Details

### What It Will Run

- **Total experiments**: 41 (experiments 11-51)
- **Order**: Reverse (experiment 51 first, experiment 11 last)
- **Estimated time**: 4-6 hours (depends on hardware)
- **Output**: JSON files in `./results/neural_sweep/`

### Experiments Skipped (Already Completed)

1. mlp_shallow_wide_baseline
2. mlp_shallow_narrow_baseline
3. mlp_deep_wide_baseline
4. mlp_deep_medium_baseline
5. mlp_deep_narrow_baseline
6. mlp_very_deep_baseline
7. resnet_2blocks_baseline
8. resnet_3blocks_baseline
9. resnet_4blocks_baseline
10. attention_256_baseline

### Hardware Requirements

- **RAM**: 8GB+ recommended
- **Disk**: 2GB free space for results
- **GPU**: Optional but recommended (CUDA support auto-detected)
- **CPU**: Will use CPU if no GPU available

## Troubleshooting

### Issue: "Module not found" errors

```bash
# Install missing packages
pip install <missing_package>
```

### Issue: Out of memory

Edit the script to reduce batch size:
```python
# Line ~1485: Change batch_size in OPTIMIZATION configs
# From: {"lr": 3e-4, "batch_size": 512, "optimizer": "adam"}
# To: {"lr": 3e-4, "batch_size": 256, "optimizer": "adam"}
```

### Issue: Process killed unexpectedly

Check system resources:
```bash
free -h  # Check RAM
df -h    # Check disk space
```

### Issue: Can't find dataset

Verify paths:
```bash
ls -lh ~/neural_experiments/fam_aggregated_filtered.pkl
```

## Downloading Results Back to Your Mac

From your local machine:

```bash
# Via SCP
scp -i ~/.ssh/barcelona -r \
    nebius@89.169.122.10:~/neural_experiments/results/neural_sweep/ \
    ~/Desktop/hea_hackathon/hea_competition/results/

# Or via S3 (if you copied to S3 first)
aws --endpoint-url=https://storage.eu-north1.nebius.cloud:443 \
    s3 sync s3://hackathon-team-fabric3-8/results/ \
    ~/Desktop/hea_hackathon/hea_competition/results/neural_sweep/
```

## Viewing Results on Server

```bash
# Install pandas on server (if not already)
pip install pandas

# Create a simple results viewer (copy view_results.py to server)
# Or use this one-liner:
python3 << 'EOF'
import json
import glob
results = []
for f in glob.glob("results/neural_sweep/*.json"):
    with open(f) as fp:
        d = json.load(fp)
        if "aggregate_metrics" in d:
            results.append((d['aggregate_metrics']['mean_roc_auc'], f))
results.sort(reverse=True)
print("\nTop 10 Results:")
for i, (roc, fname) in enumerate(results[:10], 1):
    print(f"{i}. {roc:.4f} - {fname}")
EOF
```

## Quick Start (One Command)

```bash
# Execute everything in one go (after SSH)
cd ~ && \
mkdir -p neural_experiments && \
cp /s3/fam_aggregated_filtered.pkl neural_experiments/ && \
cp /s3/run_neural_sweep_standalone.py neural_experiments/ && \
cd neural_experiments && \
sed -i 's|DATASET_PATH = "/path/to/fam_aggregated_filtered.pkl"|DATASET_PATH = "/home/nebius/neural_experiments/fam_aggregated_filtered.pkl"|' run_neural_sweep_standalone.py && \
pip install -q torch numpy pandas scikit-learn imbalanced-learn && \
nohup python run_neural_sweep_standalone.py > neural_experiments.log 2>&1 &
echo "Process started! Monitor with: tail -f ~/neural_experiments/neural_experiments.log"
```

## Expected Output Format

Each experiment creates a JSON file like:
```json
{
  "experiment_name": "wide_deep_large_baseline",
  "epochs_trained": 23,
  "training_time_sec": 645.3,
  "aggregate_metrics": {
    "mean_roc_auc": 0.6892,
    "mean_pr_auc": 0.1723,
    "mean_f2_score": 0.3567
  },
  "per_task_metrics": {
    "DIABE": {"ROC_AUC": 0.6914, "PR_AUC": 0.1668, "F2_score": 0.3583},
    ...
  }
}
```

## Contact & Support

If you encounter issues, check:
1. Log file: `neural_experiments.log`
2. System resources: `htop` or `top`
3. Disk space: `df -h`
4. Process status: `ps aux | grep python`
