import re
import numpy as np
from collections import defaultdict

log_file = 'experiments/final_benchmark_100.log'

results = []

with open(log_file, 'r') as f:
    for line in f:
        # Match lines like: [ 12/98] ECG ID   116 (STTC ) | NORM Prob: 0.4380 -> 0.8845 | HR Err: 33.5 bpm | L1: 0.0474
        match = re.search(r'ECG ID\s+\d+\s*\(([^)]+)\)\s*\|\s*NORM Prob:\s*([\d.]+)\s*->\s*([\d.]+)\s*\|\s*HR Err:\s*([\d.]+)\s*bpm\s*\|\s*L1:\s*([\d.]+)', line)
        if match:
            pathology = match.group(1).strip()
            init_prob = float(match.group(2))
            final_prob = float(match.group(3))
            hr_err = float(match.group(4))
            l1 = float(match.group(5))
            
            flip_success = 1 if final_prob >= 0.80 else 0
            
            results.append({
                'pathology': pathology,
                'init_prob': init_prob,
                'final_prob': final_prob,
                'hr_err': hr_err,
                'l1': l1,
                'flip_success': flip_success
            })

print(f"Parsed {len(results)} patients from the log.")

class_stats = defaultdict(list)
for r in results:
    class_stats[r['pathology']].append(r)

print("\n" + "="*50)
print("FINAL BENCHMARK AGGREGATED METRICS")
print("="*50)

overall_hr = []
overall_l1 = []
overall_flip = []

for path, group in class_stats.items():
    hr = [r['hr_err'] for r in group]
    l1 = [r['l1'] for r in group]
    flips = [r['flip_success'] for r in group]
    
    print(f"Pathology: {path} ({len(group)} patients)")
    print(f"  Flip Rate : {np.mean(flips)*100:.1f}%")
    print(f"  Avg L1    : {np.mean(l1):.4f}")
    print(f"  Avg HR Err: {np.mean(hr):.1f} bpm")
    print("-" * 30)
    
    overall_hr.extend(hr)
    overall_l1.extend(l1)
    overall_flip.extend(flips)

print(f"\nOVERALL METRICS ({len(results)} patients)")
print(f"  Overall Flip Rate : {np.mean(overall_flip)*100:.1f}%")
print(f"  Overall Avg L1    : {np.mean(overall_l1):.4f}")
print(f"  Overall Avg HR Err: {np.mean(overall_hr):.1f} bpm")
print("="*50)
