import json
import os
import numpy as np
from collections import defaultdict
import argparse

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_intervals(data, step):
    # Group samples by interval
    intervals = defaultdict(list)
    
    for sample in data:
        sparsity = sample['sparsity']
        # Use floor division to determine the bucket index
        # For sparsity 1.0, we might want to handle it carefully, but standard floor is fine.
        # If step is 0.001, 1.0 -> 1000. Interval 1.000-1.001.
        interval_idx = int(sparsity / step)
        intervals[interval_idx].append(sample)
        
    sorted_indices = sorted(intervals.keys())
    
    print(f"\nAnalysis with interval step: {step}")
    print("="*130)
    print(f"{'Interval':<25} {'Samples':<10} {'Best Method':<15} {'Avg Score':<12} {'Min Score':<12} {'Max Score':<12} {'Avg Ratio':<12}")
    print("-"*130)
    
    for idx in sorted_indices:
        samples = intervals[idx]
        low = idx * step
        high = (idx + 1) * step
        
        # Analyze best method for this interval
        method_scores = defaultdict(list)
        method_ratios = defaultdict(list)
        
        for sample in samples:
            for res in sample['results']:
                method_scores[res['method']].append(res['score'])
                method_ratios[res['method']].append(res['compression_ratio'])
        
        best_method = None
        best_avg_score = -1.0
        best_avg_ratio = 0.0
        best_min_score = 0.0
        best_max_score = 0.0
        
        for method, scores in method_scores.items():
            avg_score = np.mean(scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_method = method
                best_avg_ratio = np.mean(method_ratios[method])
                best_min_score = np.min(scores)
                best_max_score = np.max(scores)
                
        print(f"{f'{low:.5f}-{high:.5f}':<25} {len(samples):<10} {best_method:<15} {best_avg_score:<12.6f} {best_min_score:<12.6f} {best_max_score:<12.6f} {best_avg_ratio:<12.6f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze sparsity benchmark results.')
    parser.add_argument('--step', type=float, default=0.001, help='Interval step size (default: 0.001)')
    parser.add_argument('--file', type=str, default='sparsity_benchmark_results.json', help='Path to results JSON file')
    
    args = parser.parse_args()
    
    # Resolve file path
    file_path = args.file
    if not os.path.exists(file_path):
        # Try looking in the same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, args.file)
        if os.path.exists(potential_path):
            file_path = potential_path
        else:
             # Try looking in submissions/benchmarks relative to cwd
             potential_path = os.path.join(os.getcwd(), 'submissions', 'benchmarks', args.file)
             if os.path.exists(potential_path):
                 file_path = potential_path

    if not os.path.exists(file_path):
        print(f"Error: Could not find file {args.file}")
        return

    print(f"Loading results from: {file_path}")
    data = load_results(file_path)
    
    analyze_intervals(data, args.step)

if __name__ == "__main__":
    main()
