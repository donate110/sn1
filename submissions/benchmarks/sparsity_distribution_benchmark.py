import numpy as np
import base64
import bz2
import lzma
import time
import glob
import io
import os
from collections import defaultdict
import json

def load_sample_data(filepath):
    """Load sample activation data from file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    arr_bytes = base64.b64decode(content)
    buffer = io.BytesIO(arr_bytes)
    arr = np.load(buffer, allow_pickle=False)
    return arr

def calculate_sparsity(arr):
    """Calculate sparsity of the array (proportion of zeros)."""
    if arr.size == 0:
        return 0.0
    return 1.0 - (np.count_nonzero(arr) / arr.size)

def benchmark_compression(data_bytes, method_name, compress_func, decompress_func):
    """Benchmark a specific compression method."""
    original_size = len(data_bytes)
    
    # Measure compression time
    start_time = time.time()
    try:
        compressed = compress_func(data_bytes)
    except Exception:
        return None
    compress_time = time.time() - start_time
    
    compressed_size = len(compressed)
    
    # Measure decompression time
    start_time = time.time()
    try:
        decompressed = decompress_func(compressed)
    except Exception:
        return None
    decompress_time = time.time() - start_time
    
    total_time = compress_time + decompress_time
    compression_ratio = compressed_size / original_size
    
    # Calculate score using the formula
    score = max(0.0, min(1.0, (1 - compression_ratio) * (1 - total_time / 1.012)))
    
    return {
        'method': method_name,
        'score': score,
        'compression_ratio': compression_ratio,
        'total_time': total_time
    }

def run_benchmarks_for_file(filepath):
    try:
        arr = load_sample_data(filepath)
    except Exception as e:
        print(f"Error loading {os.path.basename(filepath)}: {e}")
        return None

    sparsity = calculate_sparsity(arr)
    data_bytes = arr.tobytes()
    
    results = []
    
    # Test BZ2 with different compression levels (1-9)
    for level in range(1, 10):
        result = benchmark_compression(
            data_bytes,
            f'bz2-{level}',
            lambda data, lvl=level: bz2.compress(data, compresslevel=lvl),
            bz2.decompress
        )
        if result:
            results.append(result)
    
    # Test LZMA with different presets (0-9)
    for preset in range(10):
        result = benchmark_compression(
            data_bytes,
            f'lzma-{preset}',
            lambda data, p=preset: lzma.compress(data, preset=p),
            lzma.decompress
        )
        if result:
            results.append(result)
            
    return {
        'filename': os.path.basename(filepath),
        'sparsity': sparsity,
        'results': results
    }

def save_results(data, filename):
    """Save benchmark results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    # Get all sample files
    # Assuming script is in submissions/benchmarks/
    samples_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'samples')
    # Fallback if running from root
    if not os.path.exists(samples_dir):
        samples_dir = os.path.join('samples')
        
    sample_files = sorted(glob.glob(os.path.join(samples_dir, 'activation_*.txt')))
    sample_files = [f for f in sample_files if not f.endswith('.compressed')]
    
    if not sample_files:
        print(f"No sample files found in {samples_dir}")
        return

    print("="*100)
    print("SPARSITY DISTRIBUTION BENCHMARK")
    print("="*100)
    print(f"Found {len(sample_files)} samples")
    print("Running benchmarks...")

    all_data = []
    for i, filepath in enumerate(sample_files):
        if (i+1) % 10 == 0:
            print(f"Processing {i+1}/{len(sample_files)}...")
        res = run_benchmarks_for_file(filepath)
        if res:
            all_data.append(res)

    if not all_data:
        print("No results generated.")
        return

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'sparsity_benchmark_results.json')
    save_results(all_data, output_file)

    # 1. Overall Top 3
    method_stats = defaultdict(lambda: {'scores': [], 'ratios': [], 'times': []})
    for data in all_data:
        for res in data['results']:
            m = res['method']
            method_stats[m]['scores'].append(res['score'])
            method_stats[m]['ratios'].append(res['compression_ratio'])
            method_stats[m]['times'].append(res['total_time'])

    aggregated = []
    for m, stats in method_stats.items():
        aggregated.append({
            'method': m,
            'avg_score': np.mean(stats['scores']),
            'avg_ratio': np.mean(stats['ratios']),
            'avg_time': np.mean(stats['times'])
        })
    
    aggregated.sort(key=lambda x: x['avg_score'], reverse=True)

    print("\n" + "="*100)
    print("TOP 3 ALGORITHMS OVERALL")
    print("="*100)
    print(f"{'Rank':<6} {'Method':<12} {'Avg Score':<12} {'Avg Ratio':<12} {'Avg Time':<12}")
    print("-"*100)
    for i, res in enumerate(aggregated[:3], 1):
        print(f"{i:<6} {res['method']:<12} {res['avg_score']:<12.6f} {res['avg_ratio']:<12.6f} {res['avg_time']:<12.6f}")

    # 2. Best by Sparsity Interval
    # Define intervals
    intervals = []
    # Fine-grained intervals for 0.0 to 0.05 (step 0.005)
    for i in range(10):
        intervals.append((i * 0.005, (i + 1) * 0.005))
    
    # Coarser intervals for 0.05 to 1.0 (step 0.05)
    current = 0.05
    while current < 1.0 - 1e-9:
        next_val = current + 0.05
        if next_val > 1.0: next_val = 1.0
        intervals.append((current, next_val))
        current = next_val

    print("\n" + "="*100)
    print(f"BEST ALGORITHM BY SPARSITY INTERVAL (Fine-grained for < 0.05)")
    print("="*100)
    print(f"{'Interval':<20} {'Samples':<10} {'Best Method':<15} {'Avg Score':<12}")
    print("-"*100)

    for low, high in intervals:
        # Find samples in this interval
        if high >= 1.0 - 1e-9:
            samples_in_interval = [d for d in all_data if low <= d['sparsity'] <= high + 1e-9]
        else:
            samples_in_interval = [d for d in all_data if low <= d['sparsity'] < high]
            
        if not samples_in_interval:
            print(f"{f'{low:.3f}-{high:.3f}':<20} {'0':<10} {'-':<15} {'-':<12}")
            continue
            
        # Find best method for these samples
        interval_stats = defaultdict(list)
        for d in samples_in_interval:
            for res in d['results']:
                interval_stats[res['method']].append(res['score'])
        
        best_method = None
        best_score = -1.0
        
        for m, scores in interval_stats.items():
            avg_s = np.mean(scores)
            if avg_s > best_score:
                best_score = avg_s
                best_method = m
                
        print(f"{f'{low:.3f}-{high:.3f}':<20} {len(samples_in_interval):<10} {best_method:<15} {best_score:<12.6f}")

if __name__ == "__main__":
    main()
