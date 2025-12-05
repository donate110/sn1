"""
Comprehensive LZMA vs BZ2 Benchmark
Tests all LZMA presets and BZ2 levels across all samples
"""

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


def benchmark_compression(data_bytes, method_name, compress_func, decompress_func):
    """Benchmark a specific compression method."""
    original_size = len(data_bytes)
    
    # Measure compression time
    start_time = time.time()
    try:
        compressed = compress_func(data_bytes)
    except Exception as e:
        return None
    compress_time = time.time() - start_time
    
    compressed_size = len(compressed)
    
    # Measure decompression time
    start_time = time.time()
    try:
        decompressed = decompress_func(compressed)
    except Exception as e:
        return None
    decompress_time = time.time() - start_time
    
    # Verify correctness
    if decompressed != data_bytes:
        raise ValueError(f"{method_name}: Decompression failed!")
    
    total_time = compress_time + decompress_time
    compression_ratio = compressed_size / original_size
    
    # Calculate score using the formula
    score = max(0.0, min(1.0, (1 - compression_ratio) * (1 - total_time / 1.012)))
    
    return {
        'method': method_name,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'total_time': total_time,
        'score': score
    }


def run_benchmarks(arr):
    """Run all compression method benchmarks on a single array."""
    data_bytes = arr.tobytes()
    results = []
    
    # Test BZ2 with different compression levels (1-9)
    for level in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
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
    
    return results


def analyze_results(all_results):
    """Analyze and rank all methods."""
    # Aggregate by method
    method_stats = defaultdict(lambda: {
        'scores': [],
        'ratios': [],
        'times': [],
        'compress_times': [],
        'decompress_times': []
    })
    
    for result in all_results:
        method = result['method']
        method_stats[method]['scores'].append(result['score'])
        method_stats[method]['ratios'].append(result['compression_ratio'])
        method_stats[method]['times'].append(result['total_time'])
        method_stats[method]['compress_times'].append(result['compress_time'])
        method_stats[method]['decompress_times'].append(result['decompress_time'])
    
    # Calculate statistics
    rankings = []
    for method, stats in method_stats.items():
        rankings.append({
            'method': method,
            'avg_score': sum(stats['scores']) / len(stats['scores']),
            'min_score': min(stats['scores']),
            'max_score': max(stats['scores']),
            'std_score': np.std(stats['scores']),
            'avg_ratio': sum(stats['ratios']) / len(stats['ratios']),
            'min_ratio': min(stats['ratios']),
            'max_ratio': max(stats['ratios']),
            'avg_time': sum(stats['times']) / len(stats['times']),
            'min_time': min(stats['times']),
            'max_time': max(stats['times']),
            'avg_compress_time': sum(stats['compress_times']) / len(stats['compress_times']),
            'avg_decompress_time': sum(stats['decompress_times']) / len(stats['decompress_times']),
            'sample_count': len(stats['scores'])
        })
    
    # Sort by average score (descending)
    rankings.sort(key=lambda x: x['avg_score'], reverse=True)
    
    return rankings


def print_rankings(rankings):
    """Print detailed rankings."""
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS - ALL METHODS RANKED BY AVERAGE SCORE")
    print("="*100)
    print(f"{'Rank':<6} {'Method':<12} {'Avg Score':<12} {'Avg Ratio':<12} {'Avg Time':<12} {'Std Dev':<10}")
    print("-"*100)
    
    for i, result in enumerate(rankings, 1):
        marker = "⭐" if i <= 3 else ""
        print(f"{i:<6} {result['method']:<12} {result['avg_score']:<12.6f} "
              f"{result['avg_ratio']:<12.6f} {result['avg_time']:<12.6f} "
              f"{result['std_score']:<10.6f} {marker}")


def print_detailed_analysis(rankings):
    """Print detailed analysis of top performers."""
    print("\n" + "="*100)
    print("DETAILED ANALYSIS - TOP 5 METHODS")
    print("="*100)
    
    for i, result in enumerate(rankings[:5], 1):
        print(f"\n{i}. {result['method'].upper()}")
        print("-" * 50)
        print(f"  Average Score:       {result['avg_score']:.6f}")
        print(f"  Score Range:         {result['min_score']:.6f} - {result['max_score']:.6f}")
        print(f"  Score Std Dev:       {result['std_score']:.6f}")
        print(f"  Avg Compression:     {result['avg_ratio']:.4f} ({result['avg_ratio']*100:.2f}%)")
        print(f"  Ratio Range:         {result['min_ratio']:.4f} - {result['max_ratio']:.4f}")
        print(f"  Avg Total Time:      {result['avg_time']:.6f}s")
        print(f"  Time Range:          {result['min_time']:.6f}s - {result['max_time']:.6f}s")
        print(f"  Avg Compress Time:   {result['avg_compress_time']:.6f}s")
        print(f"  Avg Decompress Time: {result['avg_decompress_time']:.6f}s")
        print(f"  Samples Tested:      {result['sample_count']}")


def print_category_comparison(rankings):
    """Compare BZ2 vs LZMA categories."""
    print("\n" + "="*100)
    print("CATEGORY COMPARISON: BZ2 vs LZMA")
    print("="*100)
    
    bz2_methods = [r for r in rankings if r['method'].startswith('bz2')]
    lzma_methods = [r for r in rankings if r['method'].startswith('lzma')]
    
    if bz2_methods:
        best_bz2 = bz2_methods[0]
        print(f"\nBest BZ2:  {best_bz2['method']}")
        print(f"  Score: {best_bz2['avg_score']:.6f}, Ratio: {best_bz2['avg_ratio']:.4f}, Time: {best_bz2['avg_time']:.6f}s")
    
    if lzma_methods:
        best_lzma = lzma_methods[0]
        print(f"\nBest LZMA: {best_lzma['method']}")
        print(f"  Score: {best_lzma['avg_score']:.6f}, Ratio: {best_lzma['avg_ratio']:.4f}, Time: {best_lzma['avg_time']:.6f}s")
    
    if bz2_methods and lzma_methods:
        print(f"\nComparison:")
        score_diff = best_lzma['avg_score'] - best_bz2['avg_score']
        ratio_diff = best_lzma['avg_ratio'] - best_bz2['avg_ratio']
        time_diff = best_lzma['avg_time'] - best_bz2['avg_time']
        
        winner = "LZMA" if score_diff > 0 else "BZ2"
        print(f"  Winner: {winner}")
        print(f"  Score Difference:  {score_diff:+.6f} ({abs(score_diff/best_bz2['avg_score']*100):+.2f}%)")
        print(f"  Ratio Difference:  {ratio_diff:+.6f} ({abs(ratio_diff/best_bz2['avg_ratio']*100):+.2f}%)")
        print(f"  Time Difference:   {time_diff:+.6f}s ({abs(time_diff/best_bz2['avg_time']*100):+.2f}%)")


def print_recommendation(rankings):
    """Print final recommendation."""
    print("\n" + "="*100)
    print("FINAL RECOMMENDATION")
    print("="*100)
    
    winner = rankings[0]
    runner_up = rankings[1] if len(rankings) > 1 else None
    
    print(f"\n🏆 WINNER: {winner['method'].upper()}")
    print(f"{'='*50}")
    print(f"  Average Score:       {winner['avg_score']:.6f}")
    print(f"  Compression Ratio:   {winner['avg_ratio']:.4f} ({winner['avg_ratio']*100:.2f}%)")
    print(f"  Average Time:        {winner['avg_time']:.6f}s")
    print(f"  Consistency (StdDev):{winner['std_score']:.6f}")
    
    if runner_up:
        score_diff = ((winner['avg_score'] - runner_up['avg_score']) / runner_up['avg_score']) * 100
        print(f"\n  Beats {runner_up['method']} by {score_diff:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"USE THIS METHOD IN YOUR SUBMISSION!")
    
    # Extract algorithm and parameter
    if winner['method'].startswith('bz2'):
        level = winner['method'].split('-')[1]
        print(f"\nImplementation:")
        print(f"  import bz2")
        print(f"  compressed = bz2.compress(data, compresslevel={level})")
        print(f"  decompressed = bz2.decompress(compressed)")
    else:
        preset = winner['method'].split('-')[1]
        print(f"\nImplementation:")
        print(f"  import lzma")
        print(f"  compressed = lzma.compress(data, preset={preset})")
        print(f"  decompressed = lzma.decompress(compressed)")


def save_results(rankings, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(rankings, f, indent=2)
    print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    # Get all sample files
    samples_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'samples')
    sample_files = sorted(glob.glob(os.path.join(samples_dir, 'activation_*.txt')))
    
    # Filter out any .compressed files
    sample_files = [f for f in sample_files if not f.endswith('.compressed')]
    
    print("="*100)
    print("COMPREHENSIVE LZMA vs BZ2 BENCHMARK")
    print("="*100)
    print(f"Testing ALL samples in directory: {samples_dir}")
    print(f"Total samples found: {len(sample_files)}")
    print(f"Methods to test: BZ2 (levels 1-9), LZMA (presets 0-9)")
    print(f"Score formula: (1 - compression_ratio) * (1 - total_time / 1.012)")
    print("="*100 + "\n")
    
    all_results = []
    failed_samples = []
    
    for i, filepath in enumerate(sample_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(sample_files)}] {filename[:50]:<50}", end=' ')
        
        try:
            arr = load_sample_data(filepath)
            results = run_benchmarks(arr)
            all_results.extend(results)
            print(f"✓ ({len(results)} methods)")
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_samples.append((filename, str(e)))
            continue
    
    if failed_samples:
        print(f"\n⚠️  {len(failed_samples)} samples failed:")
        for filename, error in failed_samples[:5]:  # Show first 5
            print(f"  - {filename}: {error}")
    
    print(f"\n✓ Successfully tested {len(all_results) // 19} samples")  # 19 methods per sample
    print(f"  Total benchmark runs: {len(all_results)}")
    
    # Analyze results
    rankings = analyze_results(all_results)
    
    # Print all analysis
    print_rankings(rankings)
    print_detailed_analysis(rankings)
    print_category_comparison(rankings)
    print_recommendation(rankings)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    save_results(rankings, output_file)
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE!")
    print("="*100)


if __name__ == "__main__":
    main()
