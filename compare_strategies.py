#!/usr/bin/env python3
"""
Compare original vs adaptive compression strategy
"""

import os
import sys
import numpy as np
import base64
import io
import time
import lzma
import bz2

# Original LZMA filters (non-adaptive)
LZMA_FILTERS_ORIGINAL = [
    {
        "id": lzma.FILTER_LZMA2,
        "preset": 4,
        "dict_size": 640 * 1024,
        "lc": 2,
        "lp": 1,
        "pb": 2,
        "mf": lzma.MF_HC4,
    }
]

sys.path.append(os.path.join(os.getcwd(), 'submissions'))
import code_submission_v2 as submission

def compress_original(arr):
    """Original compression (always LZMA preset 4)"""
    data_bytes = arr.tobytes()
    return lzma.compress(data_bytes, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS_ORIGINAL)

def compare_strategies(arr, test_name):
    """Compare original vs adaptive compression."""
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")
    
    sparsity = 1.0 - (np.count_nonzero(arr) / arr.size)
    print(f"Matrix: {arr.shape}, dtype: {arr.dtype}, sparsity: {sparsity:.1%}")
    
    # Test ORIGINAL strategy
    start = time.time()
    compressed_orig = compress_original(arr)
    time_orig = time.time() - start
    ratio_orig = len(compressed_orig) / arr.nbytes
    score_orig = max(0.0, min(1.0, (1 - ratio_orig) * (1 - time_orig / 1.012)))
    
    # Test ADAPTIVE strategy
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    buffer.seek(0)
    data_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    compress_input = submission.CompressionInputDataSchema(
        data_to_compress_base64=data_base64,
        expected_output_filepath="temp_test.bin"
    )
    
    start = time.time()
    submission.compress(compress_input)
    time_adaptive = time.time() - start
    
    size_adaptive = os.path.getsize("temp_test.bin")
    ratio_adaptive = size_adaptive / (arr.nbytes + len(buffer.getvalue()) - arr.nbytes)
    score_adaptive = max(0.0, min(1.0, (1 - ratio_adaptive) * (1 - time_adaptive / 1.012)))
    
    os.remove("temp_test.bin")
    
    # Results
    print(f"\n  Original (always LZMA-4):")
    print(f"    Ratio: {ratio_orig:.4f}, Time: {time_orig:.4f}s, Score: {score_orig:.4f}")
    print(f"\n  Adaptive (smart selection):")
    print(f"    Ratio: {ratio_adaptive:.4f}, Time: {time_adaptive:.4f}s, Score: {score_adaptive:.4f}")
    
    improvement = ((score_adaptive - score_orig) / score_orig * 100) if score_orig > 0 else 0
    print(f"\n  📊 Improvement: {improvement:+.2f}%")
    
    return score_orig, score_adaptive

def main():
    print("="*70)
    print("COMPARISON: Original vs Adaptive Compression Strategy")
    print("="*70)
    
    tests = [
        ("Very Sparse (95% zeros)", np.random.randn(500, 500).astype(np.float32)),
        ("Moderately Sparse (60% zeros)", np.random.randn(500, 500).astype(np.float32)),
        ("Dense (0% zeros)", np.random.randn(500, 500).astype(np.float32)),
    ]
    
    # Apply sparsity
    mask = np.random.rand(500, 500) < 0.95
    tests[0][1][mask] = 0
    
    mask = np.random.rand(500, 500) < 0.60
    tests[1][1][mask] = 0
    
    orig_scores = []
    adaptive_scores = []
    
    for test_name, test_arr in tests:
        score_o, score_a = compare_strategies(test_arr, test_name)
        orig_scores.append(score_o)
        adaptive_scores.append(score_a)
    
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Average Original Score:  {np.mean(orig_scores):.4f}")
    print(f"Average Adaptive Score:  {np.mean(adaptive_scores):.4f}")
    
    overall_improvement = ((np.mean(adaptive_scores) - np.mean(orig_scores)) / np.mean(orig_scores) * 100)
    print(f"Overall Improvement:     {overall_improvement:+.2f}%")
    
    if overall_improvement > 0:
        print(f"\n✅ Adaptive strategy is BETTER by {overall_improvement:.2f}%")
    else:
        print(f"\n⚠️  Original strategy was better by {-overall_improvement:.2f}%")

if __name__ == "__main__":
    main()
