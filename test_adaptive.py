#!/usr/bin/env python3
"""
Test script for adaptive compression strategy
"""

import os
import sys
import numpy as np
import base64
import io
import time

sys.path.append(os.path.join(os.getcwd(), 'submissions'))
import code_submission_v2 as submission

def create_test_matrix(sparsity, size=1000):
    """Create a test matrix with specified sparsity."""
    arr = np.random.randn(size, size).astype(np.float32)
    if sparsity > 0:
        mask = np.random.rand(size, size) < sparsity
        arr[mask] = 0
    return arr

def test_compression(arr, test_name):
    """Test compression and decompression on a matrix."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Prepare base64 input
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    buffer.seek(0)
    arr_bytes = buffer.getvalue()
    data_base64 = base64.b64encode(arr_bytes).decode('utf-8')
    
    original_size = len(arr_bytes)
    sparsity = 1.0 - (np.count_nonzero(arr) / arr.size)
    
    print(f"Matrix shape: {arr.shape}")
    print(f"Matrix dtype: {arr.dtype}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Original size: {original_size:,} bytes")
    
    # Compression
    compress_input = submission.CompressionInputDataSchema(
        data_to_compress_base64=data_base64,
        expected_output_filepath="test_compressed.bin"
    )
    
    start_time = time.time()
    submission.compress(compress_input)
    compress_time = time.time() - start_time
    
    if not os.path.exists("test_compressed.bin"):
        print("❌ FAILED: Compressed file not created")
        return False
    
    compressed_size = os.path.getsize("test_compressed.bin")
    compression_ratio = compressed_size / original_size
    
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.4f}")
    print(f"Compression time: {compress_time:.4f}s")
    
    # Decompression
    with open("test_compressed.bin", "rb") as f:
        compressed_data = f.read()
    compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
    
    decompress_input = submission.DecompressionInputDataSchema(
        data_to_decompress_base64=compressed_base64,
        expected_output_filepath="test_decompressed.npy"
    )
    
    start_time = time.time()
    submission.decompress(decompress_input)
    decompress_time = time.time() - start_time
    
    print(f"Decompression time: {decompress_time:.4f}s")
    
    # Verify correctness
    if not os.path.exists("test_decompressed.npy"):
        print("❌ FAILED: Decompressed file not created")
        return False
    
    arr_restored = np.load("test_decompressed.npy")
    is_identical = np.array_equal(arr, arr_restored)
    
    total_time = compress_time + decompress_time
    print(f"Total time: {total_time:.4f}s")
    
    # Calculate score
    score = max(0.0, min(1.0, (1 - compression_ratio) * (1 - total_time / 1.012)))
    print(f"Score: {score:.6f}")
    
    # Cleanup
    for f in ["test_compressed.bin", "test_decompressed.npy"]:
        if os.path.exists(f):
            os.remove(f)
    
    if is_identical:
        print("✅ SUCCESS: Arrays match perfectly")
        return True
    else:
        print("❌ FAILED: Arrays do not match")
        max_diff = np.max(np.abs(arr - arr_restored))
        print(f"   Max difference: {max_diff}")
        return False

def main():
    print("Testing Adaptive Compression Strategy")
    print("="*60)
    
    tests = [
        ("Very Sparse Matrix (95% zeros)", create_test_matrix(0.95, 500)),
        ("Highly Sparse Matrix (85% zeros)", create_test_matrix(0.85, 500)),
        ("Moderately Sparse Matrix (60% zeros)", create_test_matrix(0.60, 500)),
        ("Low Sparse Matrix (30% zeros)", create_test_matrix(0.30, 500)),
        ("Dense Matrix (0% zeros)", create_test_matrix(0.0, 500)),
    ]
    
    results = []
    for test_name, test_matrix in tests:
        success = test_compression(test_matrix, test_name)
        results.append((test_name, success))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Adaptive compression is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - total_passed} test(s) failed.")

if __name__ == "__main__":
    main()
