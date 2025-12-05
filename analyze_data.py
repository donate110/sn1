
import os
import glob
import numpy as np
import base64
import io
import zlib
import lzma
import bz2
import time

def analyze_samples(samples_dir='samples', limit=10):
    sample_files = glob.glob(os.path.join(samples_dir, '*.txt'))
    
    if not sample_files:
        print(f"No sample files found in {samples_dir}")
        return

    print(f"Analyzing {limit} samples...")
    
    for i, sample_file in enumerate(sample_files[:limit]):
        print(f"\n--- {os.path.basename(sample_file)} ---")
        
        # Read base64 data
        with open(sample_file, 'r') as f:
            data_base64 = f.read().strip()
            
        # Get original size
        original_bytes = base64.b64decode(data_base64)
        buffer = io.BytesIO(original_bytes)
        arr = np.load(buffer, allow_pickle=False)
        
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        
        # Sparsity
        zeros = np.count_nonzero(arr == 0)
        total = arr.size
        sparsity = zeros / total
        print(f"Sparsity: {sparsity:.4f} ({zeros}/{total})")
        
        # Unique values
        unique_count = len(np.unique(arr))
        print(f"Unique values: {unique_count}")
        
        # Min/Max
        print(f"Min: {arr.min()}, Max: {arr.max()}")
        
        # Benchmark Compressors
        raw_bytes = arr.tobytes()
        
        def bench(name, compress_func, decompress_func):
            # Compress
            start = time.time()
            compressed = compress_func(raw_bytes)
            comp_time = time.time() - start
            
            # Decompress
            start = time.time()
            _ = decompress_func(compressed)
            decomp_time = time.time() - start
            
            total_time = comp_time + decomp_time
            ratio = len(compressed) / len(raw_bytes)
            
            # Score
            score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
            
            print(f"{name}: Ratio={ratio:.4f}, Time={total_time:.6f}s (C:{comp_time:.4f}, D:{decomp_time:.4f}), Score={score:.4f}")

        # Zlib 1
        bench("Zlib 1", lambda d: zlib.compress(d, level=1), zlib.decompress)
        
        # Zlib 9
        bench("Zlib 9", lambda d: zlib.compress(d, level=9), zlib.decompress)
        
        # LZMA
        bench("LZMA  ", lzma.compress, lzma.decompress)
        
        # BZ2
        bench("BZ2   ", bz2.compress, bz2.decompress)

        # Delta + BZ2
        def delta_compress(data_bytes):
            # Convert bytes to numpy, compute delta, convert back
            arr = np.frombuffer(data_bytes, dtype=np.uint8)
            # Simple delta: x[i] - x[i-1]
            # We need to store the first element separately or assume 0
            # But for uint8, we need to handle overflow/underflow or use int8
            # Let's try simple byte-wise delta
            # This is slow in python, but let's see if numpy is fast enough
            delta = np.diff(arr, prepend=0).astype(np.uint8)
            return bz2.compress(delta.tobytes())

        def delta_decompress(data_bytes):
            decompressed = bz2.decompress(data_bytes)
            delta = np.frombuffer(decompressed, dtype=np.uint8)
            # Cumsum to restore
            arr = np.cumsum(delta, dtype=np.uint8).astype(np.uint8)
            return arr.tobytes()

        bench("Delta+BZ2", delta_compress, delta_decompress)

if __name__ == "__main__":
    analyze_samples()
