
import numpy as np
import lzma
import bz2
import zlib
import time
import os

def benchmark_compression():
    # Simulate dense activation data (int16, similar to benchmark_v3.py)
    # Activations often have some structure, but let's try random first
    size = 500 * 500
    data = np.random.randint(-1000, 1000, size=size, dtype=np.int16)
    data_bytes = data.tobytes()
    
    print(f"Data size: {len(data_bytes)} bytes")
    
    strategies = [
        ("lzma_preset_4", lambda b: lzma.compress(b, preset=4)),
        ("lzma_preset_2", lambda b: lzma.compress(b, preset=2)),
        ("bz2_9", lambda b: bz2.compress(b, compresslevel=9)),
        ("zlib_9", lambda b: zlib.compress(b, level=9)),
    ]
    
    # Add delta filter for LZMA
    filters_delta = [
        {"id": lzma.FILTER_DELTA, "dist": 2}, # 2 bytes for int16
        {"id": lzma.FILTER_LZMA2, "preset": 4}
    ]
    strategies.append(("lzma_delta_p4", lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=filters_delta)))

    for name, func in strategies:
        start = time.time()
        compressed = func(data_bytes)
        comp_time = time.time() - start
        
        # Decompress to check time
        start = time.time()
        if "bz2" in name:
            bz2.decompress(compressed)
        elif "zlib" in name:
            zlib.decompress(compressed)
        elif "delta" in name:
            lzma.decompress(compressed, format=lzma.FORMAT_RAW, filters=filters_delta)
        else:
            lzma.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / len(data_bytes)
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"{name:<15}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

if __name__ == "__main__":
    benchmark_compression()
