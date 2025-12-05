
import numpy as np
import lzma
import bz2
import time

def benchmark_very_sparse():
    size = 500 * 500
    data = np.zeros(size, dtype=np.int16)
    # Fill 1% with random values
    indices = np.random.choice(size, int(size * 0.01), replace=False)
    data[indices] = np.random.randint(-1000, 1000, size=len(indices), dtype=np.int16)
    
    data_bytes = data.tobytes()
    print(f"Data size: {len(data_bytes)} bytes, Sparsity: 99%")
    
    strategies = [
        ("lzma_preset_1", lambda b: lzma.compress(b, preset=1)),
        ("bz2_9", lambda b: bz2.compress(b, compresslevel=9)),
    ]
    
    for name, func in strategies:
        start = time.time()
        compressed = func(data_bytes)
        comp_time = time.time() - start
        
        start = time.time()
        if "bz2" in name:
            bz2.decompress(compressed)
        else:
            lzma.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / len(data_bytes)
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"{name:<15}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

if __name__ == "__main__":
    benchmark_very_sparse()
