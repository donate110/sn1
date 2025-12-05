
import numpy as np
import zlib
import time

def benchmark_zlib_sparse():
    size = 500 * 500
    data = np.zeros(size, dtype=np.int16)
    indices = np.random.choice(size, int(size * 0.1), replace=False)
    data[indices] = np.random.randint(-1000, 1000, size=len(indices), dtype=np.int16)
    data_bytes = data.tobytes()
    
    start = time.time()
    compressed = zlib.compress(data_bytes, level=9)
    comp_time = time.time() - start
    
    start = time.time()
    zlib.decompress(compressed)
    decomp_time = time.time() - start
    
    total_time = comp_time + decomp_time
    ratio = len(compressed) / len(data_bytes)
    score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
    
    print(f"zlib_9: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

if __name__ == "__main__":
    benchmark_zlib_sparse()
