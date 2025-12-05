
import numpy as np
import bz2
import time

def benchmark_bz2_levels():
    size = 500 * 500
    data = np.random.randint(-1000, 1000, size=size, dtype=np.int16).tobytes()
    
    for level in [1, 5, 9]:
        start = time.time()
        compressed = bz2.compress(data, compresslevel=level)
        comp_time = time.time() - start
        
        start = time.time()
        bz2.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / len(data)
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"bz2_lvl_{level}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

if __name__ == "__main__":
    benchmark_bz2_levels()
