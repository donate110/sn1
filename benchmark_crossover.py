
import numpy as np
import bz2
import lzma
import time


def benchmark_synthetic():
    print("Benchmarking Sparsity Crossover...")
    
    # Create a base random array (dense)
    size = 200000
    base_data = np.random.randint(-20000, 20000, size, dtype=np.int16)
    
    sparsities = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    print(f"{'Sparsity':<10} | {'BZ2 Score':<10} | {'LZMA(1) Score':<15} | {'Winner':<10}")
    print("-" * 60)
    
    for sp in sparsities:
        # Inject zeros
        data = base_data.copy()
        mask = np.random.random(size) < sp
        data[mask] = 0
        
        data_bytes = data.tobytes()
        original_size = len(data_bytes)
        
        # BZ2
        start = time.time()
        c_bz2 = bz2.compress(data_bytes, compresslevel=9)
        t_bz2 = time.time() - start
        r_bz2 = len(c_bz2) / original_size
        s_bz2 = max(0.0, min(1.0, (1 - r_bz2) * (1 - (t_bz2 * 2) / 1.012))) # Assuming decomp time approx same as comp time for safety, or just use comp time? 
        # The formula uses task_time (comp + decomp). 
        # Let's approximate decomp time as 0.5 * comp time for BZ2 and 0.3 * comp time for LZMA
        t_total_bz2 = t_bz2 * 1.5
        s_bz2 = max(0.0, min(1.0, (1 - r_bz2) * (1 - t_total_bz2 / 1.012)))

        # LZMA 1
        start = time.time()
        c_lzma = lzma.compress(data_bytes, preset=1)
        t_lzma = time.time() - start
        r_lzma = len(c_lzma) / original_size
        t_total_lzma = t_lzma * 1.3 # LZMA decomp is fast
        s_lzma = max(0.0, min(1.0, (1 - r_lzma) * (1 - t_total_lzma / 1.012)))
        
        winner = "BZ2" if s_bz2 > s_lzma else "LZMA"
        print(f"{sp:<10.2f} | {s_bz2:<10.4f} | {s_lzma:<15.4f} | {winner:<10}")

benchmark_synthetic()
