
import os
import json
import base64
import numpy as np
import bz2
import time

def load_sample(filepath):
    with open(filepath, 'r') as f:
        raw_content = f.read().strip()
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict) and 'rows' in data:
            return np.array(data['rows'], dtype=np.int16)
    except: pass
    try:
        decoded = base64.b64decode(raw_content)
        return np.frombuffer(decoded, dtype=np.int16)
    except: pass
    return None

def benchmark_bz2_levels(filepath):
    print(f"Benchmarking BZ2 Levels on {os.path.basename(filepath)}...")
    data = load_sample(filepath)
    if data is None: return

    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    for level in range(1, 10):
        start = time.time()
        compressed = bz2.compress(data_bytes, compresslevel=level)
        comp_time = time.time() - start
        
        # Decomp time approx
        start = time.time()
        bz2.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / original_size
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"Level {level}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

sample_dir = 'samples'
target_file = 'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt'
full_path = os.path.join(sample_dir, target_file)
if os.path.exists(full_path):
    benchmark_bz2_levels(full_path)
