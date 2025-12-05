
import os
import json
import base64
import numpy as np
import zlib
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

def benchmark_zlib(filepath):
    print(f"Benchmarking Zlib Strategies on {os.path.basename(filepath)}...")
    data = load_sample(filepath)
    if data is None: return

    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    strategies = [
        ('DEFAULT', zlib.Z_DEFAULT_STRATEGY),
        ('FILTERED', zlib.Z_FILTERED),
        ('HUFFMAN_ONLY', zlib.Z_HUFFMAN_ONLY),
        ('RLE', zlib.Z_RLE),
        ('FIXED', zlib.Z_FIXED)
    ]
    
    for name, strat in strategies:
        start = time.time()
        c_obj = zlib.compressobj(level=9, strategy=strat)
        compressed = c_obj.compress(data_bytes) + c_obj.flush()
        comp_time = time.time() - start
        
        start = time.time()
        zlib.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / original_size
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"Zlib {name}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

sample_dir = 'samples'
target_file = 'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt'
full_path = os.path.join(sample_dir, target_file)
if os.path.exists(full_path):
    benchmark_zlib(full_path)
