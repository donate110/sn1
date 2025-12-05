
import os
import json
import base64
import numpy as np
import lzma
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

def benchmark_lzma_presets(filepath):
    print(f"Benchmarking LZMA Presets on {os.path.basename(filepath)}...")
    data = load_sample(filepath)
    if data is None: return

    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    for p in range(0, 7):
        start = time.time()
        compressed = lzma.compress(data_bytes, preset=p)
        comp_time = time.time() - start
        
        start = time.time()
        lzma.decompress(compressed)
        decomp_time = time.time() - start
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / original_size
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"Preset {p}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

sample_dir = 'samples'
target_file = 'activation_01f10ab6-b769-4fe0-8ce1-9a868e92c250.txt'
full_path = os.path.join(sample_dir, target_file)
if os.path.exists(full_path):
    benchmark_lzma_presets(full_path)
