
import os
import time
import json
import base64
import bz2
import lzma
import zlib
import numpy as np

def load_sample(filepath):
    with open(filepath, 'r') as f:
        raw_content = f.read().strip()
    
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict) and 'rows' in data:
            return np.array(data['rows'], dtype=np.int16)
    except:
        pass
        
    try:
        decoded = base64.b64decode(raw_content)
        return np.frombuffer(decoded, dtype=np.int16)
    except:
        pass
    return None

def benchmark_file(filepath):
    print(f"Benchmarking {os.path.basename(filepath)}...")
    data = load_sample(filepath)
    if data is None:
        print("Could not load data")
        return

    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    # Standard BZ2
    start = time.time()
    comp_std = bz2.compress(data_bytes, compresslevel=9)
    time_std = time.time() - start
    ratio_std = len(comp_std) / original_size
    score_std = max(0.0, min(1.0, (1 - ratio_std) * (1 - time_std / 1.012)))
    print(f"Standard BZ2: Ratio={ratio_std:.4f}, Time={time_std:.4f}s, Score={score_std:.4f}")
    
    # Split + BZ2
    start = time.time()
    lbs = data_bytes[0::2]
    hbs = data_bytes[1::2]
    comp_lbs = bz2.compress(lbs, compresslevel=9)
    comp_hbs = bz2.compress(hbs, compresslevel=9)
    total_len = len(comp_lbs) + len(comp_hbs)
    time_split = time.time() - start
    ratio_split = total_len / original_size
    score_split = max(0.0, min(1.0, (1 - ratio_split) * (1 - time_split / 1.012)))
    print(f"Split + BZ2:  Ratio={ratio_split:.4f}, Time={time_split:.4f}s, Score={score_split:.4f}")
    
    # Split + LZMA
    start = time.time()
    lbs = data_bytes[0::2]
    hbs = data_bytes[1::2]
    comp_lbs = lzma.compress(lbs, preset=1)
    comp_hbs = lzma.compress(hbs, preset=1)
    total_len = len(comp_lbs) + len(comp_hbs)
    time_split_lzma = time.time() - start
    ratio_split_lzma = total_len / original_size
    score_split_lzma = max(0.0, min(1.0, (1 - ratio_split_lzma) * (1 - time_split_lzma / 1.012)))
    print(f"Split + LZMA: Ratio={ratio_split_lzma:.4f}, Time={time_split_lzma:.4f}s, Score={score_split_lzma:.4f}")

    # Split + Zlib
    start = time.time()
    lbs = data_bytes[0::2]
    hbs = data_bytes[1::2]
    comp_lbs = zlib.compress(lbs, level=9)
    comp_hbs = zlib.compress(hbs, level=9)
    total_len = len(comp_lbs) + len(comp_hbs)
    time_split_zlib = time.time() - start
    ratio_split_zlib = total_len / original_size
    score_split_zlib = max(0.0, min(1.0, (1 - ratio_split_zlib) * (1 - time_split_zlib / 1.012)))
    print(f"Split + Zlib: Ratio={ratio_split_zlib:.4f}, Time={time_split_zlib:.4f}s, Score={score_split_zlib:.4f}")

sample_dir = 'samples'
target_file = 'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt'
full_path = os.path.join(sample_dir, target_file)

if os.path.exists(full_path):
    benchmark_file(full_path)
