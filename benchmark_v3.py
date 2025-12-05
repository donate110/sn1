
import os
import time
import json
import base64
import bz2
import lzma
import zlib
import numpy as np

LZMA2_FILTERS_V1 = [
    {
        "id": lzma.FILTER_LZMA2,
        "preset": 4,
        "dict_size": 640 * 1024,
        "lc": 2,
        "lp": 1,
        "pb": 2,
        "mf": lzma.MF_HC4,
    }
]

def load_sample(filepath):
    with open(filepath, 'r') as f:
        raw_content = f.read().strip()
    
    # Try JSON
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict) and 'rows' in data:
            return np.array(data['rows'], dtype=np.int16)
    except json.JSONDecodeError:
        pass
        
    # Try Base64
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
    
    sparsity = 1.0 - (np.count_nonzero(data) / data.size)
    print(f"Sparsity: {sparsity:.4f}")
    
    strategies = [
        ('bz2_9 (v2 dense)', lambda d: bz2.compress(d, compresslevel=9)),
        ('lzma_1 (v2 sparse)', lambda d: lzma.compress(d, preset=1)),
        ('lzma_v1 (custom)', lambda d: lzma.compress(d, format=lzma.FORMAT_RAW, filters=LZMA2_FILTERS_V1)),
    ]
    
    for name, func in strategies:
        start_time = time.time()
        compressed = func(data_bytes)
        comp_time = time.time() - start_time
        
        # Decompression check
        start_time = time.time()
        if 'bz2' in name:
            bz2.decompress(compressed)
        elif 'lzma_v1' in name:
            lzma.decompress(compressed, format=lzma.FORMAT_RAW, filters=LZMA2_FILTERS_V1)
        else:
            lzma.decompress(compressed)
        decomp_time = time.time() - start_time
        
        total_time = comp_time + decomp_time
        ratio = len(compressed) / original_size
        
        # Score formula
        score = max(0.0, min(1.0, (1 - ratio) * (1 - total_time / 1.012)))
        
        print(f"  {name}: Ratio={ratio:.4f}, Time={total_time:.4f}s, Score={score:.4f}")

sample_dir = 'samples'
files_to_test = [
    'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt', # Dense
    'activation_01f10ab6-b769-4fe0-8ce1-9a868e92c250.txt'  # Sparse
]

for f in files_to_test:
    full_path = os.path.join(sample_dir, f)
    if os.path.exists(full_path):
        benchmark_file(full_path)
    else:
        print(f"File {f} not found")
