import os
import json
import base64
import numpy as np
import lzma
import bz2
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

def test_file(filename):
    filepath = os.path.join('samples', filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return

    data = load_sample(filepath)
    if data is None:
        print(f"Could not load {filename}")
        return

    raw_bytes = data.tobytes()
    original_size = len(raw_bytes)
    sparsity = 1.0 - (np.count_nonzero(data) / data.size)
    
    print(f"\nFile: {filename}")
    print(f"Original Size: {original_size} bytes")
    print(f"Sparsity: {sparsity:.4f}")

    # LZMA Preset 1
    start = time.time()
    lzma_1 = lzma.compress(raw_bytes, preset=1)
    t_lzma_1 = time.time() - start
    print(f"LZMA (1): {len(lzma_1)} bytes ({len(lzma_1)/original_size:.4f}) - {t_lzma_1:.4f}s")

    # LZMA Preset 3
    start = time.time()
    lzma_3 = lzma.compress(raw_bytes, preset=3)
    t_lzma_3 = time.time() - start
    print(f"LZMA (3): {len(lzma_3)} bytes ({len(lzma_3)/original_size:.4f}) - {t_lzma_3:.4f}s")

    # LZMA Preset 5
    start = time.time()
    lzma_5 = lzma.compress(raw_bytes, preset=5)
    t_lzma_5 = time.time() - start
    print(f"LZMA (5): {len(lzma_5)} bytes ({len(lzma_5)/original_size:.4f}) - {t_lzma_5:.4f}s")
    
    # LZMA Preset 9
    start = time.time()
    lzma_9 = lzma.compress(raw_bytes, preset=9)
    t_lzma_9 = time.time() - start
    print(f"LZMA (9): {len(lzma_9)} bytes ({len(lzma_9)/original_size:.4f}) - {t_lzma_9:.4f}s")

    # BZ2 Level 9
    start = time.time()
    bz2_9 = bz2.compress(raw_bytes, compresslevel=9)
    t_bz2_9 = time.time() - start
    print(f"BZ2 (9):  {len(bz2_9)} bytes ({len(bz2_9)/original_size:.4f}) - {t_bz2_9:.4f}s")

# Group 1: Very Dense
test_file('activation_00baf660-d0ce-48aa-bf8d-d0b1c37f6d17.txt')

# Group 2: Less Dense
test_file('activation_0155ae68-910e-4ad8-ac0e-30a81fa6b37d.txt')
