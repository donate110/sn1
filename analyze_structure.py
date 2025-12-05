import os
import json
import base64
import numpy as np
import bz2
import lzma
import zlib
import time

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

def benchmark_split(filepath):
    print(f"Benchmarking Split Strategy on {os.path.basename(filepath)}...")
    data = load_sample(filepath)
    if data is None: return

    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    # Standard BZ2
    start = time.time()
    std_compressed = bz2.compress(data_bytes, compresslevel=9)
    std_time = time.time() - start
    std_ratio = len(std_compressed) / original_size
    print(f"Standard BZ2: Ratio={std_ratio:.4f}, Time={std_time:.4f}s")
    
    # Split Strategy
    # Separate low and high bytes
    low_bytes = data_bytes[0::2]
    high_bytes = data_bytes[1::2]
    
    # Concatenate them: High bytes first (more structure), then Low bytes
    split_data = high_bytes + low_bytes
    
    start = time.time()
    split_compressed = bz2.compress(split_data, compresslevel=9)
    split_time = time.time() - start
    split_ratio = len(split_compressed) / original_size
    print(f"Split + BZ2:  Ratio={split_ratio:.4f}, Time={split_time:.4f}s")
    
    # Delta Strategy
    # Use int16 subtraction
    # We need to handle overflow/underflow if we want to be exact, but for compression test
    # we can just use numpy diff which promotes to larger type, or cast back.
    # Let's use simple diff and cast to int16 (wrapping)
    
    # Simple delta: x[i] - x[i-1]
    # Prepend 0 to keep size same
    delta_data = np.diff(data, prepend=0).astype(np.int16)
    delta_bytes = delta_data.tobytes()
    
    start = time.time()
    delta_compressed = bz2.compress(delta_bytes, compresslevel=9)
    delta_time = time.time() - start
    delta_ratio = len(delta_compressed) / original_size
    print(f"Delta + BZ2:  Ratio={delta_ratio:.4f}, Time={delta_time:.4f}s")
    
    start = time.time()
    delta_lzma = lzma.compress(delta_bytes, preset=1)
    delta_lzma_time = time.time() - start
    delta_lzma_ratio = len(delta_lzma) / original_size
    print(f"Delta + LZMA: Ratio={delta_lzma_ratio:.4f}, Time={delta_lzma_time:.4f}s")

# Find a dense file
sample_dir = 'samples'
target_file = 'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt'
full_path = os.path.join(sample_dir, target_file)

if os.path.exists(full_path):
    benchmark_split(full_path)
