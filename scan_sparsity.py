
import os
import json
import base64
import numpy as np

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

def scan_sparsity():
    sample_dir = 'samples'
    files = os.listdir(sample_dir)
    
    print(f"Scanning {len(files)} files...")
    
    results = []
    for f in files:
        if not f.endswith('.txt'): continue
        path = os.path.join(sample_dir, f)
        data = load_sample(path)
        if data is None: continue
        
        sparsity = 1.0 - (np.count_nonzero(data) / data.size)
        results.append((f, sparsity))
        
    results.sort(key=lambda x: x[1])
    
    print("\nSparsity Distribution:")
    for f, s in results:
        if 0.0 < s < 0.2:
            print(f"{f}: {s:.4f}")

scan_sparsity()
