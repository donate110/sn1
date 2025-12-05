import os
import json
import base64
import numpy as np
import collections
import math

def entropy(data):
    if len(data) == 0:
        return 0
    counts = collections.Counter(data)
    probs = [c / len(data) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def analyze_file(filepath):
    print(f"Analyzing {os.path.basename(filepath)}...")
    
    with open(filepath, 'r') as f:
        raw_content = f.read().strip()
    
    data = None
    try:
        json_data = json.loads(raw_content)
        if isinstance(json_data, dict) and 'rows' in json_data:
            data = np.array(json_data['rows'], dtype=np.int16)
    except:
        pass
        
    if data is None:
        try:
            decoded = base64.b64decode(raw_content)
            data = np.frombuffer(decoded, dtype=np.int16)
        except:
            print("Failed to load data")
            return

    print(f"Shape: {data.shape}")
    print(f"Size: {data.size}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}")
    print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
    
    flat_data = data.flatten()
    
    # Raw Entropy (treating as 16-bit symbols)
    print(f"Entropy (16-bit symbols): {entropy(flat_data):.4f} bits/symbol")
    
    # Byte Entropy
    byte_data = data.tobytes()
    print(f"Entropy (8-bit bytes): {entropy(byte_data):.4f} bits/byte")
    
    # Split Bytes
    # Little endian: [LB, HB, LB, HB, ...]
    lbs = byte_data[0::2]
    hbs = byte_data[1::2]
    
    print(f"Entropy (Low Bytes): {entropy(lbs):.4f} bits/byte")
    print(f"Entropy (High Bytes): {entropy(hbs):.4f} bits/byte")
    
    # Combined Split Entropy
    # If we compress LBs and HBs separately, the total size is roughly proportional to sum of entropies
    avg_split_entropy = (entropy(lbs) + entropy(hbs)) / 2
    print(f"Average Split Entropy: {avg_split_entropy:.4f} bits/byte")
    
    # Delta Encoding (1st order)
    delta1 = np.diff(flat_data)
    print(f"Delta (1st order) Entropy: {entropy(delta1):.4f} bits/symbol")
    
    # Delta Encoding (2nd order)
    delta2 = np.diff(delta1)
    print(f"Delta (2nd order) Entropy: {entropy(delta2):.4f} bits/symbol")
    
    # Transpose (if 2D)
    if len(data.shape) == 2:
        transposed = data.T.flatten()
        print(f"Transposed Entropy (16-bit): {entropy(transposed):.4f} bits/symbol")
        
        # Delta on Transposed
        delta_t = np.diff(transposed)
        print(f"Transposed Delta Entropy: {entropy(delta_t):.4f} bits/symbol")

sample_dir = 'samples'
target_file = 'activation_008d6431-b68d-4411-a4d4-240bc97a4dbd.txt'
full_path = os.path.join(sample_dir, target_file)

if os.path.exists(full_path):
    analyze_file(full_path)
else:
    print("File not found")
