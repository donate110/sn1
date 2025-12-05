
import os
import glob
import sys
import numpy as np
import base64
import io
import time

# Add submissions folder to path
sys.path.append(os.path.join(os.getcwd(), 'submissions'))
import submissions.code_submission_v2_random as submission

def test_samples(samples_dir='samples', limit=1000):
    sample_files = glob.glob(os.path.join(samples_dir, '*.txt'))
    
    if not sample_files:
        print(f"No sample files found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} samples. Testing first {limit}...")
    
    scores = []
    
    for i, sample_file in enumerate(sample_files[:limit]):
        # print(f"\nTesting {os.path.basename(sample_file)}...")
        
        # Read base64 data
        with open(sample_file, 'r') as f:
            data_base64 = f.read().strip()
            
        # Get original size
        original_bytes = base64.b64decode(data_base64)
        original_size = len(original_bytes)
        
        # Define paths
        compressed_path = f"temp_compressed_{i}.bin"
        decompressed_path = f"temp_decompressed_{i}.npy"
        
        # Create input objects
        compress_input = submission.CompressionInputDataSchema(
            data_to_compress_base64=data_base64,
            expected_output_filepath=compressed_path
        )
        
        # Measure compression time
        start_time = time.time()
        submission.compress(compress_input)
        compress_time = time.time() - start_time
        
        # Check compressed size
        if not os.path.exists(compressed_path):
            print("Error: Compressed file not created")
            continue
            
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = compressed_size / original_size
        
        # Prepare decompression
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        
        decompress_input = submission.DecompressionInputDataSchema(
            data_to_decompress_base64=compressed_base64,
            expected_output_filepath=decompressed_path
        )
        
        # Measure decompression time
        start_time = time.time()
        submission.decompress(decompress_input)
        decompress_time = time.time() - start_time
        
        total_time = compress_time + decompress_time
        
        # Verify correctness
        if not os.path.exists(decompressed_path):
            print("Error: Decompressed file not created")
            continue
            
        # Load original array
        buffer = io.BytesIO(original_bytes)
        arr_original = np.load(buffer, allow_pickle=False)
        
        # Load restored array
        arr_restored = np.load(decompressed_path)
        
        is_identical = np.array_equal(arr_original, arr_restored)
        
        if is_identical:
            # Calculate score based on formula:
            # score = np.clip((1 - compression) * (1 - task_time / (1 + 0.012)), 0.0, 1.0)
            # Note: The formula in the prompt was: score = np.clip((1 - compression) * (1 - task_time / (1 + 0.012)), 0.0, 1.0)
            # But usually task_time is normalized by some factor. The prompt says "1 + 0.012" which is ~1.012. 
            # If task_time is in seconds, and it's small, this might be fine.
            # Let's assume the formula is correct as given.
            
            score = max(0.0, min(1.0, (1 - compression_ratio) * (1 - total_time / 1.012)))
            
            # print(f"  Success: Arrays match")
            # print(f"  Original Size: {original_size} bytes")
            # print(f"  Compressed Size: {compressed_size} bytes")
            # print(f"  Ratio: {compression_ratio:.4f}")
            # print(f"  Time: {total_time:.4f}s (Comp: {compress_time:.4f}s, Decomp: {decompress_time:.4f}s)")
            # print(f"  Score: {score:.4f}")
            scores.append(score)
        else:
            print("  FAILURE: Arrays do not match")
            
        # Cleanup
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        if os.path.exists(decompressed_path):
            os.remove(decompressed_path)
            
    if scores:
        print(f"\nAverage Score: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    test_samples()
