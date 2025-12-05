"""
Matrix Compression Competition Submission
"""

import numpy as np
import base64
import zlib
import bz2
import lzma
import struct
from pydantic import BaseModel
import io


class CompressionInputDataSchema(BaseModel):
    data_to_compress_base64: str
    expected_output_filepath: str


class DecompressionInputDataSchema(BaseModel):
    data_to_decompress_base64: str
    expected_output_filepath: str


class MatrixCompressTaskSchema(BaseModel):
    task_name: str
    input: CompressionInputDataSchema | DecompressionInputDataSchema


class MatrixCompressEvalInputDataSchema(BaseModel):
    tasks: list[MatrixCompressTaskSchema]


def compress(input_data: CompressionInputDataSchema) -> None:
    # Decode the base64 data back to numpy array inline
    arr_bytes = base64.b64decode(input_data.data_to_compress_base64)
    buffer = io.BytesIO(arr_bytes)
    arr = np.load(buffer, allow_pickle=False)
    
    # Determine compression method based on sparsity
    # Sparsity check is fast
    # Sparsity = 1 - density
    sparsity = 1.0 - (np.count_nonzero(arr) / arr.size)
    
    # Method ID: 1=BZ2, 2=LZMA
    if sparsity > 0.05:
        method = 2 # LZMA for sparse data
        # Preset 1 is faster and provides good enough compression for sparse data
        compressed = lzma.compress(arr.tobytes(), preset=1)
    else:
        method = 1 # BZ2 for dense data
        # Level 9 is default but explicit is better
        compressed = bz2.compress(arr.tobytes(), compresslevel=9)
    
    header = struct.pack('B', method)
    header += struct.pack('I', len(arr.shape))
    for dim in arr.shape:
        header += struct.pack('I', dim)
    dtype_str = arr.dtype.str.encode().ljust(8, b'\x00')
    header += dtype_str
    
    with open(input_data.expected_output_filepath, 'wb') as f:
        f.write(header + compressed)


def decompress(input_data: DecompressionInputDataSchema) -> None:
    blob = base64.b64decode(input_data.data_to_decompress_base64)
    
    # Parse header
    offset = 0
    
    method = struct.unpack('B', blob[offset:offset+1])[0]
    offset += 1
    
    ndims = struct.unpack('I', blob[offset:offset+4])[0]
    offset += 4
    shape = []
    for _ in range(ndims):
        dim = struct.unpack('I', blob[offset:offset+4])[0]
        shape.append(dim)
        offset += 4
    dtype_str = blob[offset:offset+8].rstrip(b'\x00').decode()
    offset += 8
    
    # Decompress body
    if method == 2:
        decompressed = lzma.decompress(blob[offset:])
    elif method == 1:
        decompressed = bz2.decompress(blob[offset:])
    else:
        # Fallback
        decompressed = zlib.decompress(blob[offset:])

    arr = np.frombuffer(decompressed, dtype=dtype_str).reshape(shape)
    
    # Save to output path
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    buffer.seek(0)
    result_bytes = buffer.getvalue()

    with open(input_data.expected_output_filepath, "wb") as f:
        f.write(result_bytes)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--input-file", type=str)
    args = parser.parse_args()

    if args.compress:
        with open(args.input_file, "r") as f:
            content = f.read().strip()
            
        try:
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(content)
        except Exception:
            # Fallback for raw base64 files (local testing)
            # Create a dummy task wrapper
            print("Warning: Input file is not JSON. Treating as raw base64 data.")
            output_path = args.input_file + ".compressed"
            input_data = MatrixCompressEvalInputDataSchema(
                tasks=[
                    MatrixCompressTaskSchema(
                        task_name="local_test",
                        input=CompressionInputDataSchema(
                            data_to_compress_base64=content,
                            expected_output_filepath=output_path
                        )
                    )
                ]
            )

        for task in input_data.tasks:
            compress(input_data=task.input)
    elif args.decompress:
        with open(args.input_file, "r") as f:
            content = f.read().strip()

        try:
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(content)
        except Exception:
            # Fallback for raw base64 files (local testing)
            print("Warning: Input file is not JSON. Treating as raw base64 data.")
            output_path = args.input_file + ".decompressed.npy"
            input_data = MatrixCompressEvalInputDataSchema(
                tasks=[
                    MatrixCompressTaskSchema(
                        task_name="local_test",
                        input=DecompressionInputDataSchema(
                            data_to_decompress_base64=content,
                            expected_output_filepath=output_path
                        )
                    )
                ]
            )

        for task in input_data.tasks:
            decompress(input_data=task.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
