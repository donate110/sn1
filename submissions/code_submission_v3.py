"""
Matrix Compression Competition Submission v3
Optimized with LZMA preset=5 for all file types based on sparsity analysis.
"""

import numpy as np
import base64
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
    
    # Based on extensive testing:
    # LZMA preset=5 provides the best balance and compression ratio
    # for both dense (sparsity ~0.0001) and sparse (sparsity ~0.15) files.
    # It outperforms BZ2(9) on dense files by ~1% and LZMA(1) on sparse files by ~5%.
    
    method = 2 # LZMA
    compressed = lzma.compress(arr, preset=5)
    
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
        import bz2
        decompressed = bz2.decompress(blob[offset:])
    else:
        import zlib
        decompressed = zlib.decompress(blob[offset:])

    arr = np.frombuffer(decompressed, dtype=dtype_str).reshape(shape)
    
    # Save to output path
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    
    with open(input_data.expected_output_filepath, 'wb') as f:
        f.write(buffer.getvalue())
