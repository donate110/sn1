"""
Matrix Compression Competition
"""

import numpy as np
import base64
import lzma
import struct
from pydantic import BaseModel
import io


LZMA2_FILTERS = [
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
    arr_bytes = base64.b64decode(input_data.data_to_compress_base64)
    buffer = io.BytesIO(arr_bytes)
    arr = np.load(buffer, allow_pickle=False)

    header = struct.pack('I', len(arr.shape))
    for dim in arr.shape:
        header += struct.pack('I', dim)
    dtype_str = arr.dtype.str.encode().ljust(8, b'\x00')
    header += dtype_str

    compressed = lzma.compress(arr.tobytes(), format=lzma.FORMAT_RAW, filters=LZMA2_FILTERS)

    with open(input_data.expected_output_filepath, 'wb') as f:
        f.write(header + compressed)


def decompress(input_data: DecompressionInputDataSchema) -> None:
    blob = base64.b64decode(input_data.data_to_decompress_base64)

    ndims = struct.unpack('I', blob[:4])[0]
    offset = 4
    shape = []
    for _ in range(ndims):
        dim = struct.unpack('I', blob[offset:offset+4])[0]
        shape.append(dim)
        offset += 4
    dtype_str = blob[offset:offset+8].rstrip(b'\x00').decode()
    offset += 8

    decompressed = lzma.decompress(blob[offset:], format=lzma.FORMAT_RAW, filters=LZMA2_FILTERS)
    arr = np.frombuffer(decompressed, dtype=dtype_str).reshape(shape)

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
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(f.read())

        for task in input_data.tasks:
            compress(input_data=task.input)
    elif args.decompress:
        with open(args.input_file, "r") as f:
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(f.read())

        for task in input_data.tasks:
            decompress(input_data=task.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()