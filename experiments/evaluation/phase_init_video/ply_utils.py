"""Small PLY vertex reader for PhysCtrl/LGM inference artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_ply_vertices(path: Path) -> np.ndarray:
    """Read x/y/z from an ASCII or binary scalar-property PLY file."""

    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        file_format = None
        vertex_count = None
        current_element = None
        vertex_properties: list[tuple[str, str]] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                break
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            fields = line.split()
            if fields[0] == "format":
                file_format = fields[1]
            elif fields[0] == "element":
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                if fields[1] not in _PLY_TYPES:
                    raise ValueError(f"Unsupported PLY scalar type: {fields[1]}")
                vertex_properties.append((fields[2], _PLY_TYPES[fields[1]]))

        if file_format is None or vertex_count is None:
            raise ValueError(f"PLY header lacks format or vertex count: {path}")
        names = [name for name, _ in vertex_properties]
        missing = [axis for axis in ("x", "y", "z") if axis not in names]
        if missing:
            raise ValueError(f"PLY vertices lack {missing}: {path}")

        if file_format == "ascii":
            values = np.loadtxt(handle, max_rows=vertex_count, ndmin=2)
            if values.shape != (vertex_count, len(vertex_properties)):
                raise ValueError(
                    f"Unexpected ASCII PLY vertex table {values.shape}: {path}"
                )
            indices = [names.index(axis) for axis in ("x", "y", "z")]
            return values[:, indices].astype(np.float32)

        endian = {
            "binary_little_endian": "<",
            "binary_big_endian": ">",
        }.get(file_format)
        if endian is None:
            raise ValueError(f"Unsupported PLY format {file_format}: {path}")
        dtype = np.dtype(
            [(name, endian + scalar_type) for name, scalar_type in vertex_properties]
        )
        values = np.fromfile(handle, dtype=dtype, count=vertex_count)
        if values.shape[0] != vertex_count:
            raise ValueError(f"Truncated binary PLY vertex table: {path}")
        return np.stack([values["x"], values["y"], values["z"]], axis=-1).astype(
            np.float32
        )
