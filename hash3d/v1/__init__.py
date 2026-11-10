"""ShapeHash v1: the frozen placeholder format (see SPEC.md).

Python reference implementation: `decode` is normative together with SPEC.md;
`encode` mirrors the TypeScript encoder in ts/src/encode.
"""
from .decode import Decoded, decode, parse  # noqa: F401
from .format import FormatError  # noqa: F401
