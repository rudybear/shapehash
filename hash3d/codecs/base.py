"""Common codec interface.

    encode(asset, budget_bytes) -> str     (len(payload bytes) <= budget_bytes)
    decode(str) -> ColorMesh

Byte 0 of every payload: high nibble = method tag, low nibble = free for the
method (config class / flags).  The tag byte is counted in the budget, so a
"pick the best method per model" selector costs nothing extra.
"""
from ..bitio import BitReader, BitWriter, from_string, to_string

REGISTRY = {}  # name -> codec instance
BY_TAG = {}

# what the *decoder* needs, for the "can this run in JS on a phone" column
DECODER_NOTES = {}


class Codec:
    name = "base"
    tag = 0
    decoder_cost = "low"  # low | medium | high
    decoder_needs = ""

    def start(self, nibble=0):
        w = BitWriter()
        w.write(self.tag, 4)
        w.write(nibble, 4)
        return w

    def finish(self, w, budget_bytes):
        data = w.to_bytes()
        if len(data) > budget_bytes:
            raise ValueError(f"{self.name}: payload {len(data)} B exceeds budget {budget_bytes} B")
        return to_string(data)

    def encode(self, asset, budget_bytes) -> str:
        raise NotImplementedError

    def decode_payload(self, r: BitReader, nibble: int):
        raise NotImplementedError

    def decode(self, s: str):
        data = from_string(s)
        r = BitReader(data)
        tag = r.read(4)
        assert tag == self.tag, f"tag mismatch {tag} != {self.tag}"
        return self.decode_payload(r, r.read(4))


def register(cls):
    inst = cls()
    REGISTRY[inst.name] = inst
    BY_TAG[inst.tag] = inst
    return cls


def decode_any(s: str):
    """Decode a hash string of any method (dispatch on the tag nibble)."""
    tag = from_string(s)[0] >> 4
    return BY_TAG[tag].decode(s)


def payload_bytes(s: str) -> int:
    return len(from_string(s))
