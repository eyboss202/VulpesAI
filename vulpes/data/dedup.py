"""Exact and near-duplicate detection.

Why dedup: the web is full of copies (mirrors, syndicated articles, templated
pages). Training on the same text many times wastes compute and makes the
model memorise and regurgitate it. Two passes are used:

**Exact:** a 128-bit BLAKE2b hash of the normalised text. Identical documents
collide, so the first one is kept and the rest are dropped.

**Near-duplicates: MinHash + LSH** (Broder 1997; as used for GPT-3, The Pile
and FineWeb). Two documents are compared by the *Jaccard similarity* of their
sets of word 5-grams ("shingles"): |A ∩ B| / |A ∪ B|. Comparing every pair is
impossible, so:

1. *MinHash:* apply ``num_perm`` random hash functions to every shingle and
   keep each function's minimum. For any one function, P(min over A == min
   over B) is exactly the Jaccard similarity. So the fraction of equal
   entries in two signatures estimates it, using only ``num_perm`` numbers per
   document.
2. *LSH banding:* cut the signature into ``bands`` bands of ``rows`` values
   and hash each band. Similar documents very likely share at least one whole
   band; dissimilar ones almost never do (curve in ``DedupConfig``).

Documents are processed in a fixed order and the first of each near-duplicate
group is kept ("keep first"), so no all-pairs clustering is needed. Memory is
16 band keys of 8 bytes per kept document in a flat array (``KeySet``), plus the
exact hash: ~0.35 KB per document (measured), ~3.4 GB for the ~9.7M documents of
the whole ``sample-10BT`` (it was ~1.34 KB per document with Python sets).

Hash functions are ``h(x) = (a*x + b) mod p`` with p = 2^32 - 5 (prime) and
32-bit CRC shingle hashes. Everything fits in uint64 without overflow, so
NumPy does all of it in C.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from array import array
from collections.abc import Iterable

import numpy as np

from vulpes.data.config import DedupConfig

_PRIME = np.uint64(4294967291)  # largest prime below 2^32
_WORDS = re.compile(r"\w+")
_CHUNK = 4096  # shingles per NumPy block (caps memory at num_perm * 4096 * 8 bytes)


def exact_hash(text: str) -> bytes:
    """128-bit content hash (also used for the train/val split)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def shingles(text: str, n: int) -> np.ndarray:
    """Distinct word n-grams of the lowercased text, as 32-bit CRCs (in uint64)."""
    words = _WORDS.findall(text.lower())
    if len(words) <= n:
        grams = {" ".join(words)}
    else:
        grams = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    return np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))


class MinHasher:
    """Computes MinHash signatures and LSH band keys for a fixed ``DedupConfig``."""

    def __init__(self, cfg: DedupConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._a = rng.integers(1, int(_PRIME), size=(cfg.num_perm, 1), dtype=np.uint64)
        self._b = rng.integers(0, int(_PRIME), size=(cfg.num_perm, 1), dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """``num_perm`` minimum hash values (uint32)."""
        x = shingles(text, self.cfg.ngram)
        sig = np.full(self.cfg.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(x), _CHUNK):
            block = (self._a * x[None, start : start + _CHUNK] + self._b) % _PRIME
            np.minimum(sig, block.min(axis=1), out=sig)
        return sig.astype(np.uint32)

    def band_keys(self, sig: np.ndarray) -> list[int]:
        """One 64-bit key per band; the band index is mixed in so bands never collide with each other."""
        rows = sig.reshape(self.cfg.bands, self.cfg.rows)
        return [
            int.from_bytes(
                hashlib.blake2b(band.tobytes(), digest_size=8, salt=i.to_bytes(16, "little")).digest()
            )
            for i, band in enumerate(rows)
        ]


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class KeySet:
    """A set of 64-bit integers in one flat array: 4-9x less memory than a Python ``set``.

    Why: the near-dedup state is 16 band keys per kept document. In a Python ``set`` each
    key is an int object plus a hash-table slot, ~70 bytes, so ~1.3 KB per document:
    3.8 GB for Vulpes 0.1's 2.8M documents, but ~13 GB for the ~9.7M of the whole
    ``sample-10BT`` (Vulpes Flos 1.0, K3), too much for a 16 GB desktop. Here a key costs
    its 8 bytes divided by the load factor (0.35-0.7): a 2 GB table for those 155M keys.

    Open addressing with linear probing (Knuth, TAOCP vol. 3, 6.4). The keys are already
    uniform hashes, so their low bits pick the slot directly. 0 marks an empty slot, so
    the key 0 itself is kept in a flag. The array doubles when it is 70% full. Membership
    is exact, so dedup decisions are the same as with a ``set`` (tested).
    """

    _MAX_LOAD = 0.7

    def __init__(self, capacity: int = 1 << 16) -> None:
        if capacity <= 0 or capacity & (capacity - 1):
            raise ValueError(f"capacity must be a power of two, got {capacity}")
        self._slots = array("Q", [0]) * capacity
        self._mask = capacity - 1
        self._count = 0  # keys in the array (the key 0 is counted by the flag)
        self._has_zero = False

    def __len__(self) -> int:
        return self._count + self._has_zero

    def __contains__(self, key: int) -> bool:
        if key == 0:
            return self._has_zero
        slots, mask = self._slots, self._mask
        i = key & mask
        while k := slots[i]:
            if k == key:
                return True
            i = (i + 1) & mask
        return False

    def add(self, key: int) -> None:
        if key == 0:
            self._has_zero = True
            return
        slots, mask = self._slots, self._mask
        i = key & mask
        while k := slots[i]:
            if k == key:
                return
            i = (i + 1) & mask
        slots[i] = key
        self._count += 1
        if self._count > self._MAX_LOAD * len(slots):
            self._grow()

    def update(self, keys: Iterable[int]) -> None:
        for key in keys:
            self.add(key)

    def _grow(self) -> None:
        old = self._slots
        self._slots = array("Q", [0]) * (2 * len(old))
        self._mask = len(self._slots) - 1
        slots, mask = self._slots, self._mask
        for key in old:  # every key is new here: find its first free slot
            if key:
                i = key & mask
                while slots[i]:
                    i = (i + 1) & mask
                slots[i] = key

    @property
    def nbytes(self) -> int:
        return self._slots.itemsize * len(self._slots)


class Deduplicator:
    """Stateful keep-first dedup over a stream of documents."""

    def __init__(self) -> None:
        self._exact: set[bytes] = set()
        self._bands = KeySet()

    def is_exact_duplicate(self, digest: bytes) -> bool:
        """True if seen before; otherwise remember it and return False."""
        if digest in self._exact:
            return True
        self._exact.add(digest)
        return False

    def is_near_duplicate(self, band_keys: list[int]) -> bool:
        """True if any band matches a kept document; otherwise remember this one."""
        if any(k in self._bands for k in band_keys):
            return True
        self._bands.update(band_keys)
        return False
