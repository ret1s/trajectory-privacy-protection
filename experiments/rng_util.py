"""Deterministic, order-INDEPENDENT RNG derivation (verifier R3-002).

Round-3 seeded each mechanism from `SeedSequence([SEED, ei, mi])` where `ei` and
`mi` are the LIST POSITIONS of the epsilon and the mechanism. That is not
order-independent: reordering `EPSILONS` or the mechanism list, or inserting a
new mechanism before REM, changes the seed of unrelated runs.

Here the seed is derived from a STABLE SEMANTIC KEY
`"{schema}|{root_seed}|{eps:.17g}|{mechanism_name}"` via a SHA-256 digest. It
does not depend on run order, list position, or how many other mechanisms exist,
and it avoids Python's salted built-in `hash()` (which is randomized per
process). `eps` is formatted with `%.17g` so the exact IEEE-754 value keys the
stream (0.02 and 0.0200000001 differ).

The resolved substream words are exposed so provenance can record exactly which
stream produced which row.
"""
import hashlib

import numpy as np

RNG_SCHEMA = "rng-v1"


def key_words(*parts, schema=RNG_SCHEMA):
    """4 uint32 words (as a list) keyed by an arbitrary tuple of semantic parts.
    Floats are formatted with %.17g so the exact IEEE-754 value is part of the
    key; everything else via str(). Stable across process runs and list order."""
    tokens = [schema]
    for p in parts:
        tokens.append(f"{p:.17g}" if isinstance(p, float) else str(p))
    digest = hashlib.sha256("|".join(tokens).encode()).digest()[:16]
    return np.frombuffer(digest, dtype="<u4").tolist()


def rng_from_key(*parts, schema=RNG_SCHEMA):
    """A NumPy Generator seeded from an arbitrary stable semantic key."""
    return np.random.default_rng(np.random.SeedSequence(key_words(*parts, schema=schema)))


def seed_words(root_seed, eps, mechanism_name, schema=RNG_SCHEMA):
    """Benchmark stream key: (root_seed, exact ε, mechanism_name)."""
    return key_words(int(root_seed), float(eps), mechanism_name, schema=schema)


def semantic_rng(root_seed, eps, mechanism_name, schema=RNG_SCHEMA):
    """A NumPy Generator seeded from the (root_seed, ε, mechanism) key."""
    return np.random.default_rng(np.random.SeedSequence(
        seed_words(root_seed, eps, mechanism_name, schema)))
