from bangla_datasets.utils.seeding import SeededRNG, derive_seed


def test_seeded_rng_reproducible() -> None:
    a = SeededRNG(seed=42)
    b = SeededRNG(seed=42)
    assert a.integers(0, 100) == b.integers(0, 100)
    assert a.choice(["x", "y", "z"]) == b.choice(["x", "y", "z"])


def test_seeded_rng_different_seeds_diverge() -> None:
    a = SeededRNG(seed=42)
    b = SeededRNG(seed=43)
    assert a.integers(0, 10000) != b.integers(0, 10000)


def test_derive_seed_deterministic() -> None:
    assert derive_seed(42, "persona") == derive_seed(42, "persona")
    assert derive_seed(42, "persona") != derive_seed(42, "tools")


def test_seeded_rng_float_range() -> None:
    rng = SeededRNG(seed=1)
    val = rng.float(0.0, 1.0)
    assert 0.0 <= val < 1.0
