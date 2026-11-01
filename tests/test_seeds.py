from pathlib import Path

import pytest

from bangla_datasets.generation.seeds import load_recipe, sample_seed
from bangla_datasets.utils.seeding import SeededRNG


def test_load_recipe_rejects_mismatched_domain_keys(tmp_path: Path) -> None:
    """A domain present in `domains` but missing from `goal_templates` (or vice
    versa) must raise a clear ValueError at load time, not a confusing KeyError
    mid-generation."""
    recipe = tmp_path / "bad.yaml"
    recipe.write_text(
        """
name: bad
target_count: 10
max_turns: 8
pass_threshold: 4
domains:
  railway_booking: { weight: 1.0, tool_count_range: [1, 1] }
  healthcare: { weight: 1.0, tool_count_range: [1, 1] }
goal_complexity_mix:
  single_shot: 1.0
persona_axes:
  age_range: [18, 70]
  locations: [urban]
  regions: [bangladesh]
  registers: [apni]
  tech_literacy: [medium]
goal_templates:
  railway_booking:
    - "x"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="goal_templates"):
        load_recipe(recipe)


def test_sample_seed_reproducible() -> None:
    rng1 = SeededRNG(seed=42)
    rng2 = SeededRNG(seed=42)
    r = load_recipe("recipes/agentic_v1.yaml")
    a = sample_seed(0, rng1, r)
    b = sample_seed(0, rng2, r)
    assert a.seed == b.seed
    assert a.domain == b.domain
    assert a.goal == b.goal


def test_sample_seed_tool_subset_size_in_range() -> None:
    r = load_recipe("recipes/agentic_v1.yaml")
    plan = sample_seed(5, SeededRNG(seed=99), r)
    assert 1 <= len(plan.tool_subset) <= 6


def test_sample_seed_aligns_with_loop_index() -> None:
    """The orchestrator assigns id f"bd-agentic-{plan.seed:06d}". For this to
    match the CLI checkpoint key f"bd-agentic-{i:06d}", plan.seed must equal the
    loop index. Regression guard for FIX 3 (id-space alignment)."""
    r = load_recipe("recipes/agentic_v1.yaml")
    plan = sample_seed(5, SeededRNG(seed=99), r)
    assert plan.seed == 5
    assert f"bd-agentic-{plan.seed:06d}" == "bd-agentic-000005"
