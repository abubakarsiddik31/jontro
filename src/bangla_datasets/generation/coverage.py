"""Coverage matrix: bias seed sampling away from over-represented tool combos."""
from __future__ import annotations

from collections import defaultdict

from bangla_datasets.generation.orchestrator import SeedPlan


class CoverageMatrix:
    """Tracks tool-combination frequency and computes a per-domain bias weight.

    `bias(domain)` returns a weight in (0, 1] — closer to 1 means under-represented
    (worth generating more), closer to 0 means over-represented (de-prioritize).
    """

    def __init__(self) -> None:
        self._domain_counts: dict[str, int] = defaultdict(int)
        self._combo_counts: dict[frozenset[str], int] = defaultdict(int)
        self._total = 0

    def record(self, plan: SeedPlan) -> None:
        self._domain_counts[plan.domain] += 1
        combo = frozenset(t.name for t in plan.tool_subset)
        self._combo_counts[combo] += 1
        self._total += 1

    def bias(self, domain: str) -> float:
        """Under-represented domains get higher bias. No data -> 1.0."""
        if self._total == 0:
            return 1.0
        share = self._domain_counts[domain] / self._total
        # If domain has its target share, bias -> 0; below target -> higher.
        return max(0.0, 1.0 - share * 8)  # heuristic decay

    def combo_count(self, plan: SeedPlan) -> int:
        return self._combo_counts[frozenset(t.name for t in plan.tool_subset)]
