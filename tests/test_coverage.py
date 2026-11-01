from bangla_datasets.generation.coverage import CoverageMatrix
from bangla_datasets.generation.orchestrator import SeedPlan
from bangla_datasets.schema import Persona
from bangla_datasets.tools.catalog import GET_TRAIN_FARE, SEARCH_TRAINS


def test_coverage_bias_decreases_with_repetition() -> None:
    cm = CoverageMatrix()
    plan = SeedPlan(seed=1, domain="railway_booking",
                    tool_subset=[SEARCH_TRAINS, GET_TRAIN_FARE], persona=Persona(), goal="x")
    cm.record(plan)
    cm.record(plan)
    b1 = cm.bias("railway_booking")
    assert b1 < 1.0
    plan2 = SeedPlan(seed=2, domain="railway_booking",
                     tool_subset=[SEARCH_TRAINS], persona=Persona(), goal="y")
    cm.record(plan2)
    # More recording -> the recorded domain stays low while a fresh domain is 1.0.
    assert cm.bias("railway_booking") < cm.bias("news")
