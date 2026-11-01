"""Generate pilot multi-turn, multi-tool trajectories and validate them."""
from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from bangla_datasets.gemini.client import GeminiClient
from bangla_datasets.generation.interactive_orchestrator import (
    InteractiveOrchestrator,
    MultiTurnSeedPlan,
)
from bangla_datasets.generation.scenarios import SCENARIO_BLUEPRINTS
from bangla_datasets.schema import Persona, Role
from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.validation.heuristics import validate_multi_turn_trajectory

OUT_DIR = Path("outputs/pilot_multiturn")
OUT_FILE = OUT_DIR / "trajectories.jsonl"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = os.getenv("BANGLA_DATASETS_MODEL", "gemini-3.5-flash")
    print(f"Connecting to Gemini client (model: {model})...")
    client = GeminiClient(model=model)
    registry = ToolRegistry()
    orch = InteractiveOrchestrator(client=client, registry=registry)

    # 5 Handcrafted diverse pilot scenarios
    pilot_configs = [
        # 1. Travel Booking in intimate 'tui' (User speaks tui, Assistant responds politely in apni)
        {
            "seed": 90001,
            "blueprint_name": "railway_travel_planner",
            "persona": Persona(
                age=24,
                profession="ছাত্র",
                location="urban",
                register="tui",
                script="bengali",
                tech_literacy="high",
            ),
            "goal": "ঢাকা থেকে চট্টগ্রাম যাওয়ার ট্রেনের সময় ও ভাড়া জেনে কালকের জন্য ১টি স্নিগ্ধা টিকিট কাটা।",
        },
        # 2. Healthcare Booking in familiar 'tumi'
        {
            "seed": 90002,
            "blueprint_name": "doctor_appointment_booking",
            "persona": Persona(
                age=32,
                profession="চাকরিজীবী",
                location="urban",
                register="tumi",
                script="bengali",
                tech_literacy="medium",
            ),
            "goal": "ঢাকায় শিশু বিশেষজ্ঞ ডাক্তার খুঁজে ২০২৬-০৭-২০ তারিখের উপযুক্ত স্লটে মেয়ে সাবিহার জন্য অ্যাপয়েন্টমেন্ট নেওয়া।",
        },
        # 3. Food Delivery in formal 'apni'
        {
            "seed": 90003,
            "blueprint_name": "food_order_and_track",
            "persona": Persona(
                age=45,
                profession="ব্যবসায়ী",
                location="urban",
                register="apni",
                script="bengali",
                tech_literacy="medium",
            ),
            "goal": "ধানমন্ডি এলাকার ভালো রেস্টুরেন্ট থেকে কাচ্চি বিরিয়ানি অর্ডার করা এবং ডেলিভারি ট্র্যাক করা।",
        },
        # 4. Cinema & Ride Booking in 'tui'
        {
            "seed": 90004,
            "blueprint_name": "cinema_movie_and_ride",
            "persona": Persona(
                age=21,
                profession="ছাত্র",
                location="urban",
                register="tui",
                script="bengali",
                tech_literacy="high",
            ),
            "goal": "স্টার সিনেপ্লেক্সে আজকের নতুন সিনেমার শো-টাইম দেখে ২টা টিকিট কাটা এবং সেখানে যাওয়ার রাইড বুক করা।",
        },
        # 5. Banglish Travel in 'tumi' (User in Banglish, Assistant in polite Banglish)
        {
            "seed": 90005,
            "blueprint_name": "railway_stay_payment",
            "persona": Persona(
                age=27,
                profession="প্রকৌশলী",
                location="urban",
                register="tumi",
                script="banglish",
                tech_literacy="high",
            ),
            "goal": "Dhaka theke Chittagong train ticket book kora ebong station-er kache bhalo hotel check kora.",
        },
    ]

    blueprints_by_name = {bp.name: bp for bp in SCENARIO_BLUEPRINTS}
    generated_trajectories = []

    print(f"\n--- Generating {len(pilot_configs)} Multi-Turn, Multi-Tool Trajectories ---\n")

    with OUT_FILE.open("w", encoding="utf-8") as f_out:
        for idx, cfg in enumerate(pilot_configs, 1):
            bp = blueprints_by_name[cfg["blueprint_name"]]
            plan = MultiTurnSeedPlan(
                seed=cfg["seed"],
                domain=bp.primary_domain,
                tool_subset=bp.tools,
                persona=cfg["persona"],
                goal=cfg["goal"],
                min_user_turns=2,
                max_user_turns=4,
                metadata={"scenario_type": bp.name},
            )

            print(f"[{idx}/{len(pilot_configs)}] Generating {bp.name} (Register: {cfg['persona'].register}, Script: {cfg['persona'].script})...")
            traj = orch.generate(plan)

            # Quality validation
            verdict = validate_multi_turn_trajectory(
                traj, min_user_turns=2, min_tools=2, forbid_terminal_question=True
            )
            status_str = "PASSED" if verdict.passed else f"FAILED: {verdict.reasons}"
            print(f"   -> Result: {len(traj.messages)} messages, tools used: {traj.metadata['tools_used']}, Quality Gate: {status_str}")

            traj_dict = traj.model_dump()
            traj_dict["validation"] = {"passed": verdict.passed, "reasons": verdict.reasons}
            f_out.write(json.dumps(traj_dict, ensure_ascii=False) + "\n")
            generated_trajectories.append((traj, verdict))

    print(f"\nWrote {len(generated_trajectories)} trajectories to {OUT_FILE}\n")


if __name__ == "__main__":
    main()
