"""Deterministic, seeded tool simulators. Pure Python — no LLM, no network.

These are the anti-hallucination guarantee: a model never produces a tool
output, only the matching simulator does, seeded for byte-determinism.
"""

import json
import random
from collections.abc import Callable


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _search_trains(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    trains = [
        {
            "train_id": "SUBARNA-701",
            "departure": "06:00",
            "arrival": "10:30",
            "classes": ["shovan", "snigdha", "ac_b"],
        },
        {
            "train_id": "MOHANAGAR-712",
            "departure": "08:15",
            "arrival": "13:00",
            "classes": ["shovan_chair", "ac_s"],
        },
        {
            "train_id": "TURNANISHA-743",
            "departure": "14:30",
            "arrival": "19:15",
            "classes": ["shovan", "snigdha"],
        },
    ]
    # Deterministically include 2-3 trains based on seed. Pool size 3, max k=3.
    n = 2 + (rng.randint(0, 99) % 2)
    chosen = rng.sample(trains, k=n)
    return {"from": args["from"], "to": args["to"], "date": args["date"], "trains": chosen}


def _get_train_fare(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    base = {"shovan": 215, "shovan_chair": 280, "snigdha": 645, "ac_b": 880, "ac_s": 1115}
    seat_class = str(args["seat_class"])
    fare = base.get(seat_class, 300)
    # Deterministic small variation.
    fare += rng.randint(-10, 10)
    return {"train_id": args["train_id"], "seat_class": seat_class, "fare_bdt": fare}


def _book_ticket(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    pnr = f"PNR-{rng.randint(100000, 999999)}"
    return {
        "status": "confirmed",
        "pnr": pnr,
        "train_id": args["train_id"],
        "seat_class": args["seat_class"],
        "passenger_name": args["passenger_name"],
        "seat_count": args["seat_count"],
    }


def _send_money(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    txid = f"TX{rng.randint(10**11, 10**12 - 1)}"
    return {
        "status": "success",
        "txid": txid,
        "to": args["to"],
        "amount": args["amount"],
        "fee_bdt": 5,
    }


def _check_balance(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {"balance_bdt": rng.randint(500, 50000)}


def _pay_bill(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "success",
        "biller": args["biller"],
        "amount": args["amount"],
        "receipt_no": f"RCP{rng.randint(100000, 999999)}",
    }


def _search_doctors(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    location = str(args.get("location", "ঢাকা"))
    pool = [
        {
            "doctor_id": "DR-001",
            "name": "ডাঃ করিম",
            "specialty": args["specialty"],
            "location": location,
        },
        {
            "doctor_id": "DR-002",
            "name": "ডাঃ রহিমা",
            "specialty": args["specialty"],
            "location": location,
        },
        {
            "doctor_id": "DR-003",
            "name": "ডাঃ সাব্বির",
            "specialty": args["specialty"],
            "location": "চট্টগ্রাম",
        },
    ]
    # Pool size 3, max k = 2 + 1 = 3.
    return {"doctors": rng.sample(pool, k=2 + rng.randint(0, 1))}


def _get_available_slots(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    hours = ["09:00", "10:30", "12:00", "15:00", "17:30"]
    # Pool size 5, k=3.
    return {
        "doctor_id": args["doctor_id"],
        "slots": [{"slot_id": f"S{i}", "time": h} for i, h in enumerate(rng.sample(hours, k=3))],
    }


def _book_appointment(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "confirmed",
        "doctor_id": args["doctor_id"],
        "slot_id": args["slot_id"],
        "patient_name": args["patient_name"],
        "appointment_no": f"AP{rng.randint(1000, 9999)}",
    }


def _search_products(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    query = str(args["query"])
    min_price = int(args.get("min_price", 0))
    max_price = int(args.get("max_price", 999999))
    product_pool = [
        ("স্যামসাং গ্যালাক্সি A05", 12500), ("ওয়ালটন প্রিমো N3", 10990),
        ("শাওমি রেডমি 13C", 13500), ("রিয়েলমি C53", 14990),
        ("ভিভো Y17", 16500), ("অপ্পো A17", 13990),
        ("টেকনো স্পার্ক 20", 11990), ("ইনফিনিক্স হট 40", 12990),
        ("হেয়ার ফ্রিজ ৪ কিউফটি", 28500), ("ওয়ালটন এসি ১.৫ টন", 52000),
        ("বাজাজ মাইক্রোওয়েভ", 9500), ("প্যানাসনিক ব্লেন্ডার", 3200),
        ("সিঙ্গার এলইডি টিভি ৪৩ ইঞ্চি", 41000), ("ওয়ালটন ওয়াশিং মেশিন", 18500),
        ("লোকাল হেডফোন", 450), ("ব্লুটুথ স্পিকার", 1800),
    ]
    filtered = [(n, p) for n, p in product_pool if min_price <= p <= max_price]
    if not filtered:
        filtered = product_pool[:5]
    chosen = rng.sample(filtered, k=min(4, len(filtered)))
    return {
        "query": query,
        "results": [
            {"product_id": f"P{rng.randint(1000, 9999)}", "title": name, "price_bdt": price}
            for name, price in chosen
        ],
    }


def _get_order_status(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "order_id": args["order_id"],
        "status": rng.choice(["processing", "shipped", "out_for_delivery", "delivered"]),
        "eta_days": rng.randint(1, 5),
    }


def _request_return(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "return_initiated",
        "order_id": args["order_id"],
        "return_id": f"RET{rng.randint(10000, 99999)}",
        "reason": args["reason"],
    }


def _get_service_status(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "service_type": args["service_type"],
        "app_id": args["app_id"],
        "status": rng.choice(["received", "under_review", "approved", "ready_for_delivery"]),
    }


def _web_search(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    query = str(args.get("query", "")).lower()
    results = []

    if any(k in query for k in ["পদ্মা", "padma", "টোল", "toll"]):
        results.append({
            "title": "পদ্মা সেতু টোল হার ও নিয়মাবলি — বাংলাদেশ সেতু কর্তৃপক্ষ",
            "snippet": "পদ্মা সেতুতে কার/জিপ ৭৫০ টাকা, মাইক্রোবাস ১৩০০ টাকা, ছোট বাস ১৪০০ টাকা, বড় বাস ২৪০০ টাকা, ট্রাক (৩ এক্সেল) ৫৫০০ টাকা।",
            "source": "bba.gov.bd",
        })
    if any(k in query for k in ["মেট্রোরেল", "metro", "সময়", "সময়", "schedule", "mrt"]):
        results.append({
            "title": "ঢাকা মেট্রোরেল (MRT Line 6) সময়সূচী ও ভাড়া — DMTCL",
            "snippet": "উত্তরা উত্তর থেকে মতিঝিল পর্যন্ত শনিবার থেকে বৃহস্পতিবার সকাল ৭:১০ থেকে রাত ৯:৪০ পর্যন্ত এবং শুক্রবার বিকাল ৩:০০ থেকে রাত ৯:৪০ পর্যন্ত চলে। সর্বনিম্ন ভাড়া ২০ টাকা, সর্বোচ্চ ১০০ টাকা। এমআরটি পাসে ১০% ছাড়।",
            "source": "dmtcl.gov.bd",
        })
    if any(k in query for k in ["পাসপোর্ট", "passport", "ফি", "fee", "কাগজ"]):
        results.append({
            "title": "ই-পাসপোর্ট আবেদন ও ফি সংক্রান্ত নির্দেশিকা — বহিরাগমন ও পাসপোর্ট অধিদপ্তর",
            "snippet": "৪৮ পৃষ্ঠা ৫ বছর মেয়াদি নিয়মিত ফি ৪০২৫ টাকা (জরুরি ৬৩২৫ টাকা)। ১০ বছর মেয়াদি নিয়মিত ৫৭৫০ টাকা (জরুরি ৮০৫০ টাকা)। প্রয়োজনীয় কাগজপত্র: জাতীয় পরিচয়পত্র/জন্ম নিবন্ধন ও আবেদন সামারি।",
            "source": "epassport.gov.bd",
        })
    if any(k in query for k in ["টানেল", "tunnel", "কর্ণফুলী", "bangabandhu"]):
        results.append({
            "title": "বঙ্গবন্ধু শেখ মুজিবুর রহমান টানেল টোল তালিকা",
            "snippet": "প্রাইভেট কার ও জিপ ২০০ টাকা, পিকআপ ২০০ টাকা, মাইক্রোবাস ২৫০ টাকা। মোটরসাইকেল ও ৩ চাকার যান চলাচল নিষিদ্ধ।",
            "source": "bba.gov.bd",
        })
    if any(k in query for k in ["এক্সপ্রেসওয়ে", "expressway", "এলিভেটেড"]):
        results.append({
            "title": "ঢাকা এলিভেটেড এক্সপ্রেসওয়ে ব্যবহারের নিয়ম ও টোল",
            "snippet": "কাওলা (বিমানবন্দর) থেকে ফার্মগেট পর্যন্ত কার, ট্যাক্সি, এসইউভি ও মাইক্রোবাসের টোল ৮০ টাকা। বাস ১৬০ টাকা। মোটরসাইকেল ও সিএনজি নিষিদ্ধ।",
            "source": "rthd.gov.bd",
        })
    if any(k in query for k in ["সাজেক", "sajek", "আর্মি", "এসকর্ট", "escort"]):
        results.append({
            "title": "সাজেক ভ্যালি ভ্রমণ নির্দেশিকা ও বাঘাইহাট আর্মি এসকর্ট সময়",
            "snippet": "বাঘাইহাট থেকে সাজেক যাওয়ার নিরাপত্তা এসকর্ট প্রতিদিন সকাল ১০:০০ টা এবং বিকাল ৩:০০ টায় ছেড়ে যায়। নির্দিষ্ট সময়ের মধ্যে পৌঁছানো বাধ্যতামূলক।",
            "source": "touristpolice.gov.bd",
        })

    if not results:
        results.append({
            "title": "তথ্য ও সেবা সংক্রান্ত ফলাফল — বাংলাদেশ জাতীয় তথ্য বাতায়ন",
            "snippet": f"'{args.get('query')}' সম্পর্কিত হালনাগাদ তথ্য, নির্দেশিকা ও সরকারি নাগরিক সেবা নির্দেশিকা।",
            "source": "bangladesh.gov.bd",
        })

    return {
        "query": args.get("query"),
        "results_count": len(results),
        "results": results,
    }


def _get_weather(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    days = int(args.get("days", 3))  # type: ignore[arg-type]
    return {
        "location": args["location"],
        "forecast": [
            {
                "day": i,
                "condition": rng.choice(["sunny", "cloudy", "rain", "thunderstorm"]),
                "temp_c": rng.randint(20, 36),
            }
            for i in range(days)
        ],
    }


def _get_crop_advisory(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    crops = ["ধান", "পাট", "চা", "আলু", "পেঁয়াজ"]
    return {
        "region": args["region"],
        "crop": args.get("crop", rng.choice(crops)),
        "advisory": "পরবর্তী ৭ দিন আবহাওয়া অনুকূল। সেচ দিন।",
    }


_NEWS_HEADLINES = {
    "শীর্ষ": [
        "জাতীয় সংসদে নতুন বাজেট পাস, আয়কর সীমা বাড়ল",
        "চট্টগ্রাম বন্দরে রেকর্ড পণ্য আমদানি এই মাসে",
        "ঢাকা মেট্রোরেলের নতুন রুট উদ্বোধন আগামী মাসে",
        "শেয়ার বাজারে সূচক বেড়েছে ৮২ পয়েন্ট",
    ],
    "ক্রিকেট": [
        "বাংলাদেশ শ্রীলঙ্কাকে হারিয়ে সিরিজ জিতল ২-১",
        "তামিমের অধিনায়কত্বে সেঞ্চুরি, রেকর্ড গড়লেন",
        "টি-২০ বিশ্বকাপের জন্য দল ঘোষণা, নতুন মুখ ৩ জন",
        "মিরপুর স্টেডিয়ামে আজ ভারত-বাংলাদেশ ম্যাচ",
    ],
    "রাজনীতি": [
        "বিরোধী দল সংসদ বয়কট করার হুমকি দিয়েছে",
        "নির্বাচন কমিশন নতুন ভোটার তালিকা প্রকাশ করেছে",
        "স্থানীয় নির্বাচনে ভোট চলছে তিন জেলায়",
        "সরকারি দলের মহাসমাবেশ আগামী শুক্রবার",
    ],
    "অর্থনীতি": [
        "রেমিট্যান্স এই অর্ধবার্ষিকে রেকর্ড ১২ বিলিয়ন ডলার",
        "রপ্তানি আয় বেড়েছে গত মাসে ১১ শতাংশ",
        "ডলারের দাম কমে ১০৯ টাকা, বাজার স্থিতিশীল",
        "বিশ্বব্যাংক বাংলাদেশকে নতুন ঋণ ৭৫০ মিলিয়ন ডলার",
    ],
    "আবহাওয়া": [
        "ঘূর্ণিঝড়ের আশঙ্কায় উপকূলীয় এলাকায় সতর্কতা",
        "দেশজুড়ে তীব্র তাপমাত্রা, সর্বোচ্চ ৪১ ডিগ্রি",
        "বন্যাপরিস্থিতি উন্নতি, নদীর পানি নিচে নামছে",
        "শীত মৌসুম শুরু, উত্তর বাংলায় কুয়াশাসহ শিশির",
    ],
    "খেলা": [
        "অলিম্পিকে বাংলাদেশের শুটার ফাইনালে স্থান",
        "ফুটবল ফেডারেশন নির্বাচন আগামী মাসে",
        "বাংলাদেশ হকি দল এশিয়া কাপে সেমিফাইনালে",
        "স্কোয়াশে স্বর্ণপদক জিতলেন বাংলাদেশি খেলোয়াড়",
    ],
}


def _search_news(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    topic = str(args["topic"])
    # Match topic to headline pool, fall back to generic if no match.
    pool: list[str] = []
    for key, headlines in _NEWS_HEADLINES.items():
        if key in topic or topic in key:
            pool = headlines
            break
    if not pool:
        pool = _NEWS_HEADLINES["শীর্ষ"]
    sources = ["prothom_alo", "daily_star", "jugantor", "ittefaq", "samakal"]
    n = min(4, len(pool))
    chosen = rng.sample(pool, k=n)
    return {
        "topic": topic,
        "articles": [
            {
                "article_id": f"N{rng.randint(100, 999)}",
                "headline": chosen[i],
                "source": rng.choice(sources),
            }
            for i in range(n)
        ],
    }


def _recharge_mobile(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "success",
        "number": args["number"],
        "amount": args["amount"],
        "operator": args["operator"],
        "txid": f"RC{rng.randint(10**9, 10**10 - 1)}",
    }


def _buy_data_pack(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "success",
        "number": args["number"],
        "pack_id": args["pack_id"],
        "txid": f"DP{rng.randint(10**9, 10**10 - 1)}",
    }


# =============================================================================
# New domain simulators (Bangladesh-focused). Same determinism contract:
# seeded random.Random, embedded BD data pools, json.dumps(ensure_ascii=False).
# =============================================================================

# --- Sports scores ---
_SPORTS_MATCHES = {
    "cricket": [
        {"match_id": "BPL-2026-014", "teams": "ঢাকা ডায়নামাইটস vs কুমিল্লা ওয়ারিয়র্স", "venue": "মিরপুর"},
        {"match_id": "BPL-2026-015", "teams": "চট্টগ্রাম চ্যালেঞ্জার্স vs রাজশাহী রয়্যালস", "venue": "চট্টগ্রাম"},
        {"match_id": "INT-2026-008", "teams": "বাংলাদেশ vs ভারত", "venue": "মিরপুর"},
        {"match_id": "INT-2026-009", "teams": "বাংলাদেশ vs শ্রীলঙ্কা", "venue": "চট্টগ্রাম"},
    ],
    "football": [
        {"match_id": "BFL-2026-021", "teams": "বসুন্ধরা কিংস vs ঢাকা মোহামেডান", "venue": "বসুন্ধরা"},
        {"match_id": "BFL-2026-022", "teams": " Sheikh Jamal vs আবাহনী", "venue": "বঙ্গবন্ধু"},
        {"match_id": "INT-2026-012", "teams": "বাংলাদেশ vs থাইল্যান্ড", "venue": "বঙ্গবন্ধু"},
    ],
}

_SPORTS_PLAYERS = {
    "cricket": [
        ("তামিম ইকবাল", "ব্যাটসম্যান", {"odi_runs": 8357, "t20_runs": 1701, "test_runs": 4840}),
        ("শাকিব আল হাসান", "অলরাউন্ডার", {"odi_runs": 7563, "t20_runs": 2382, "wickets": 412}),
        ("মুশফিকুর রহিম", "উইকেটকিপার", {"odi_runs": 7743, "t20_runs": 1500, "test_runs": 5667}),
        ("মাহমুদউল্লাহ", "অলরাউন্ডার", {"odi_runs": 4944, "t20_runs": 2000, "wickets": 82}),
    ],
    "football": [
        ("জামাল ভূঞা", "মিডফিল্ডার", {"caps": 78, "goals": 12}),
        ("সাব্বির হোসেন", "ফরোয়ার্ড", {"caps": 45, "goals": 18}),
        ("বিশ্বনাথ ঘোষ", "ডিফেন্ডার", {"caps": 60, "goals": 3}),
    ],
}


def _get_live_score(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    sport = str(args["sport"])
    match_id = str(args["match_id"])
    if sport == "cricket":
        return {
            "match_id": match_id,
            "status": "live",
            "innings": f"বাংলাদেশ {rng.randint(80, 280)}/{rng.randint(1, 6)} ({rng.randint(20, 50)} ওভার)",
            "detail": "২য় ইনিংস চলছে",
        }
    return {
        "match_id": match_id,
        "status": "live",
        "score": f"{rng.randint(0, 2)}-{rng.randint(0, 2)}",
        "minute": f"{rng.randint(10, 88)}' মিনিট",
    }


def _get_match_schedule(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    sport = str(args["sport"])
    days = int(args.get("days", 7))  # type: ignore[arg-type]
    pool = _SPORTS_MATCHES.get(sport, _SPORTS_MATCHES["cricket"])
    n = min(4, max(1, days // 2))
    chosen = rng.sample(pool, k=n)
    return {
        "sport": sport,
        "matches": [
            {
                "match_id": m["match_id"],
                "teams": m["teams"],
                "venue": m["venue"],
                "time": f"{rng.randint(14, 20)}:{rng.choice(['00', '30'])}",
            }
            for m in chosen
        ],
    }


def _get_player_stats(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    name = str(args["player_name"])
    fmt = str(args.get("format", "all"))
    # Find by name (case-insensitive partial), else pick a random cricket player.
    pool = _SPORTS_PLAYERS["cricket"]
    found = next((p for p in pool if name.split()[0] in p[0]), rng.choice(pool))
    pname, role, stats = found
    if fmt != "all" and fmt in stats:
        stats = {fmt: stats[fmt]}
    return {"player": pname, "role": role, "stats": stats}


def _get_standings(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    sport = str(args["sport"])
    tournament = str(args["tournament"])
    teams = (["ঢাকা ডায়নামাইটস", "কুমিল্লা ওয়ারিয়র্স", "চট্টগ্রাম চ্যালেঞ্জার্স",
              "রাজশাহী রয়্যালস", "খুলনা টাইগার্স", "সিলেট সুপার স্টার্স"]
             if sport == "cricket"
             else ["বসুন্ধরা কিংস", "ঢাকা মোহামেডান", "শেখ জামাল", "আবাহনী",
                   "শেখ রাসেল", "বাংলাদেশ পুলিশ"])
    rng.shuffle(teams)
    return {
        "tournament": tournament,
        "table": [
            {"position": i + 1, "team": t, "played": rng.randint(8, 14),
             "points": rng.randint(8, 26)}
            for i, t in enumerate(teams[:6])
        ],
    }


# --- Food delivery ---
_RESTAURANTS = {
    ("বিরিয়ানি", "ধানমন্ডি"): [("R1001", "হাজী বিরিয়ানী", 4.5), ("R1002", "ফাহিম বিরিয়ানী", 4.2)],
    ("ফাস্টফুড", "গুলশান"): [("R2001", "টেস্টি কিচেন", 4.3), ("R2002", "বার্গার হাউস", 4.0)],
    ("চাইনিজ", "মিরপুর"): [("R3001", "ড্রাগন গার্ডেন", 4.1), ("R3002", "সিয়াম চাইনিজ", 3.9)],
}
_RESTAURANT_MENU = {
    "R1001": [("M1", "মটন বিরিয়ানী", 280), ("M2", "চিকেন বিরিয়ানী", 200)],
    "R1002": [("M3", "বিশেষ বিরিয়ানী", 320), ("M4", "চিকেন বিরিয়ানী", 180)],
    "R2001": [("M5", "বার্গার কম্বো", 350), ("M6", "ফ্রাইড চিকেন", 250)],
    "R2002": [("M7", "ডবল চিজ বার্গার", 400), ("M8", "চিকেন নাগেটস", 180)],
    "R3001": [("M9", "চিকেন ফ্রাইড রাইস", 220), ("M10", "মটন চাওমিন", 280)],
    "R3002": [("M11", "চিলি চিকেন", 300), ("M12", "ভেজ মোমো", 150)],
}


def _search_restaurants(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    cuisine = str(args.get("cuisine", "বিরিয়ানি"))
    area = str(args.get("area", "ধানমন্ডি"))
    pool = _RESTAURANTS.get((cuisine, area))
    if not pool:
        pool = [
            (f"R{rng.randint(4000, 4999)}", f"{area} রেস্তোরাঁ {i}", rng.uniform(3.5, 4.5))
            for i in range(3)
        ]
    return {
        "cuisine": cuisine,
        "area": area,
        "restaurants": [
            {"restaurant_id": r[0], "name": r[1], "rating": r[2]} for r in pool
        ],
    }


def _get_menu(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    rid = str(args["restaurant_id"])
    menu = _RESTAURANT_MENU.get(rid, [("MX1", "বিশেষ থালা", rng.randint(150, 400))])
    return {"restaurant_id": rid, "menu": [
        {"item_id": m[0], "name": m[1], "price_bdt": m[2]} for m in menu
    ]}


def _place_order(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    items = args.get("items", [])  # type: ignore[assignment]
    total = sum(int(i.get("quantity", 1)) * rng.randint(150, 350) for i in items)  # type: ignore[union-attr]
    return {
        "status": "confirmed",
        "order_id": f"FD{rng.randint(100000, 999999)}",
        "restaurant_id": args["restaurant_id"],
        "total_bdt": total,
        "eta_minutes": rng.randint(25, 55),
    }


def _track_delivery(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "order_id": args["order_id"],
        "status": rng.choice(["preparing", "on_the_way", "nearby", "delivered"]),
        "eta_minutes": rng.randint(5, 35),
    }


# --- Travel planning ---
_DESTINATIONS = [
    ("D01", "কক্সবাজার", "সমুদ্রতীর", 120),
    ("D02", "সেন্টমার্টিন", "সমুদ্রতীর", 180),
    ("D03", "সুন্দরবন", "বন", 150),
    ("D04", "বান্দরবান", "পাহাড়", 100),
    ("D05", "রাঙামাটি", "পাহাড়", 90),
    ("D06", "সিলেট-জাফলং", "পাহাড়", 80),
    ("D07", "কুয়াকাটা", "সমুদ্রতীর", 110),
    ("D08", "মৌলভীবাজার-মালনীছড়া", "পাহাড়", 75),
]


def _search_destinations(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    region = str(args.get("region", ""))
    budget = int(args.get("budget_bdt", 999999))  # type: ignore[arg-type]
    pool = [d for d in _DESTINATIONS if (not region or region in d[2]) and d[3] * 1000 <= budget]
    if not pool:
        pool = _DESTINATIONS[:4]
    chosen = rng.sample(pool, k=min(4, len(pool)))
    return {"destinations": [
        {"destination_id": d[0], "name": d[1], "type": d[2], "estimated_cost_bdt": d[3] * 1000}
        for d in chosen
    ]}


def _get_packages(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    dest = str(args["destination"])
    duration = int(args.get("duration_days", 3))  # type: ignore[arg-type]
    return {
        "destination": dest,
        "packages": [
            {
                "package_id": f"PKG-{rng.randint(100, 999)}",
                "duration_days": duration,
                "price_bdt": rng.randint(8000, 35000),
                "includes": ["থাকা", "খাওয়া", "গাড়ি", "গাইড"],
            }
            for _ in range(rng.randint(2, 3))
        ],
    }


def _check_hotel_availability(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    avail = rng.random() > 0.15  # 85% available, 15% unavailable
    hotels = [
        {"hotel_id": f"H{rng.randint(1000, 9999)}",
         "name": f"{args['destination']} রিসোর্ট",
         "price_per_night_bdt": rng.randint(2500, 12000)}
        for _ in range(rng.randint(2, 3))
    ] if avail else []
    return {
        "destination": args["destination"],
        "check_in": args["check_in"],
        "check_out": args["check_out"],
        "available": avail,
        "hotels": hotels,
    }


def _book_tour(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "confirmed",
        "booking_id": f"TB{rng.randint(100000, 999999)}",
        "package_id": args["package_id"],
        "traveler_name": args["traveler_name"],
        "travelers": args["travelers"],
    }


def _book_hotel(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "confirmed",
        "booking_id": f"HTL-{rng.randint(100000, 999999)}",
        "hotel_id": args["hotel_id"],
        "guest_name": args["guest_name"],
        "check_in": args["check_in"],
        "check_out": args["check_out"],
        "room_type": args.get("room_type", "deluxe"),
        "total_bdt": rng.randint(2500, 15000),
    }


# --- Education ---
_UNIVERSITIES = [
    ("DU", "ঢাকা বিশ্ববিদ্যালয়", "ঢাকা", "সব বিষয়"),
    ("BUET", "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়", "ঢাকা", "প্রকৌশল"),
    ("JU", "জাহাঙ্গীরনগর বিশ্ববিদ্যালয়", "সাভার", "সব বিষয়"),
    ("RU", "রাজশাহী বিশ্ববিদ্যালয়", "রাজশাহী", "সব বিষয়"),
    ("CU", "চট্টগ্রাম বিশ্পবিদ্যালয়", "চট্টগ্রাম", "সব বিষয়"),
    ("SUST", "শাহজালাল বিজ্ঞান ও প্রযুক্তি বিশ্ববিদ্যালয়", "সিলেট", "বিজ্ঞান"),
]


def _search_universities(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    subject = str(args.get("subject", ""))
    city = str(args.get("city", ""))
    pool = [u for u in _UNIVERSITIES
            if (not subject or subject in u[3]) and (not city or city in u[2])]
    if not pool:
        pool = _UNIVERSITIES[:3]
    chosen = rng.sample(pool, k=min(4, len(pool)))
    return {"universities": [
        {"university_id": u[0], "name": u[1], "city": u[2], "subjects": u[3]} for u in chosen
    ]}


def _get_admission_info(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    uid = str(args["university_id"])
    found = next((u for u in _UNIVERSITIES if u[0] == uid), _UNIVERSITIES[0])
    return {
        "university_id": uid,
        "name": found[1],
        "circular": "২০২৬-২৭ শিক্ষাবর্ষের ভর্তি বিজ্ঞপ্তি প্রকাশিত",
        "deadline": "২০২৬-১০-৩১",
        "min_gpa": rng.choice([3.5, 4.0, 4.5]),
    }


def _check_result(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    exam = str(args["exam"])
    roll = str(args["roll"])
    year = int(args.get("year", 2026))  # type: ignore[arg-type]
    return {
        "exam": exam.upper(),
        "roll": roll,
        "year": year,
        "gpa": round(rng.uniform(3.0, 5.0), 2),
        "status": rng.choice(["passed", "passed", "passed", "failed"]),
    }


def _get_scholarship_list(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    level = str(args.get("level", "undergraduate"))
    scholarships = [
        ("S1", "বৃত্তি অনুদান কমিশন", level, rng.randint(2000, 8000)),
        ("S2", "বিশ্ববিদ্যালয় মেধা বৃত্তি", level, rng.randint(1500, 6000)),
        ("S3", "প্রতিবন্ধী বৃত্তি", level, rng.randint(1000, 4000)),
        ("S4", "বিদেশে উচ্চশিক্ষা বৃত্তি", level, rng.randint(5000, 15000)),
    ]
    return {"level": level, "scholarships": [
        {"scholarship_id": s[0], "name": s[1], "amount_bdt_month": s[3]} for s in scholarships
    ]}


# --- Bus + launch booking ---
def _search_bus_routes(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "from": args["from"], "to": args["to"], "date": args["date"],
        "buses": [
            {"route_id": f"BUS-{rng.randint(100, 999)}",
             "company": rng.choice(["গ্রীনলাইন", "শ্যামলী", "সোহাগ", "হানিফ"]),
             "departure": f"{rng.randint(6, 22):02d}:{rng.choice(['00', '30'])}",
             "type": rng.choice(["এসি", "নন-এসি"])}
            for _ in range(rng.randint(2, 4))
        ],
    }


def _search_launch_routes(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "from": args["from"], "to": args["to"], "date": args["date"],
        "launches": [
            {"route_id": f"LNCH-{rng.randint(100, 999)}",
             "company": rng.choice(["সুন্দরবন", "কুমিল্লা এক্সপ্রেস", "প্রিন্স অফ সাউথ"]),
             "departure": f"{rng.randint(18, 22):02d}:{rng.choice(['00', '30'])} (রাত্রি)"}
            for _ in range(rng.randint(2, 3))
        ],
    }


def _get_transport_fare(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    ttype = str(args["transport_type"])
    base_fare = {"bus": {"ac": 1200, "non_ac": 700},
                 "launch": {"deck": 350, "cabin": 1800, "vip": 3000}}
    seat_class = str(args["seat_class"])
    fare = base_fare.get(ttype, {}).get(seat_class, rng.randint(400, 2000))
    fare += rng.randint(-50, 50)
    return {
        "transport_type": ttype, "route_id": args["route_id"],
        "seat_class": seat_class, "fare_bdt": fare,
    }


def _book_transport_ticket(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "confirmed",
        "pnr": f"TX-{rng.randint(100000, 999999)}",
        "transport_type": args["transport_type"],
        "route_id": args["route_id"],
        "passenger_name": args["passenger_name"],
        "seat_count": args["seat_count"],
    }


# --- Flight booking ---
def _search_flights(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "from": args["from"], "to": args["to"], "date": args["date"],
        "flights": [
            {"flight_id": f"{rng.choice(['BG', 'BS', 'VQ'])}{rng.randint(100, 999)}",
             "airline": rng.choice(["বিমান", "US-Bangla", "নোভোএয়ার"]),
             "departure": f"{rng.randint(6, 22):02d}:{rng.choice(['00', '15', '30', '45'])}"}
            for _ in range(rng.randint(2, 4))
        ],
    }


def _get_flight_fare(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    cabin = str(args["cabin_class"])
    base = 4500 if cabin == "economy" else 9000
    return {
        "flight_id": args["flight_id"], "cabin_class": cabin,
        "fare_bdt": base + rng.randint(0, 2000),
    }


def _book_flight(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "confirmed",
        "pnr": f"FL-{rng.randint(100000, 999999)}",
        "flight_id": args["flight_id"],
        "cabin_class": args["cabin_class"],
        "passenger_name": args["passenger_name"],
        "seat_count": args["seat_count"],
    }


# --- Real estate ---
def _search_properties(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    ltype = str(args["listing_type"])
    ptype = str(args.get("property_type", "ফ্ল্যাট"))
    area = str(args.get("area", "উত্তরা"))
    budget = int(args.get("max_budget_bdt", 999999))  # type: ignore[arg-type]
    base_rent = rng.randint(15000, 60000) if ltype == "rent" else rng.randint(6000000, 20000000)
    return {
        "listing_type": ltype, "area": area,
        "properties": [
            {"property_id": f"PROP-{rng.randint(1000, 9999)}",
             "type": ptype, "area": area,
             "price_bdt": min(base_rent + rng.randint(-5000, 5000), budget),
             "bedrooms": rng.randint(2, 5)}
            for _ in range(rng.randint(2, 4))
        ],
    }


def _get_property_details(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "property_id": args["property_id"],
        "address": f"{rng.choice(['উত্তরা', 'বনানী', 'ধানমন্ডি', 'গুলশান'])}, ঢাকা",
        "area_sqft": rng.randint(900, 2500),
        "bedrooms": rng.randint(2, 5),
        "bathrooms": rng.randint(2, 4),
        "price_bdt": rng.randint(20000, 80000),
    }


def _schedule_visit(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "scheduled",
        "visit_id": f"VIS-{rng.randint(1000, 9999)}",
        "property_id": args["property_id"],
        "date": args["date"],
        "visitor_name": args["visitor_name"],
    }


# --- Job search ---
_JOBS_POOL = [
    ("JOB-1001", "জুনিয়র সফটওয়্যার ইঞ্জিনিয়ার", "ঢাকা", "টেক সলিউশনস লিমিটেড", "full_time"),
    ("JOB-1002", "ডিজিটাল মার্কেটিং এক্সিকিউটিভ", "চট্টগ্রাম", "মিডিয়া হাব", "full_time"),
    ("JOB-1003", "অ্যাকাউন্টেন্ট", "ঢাকা", "এফসি অ্যাসোসিয়েটস", "full_time"),
    ("JOB-1004", "গ্রাফিক ডিজাইনার ইন্টার্ন", "সিলেট", "ক্রিয়েটিভ ল্যাব", "internship"),
    ("JOB-1005", "সেলস রিপ্রেজেন্টেটিভ", "রাজশাহী", "কনজ্যুমার গুডস", "full_time"),
    ("JOB-1006", "পার্ট টাইম টিউটর", "ঢাকা", "শিক্ষা নেটওয়ার্ক", "part_time"),
]


def _search_jobs(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    query = str(args["query"])
    location = str(args.get("location", ""))
    jtype = str(args.get("job_type", ""))
    pool = [
        j for j in _JOBS_POOL
        if (query.lower() in j[1].lower() or not query)
        and (not location or location in j[2])
        and (not jtype or jtype == j[4])
    ]
    if not pool:
        pool = _JOBS_POOL[:3]
    chosen = rng.sample(pool, k=min(4, len(pool)))
    return {"jobs": [
        {"job_id": j[0], "title": j[1], "location": j[2], "company": j[3], "type": j[4]}
        for j in chosen
    ]}


def _get_job_details(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    jid = str(args["job_id"])
    found = next((j for j in _JOBS_POOL if j[0] == jid), _JOBS_POOL[0])
    return {
        "job_id": jid, "title": found[1], "location": found[2], "company": found[3],
        "salary_bdt": rng.randint(20000, 80000),
        "requirements": "স্নাতক ডিগ্রি, ১-২ বছর অভিজ্ঞতা",
        "deadline": "২০২৬-০৮-১৫",
    }


def _apply_job(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "status": "applied",
        "application_id": f"APP-{rng.randint(10000, 99999)}",
        "job_id": args["job_id"],
        "applicant_name": args["applicant_name"],
        "email": args["email"],
    }


def _get_company_info(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    cid = str(args["company_id"])
    return {
        "company_id": cid,
        "name": rng.choice(["টেক সলিউশনস লিমিটেড", "মিডিয়া হাব", "এফসি অ্যাসোসিয়েটস"]),
        "industry": rng.choice(["আইটি", "মিডিয়া", "অর্থ", "কনজ্যুমার"]),
        "size": f"{rng.randint(50, 500)} কর্মী",
        "founded": rng.randint(2005, 2020),
    }


# --- Movie / cinema ---
_MOVIES = [
    ("MV1", "প্রিয়তমা", "bangla"),
    ("MV2", "সুরাইয়া", "bangla"),
    ("MV3", "ঢাকা অ্যাটাক", "bangla"),
    ("MV4", "অভিযান", "bangla"),
]


def _search_movies(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    lang = str(args.get("language", "bangla"))
    city = str(args.get("city", "ঢাকা"))
    pool = [m for m in _MOVIES if m[2] == lang] if lang else _MOVIES
    if not pool:
        pool = _MOVIES
    chosen = rng.sample(pool, k=min(3, len(pool)))
    return {"city": city, "movies": [
        {"movie_id": m[0], "title": m[1], "language": m[2]} for m in chosen
    ]}


def _get_showtimes(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    mid = str(args["movie_id"])
    date = str(args.get("date", ""))
    return {
        "movie_id": mid, "date": date,
        "showtimes": [
            {"showtime_id": f"ST-{rng.randint(1000, 9999)}",
             "cinema": rng.choice(["বলিউড মাল্টিপ্লেক্স", "স্টার সিনেপ্লেক্স", "ব্লকবাস্টার"]),
             "time": f"{rng.randint(10, 22):02d}:{rng.choice(['00', '15', '30', '45'])}",
             "available_seats": rng.randint(10, 80)}
            for _ in range(rng.randint(2, 4))
        ],
    }


def _book_cinema_tickets(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    seats = int(args["seats"])  # type: ignore[arg-type]
    return {
        "status": "confirmed",
        "booking_id": f"CIN-{rng.randint(100000, 999999)}",
        "showtime_id": args["showtime_id"],
        "seats": seats,
        "total_bdt": seats * rng.randint(300, 600),
    }


# --- City transport ---
_RIDE_BASE = {"car": 80, "bike": 35, "cng": 50, "rickshaw": 30}


def _estimate_ride(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    vehicle = str(args["vehicle"])
    base = _RIDE_BASE.get(vehicle, 50)
    distance_km = rng.randint(3, 18)
    return {
        "from": args["from"], "to": args["to"], "vehicle": vehicle,
        "distance_km": distance_km,
        "fare_bdt": base * distance_km + rng.randint(0, 30),
        "eta_minutes": rng.randint(5, 25),
    }


def _book_ride(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    vehicle = str(args["vehicle"])
    base = _RIDE_BASE.get(vehicle, 50)
    distance = rng.randint(3, 18)
    return {
        "status": "searching",
        "ride_id": f"RD-{rng.randint(100000, 999999)}",
        "from": args["from"], "to": args["to"], "vehicle": vehicle,
        "fare_bdt": base * distance + rng.randint(0, 30),
        "eta_minutes": rng.randint(3, 12),
    }


def _get_ride_fare(args: dict[str, object], rng: random.Random) -> dict[str, object]:
    return {
        "ride_id": args["ride_id"],
        "fare_bdt": rng.randint(80, 600),
        "distance_km": rng.randint(3, 18),
        "duration_minutes": rng.randint(10, 45),
    }


_Simulator = Callable[[dict[str, object], random.Random], dict[str, object]]

_SIMULATORS: dict[str, _Simulator] = {
    "search_trains": _search_trains,
    "get_train_fare": _get_train_fare,
    "book_ticket": _book_ticket,
    "send_money": _send_money,
    "check_balance": _check_balance,
    "pay_bill": _pay_bill,
    "search_doctors": _search_doctors,
    "get_available_slots": _get_available_slots,
    "book_appointment": _book_appointment,
    "search_products": _search_products,
    "get_order_status": _get_order_status,
    "request_return": _request_return,
    "get_service_status": _get_service_status,
    "web_search": _web_search,
    "get_weather": _get_weather,
    "get_crop_advisory": _get_crop_advisory,
    "search_news": _search_news,
    "recharge_mobile": _recharge_mobile,
    "buy_data_pack": _buy_data_pack,
    # New domains
    "get_live_score": _get_live_score,
    "get_match_schedule": _get_match_schedule,
    "get_player_stats": _get_player_stats,
    "get_standings": _get_standings,
    "search_restaurants": _search_restaurants,
    "get_menu": _get_menu,
    "place_order": _place_order,
    "track_delivery": _track_delivery,
    "search_destinations": _search_destinations,
    "get_packages": _get_packages,
    "check_hotel_availability": _check_hotel_availability,
    "book_tour": _book_tour,
    "book_hotel": _book_hotel,
    "search_universities": _search_universities,
    "get_admission_info": _get_admission_info,
    "check_result": _check_result,
    "get_scholarship_list": _get_scholarship_list,
    "search_bus_routes": _search_bus_routes,
    "search_launch_routes": _search_launch_routes,
    "get_transport_fare": _get_transport_fare,
    "book_transport_ticket": _book_transport_ticket,
    "search_flights": _search_flights,
    "get_flight_fare": _get_flight_fare,
    "book_flight": _book_flight,
    "search_properties": _search_properties,
    "get_property_details": _get_property_details,
    "schedule_visit": _schedule_visit,
    "search_jobs": _search_jobs,
    "get_job_details": _get_job_details,
    "apply_job": _apply_job,
    "get_company_info": _get_company_info,
    "search_movies": _search_movies,
    "get_showtimes": _get_showtimes,
    "book_cinema_tickets": _book_cinema_tickets,
    "estimate_ride": _estimate_ride,
    "book_ride": _book_ride,
    "get_ride_fare": _get_ride_fare,
}


def simulate(name: str, args: dict[str, object], seed: int) -> str:
    """Execute a tool deterministically. Returns a JSON string.

    ensure_ascii=False so Bangla values render as Bangla (not \\u09xx escapes);
    sort_keys=True so byte-equality holds regardless of insertion order.
    """
    if name not in _SIMULATORS:
        raise KeyError(f"Unknown tool: {name}")
    result = _SIMULATORS[name](args, _rng(seed))
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
