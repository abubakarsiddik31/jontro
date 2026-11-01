"""Tool definitions across 8 domains. Names are English; values may be Bangla."""

from bangla_datasets.schema import ToolDef


def _train_search_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "Departure station, e.g. ঢাকা"},
            "to": {"type": "string", "description": "Arrival station, e.g. চট্টগ্রাম"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["from", "to", "date"],
    }


# --- Railway ---
SEARCH_TRAINS = ToolDef(
    name="search_trains",
    description="Search available trains between two stations on a date.",
    parameters_json_schema=_train_search_schema(),
)
GET_TRAIN_FARE = ToolDef(
    name="get_train_fare",
    description="Get fare for a train by class.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "train_id": {"type": "string"},
            "seat_class": {
                "type": "string",
                "enum": ["shovan", "shovan_chair", "snigdha", "ac_b", "ac_s"],
            },
        },
        "required": ["train_id", "seat_class"],
    },
)
BOOK_TICKET = ToolDef(
    name="book_ticket",
    description="Book a train ticket.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "train_id": {"type": "string"},
            "seat_class": {"type": "string"},
            "passenger_name": {"type": "string"},
            "seat_count": {"type": "integer"},
        },
        "required": ["train_id", "seat_class", "passenger_name", "seat_count"],
    },
)

# --- Mobile finance ---
SEND_MONEY = ToolDef(
    name="send_money",
    description="Send money to a recipient.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient mobile number"},
            "amount": {"type": "number"},
            "currency": {"type": "string", "enum": ["BDT"]},
        },
        "required": ["to", "amount"],
    },
)
CHECK_BALANCE = ToolDef(
    name="check_balance",
    description="Check account balance.",
    parameters_json_schema={"type": "object", "properties": {}, "required": []},
)
PAY_BILL = ToolDef(
    name="pay_bill",
    description="Pay a utility or service bill.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "biller": {
                "type": "string",
                "enum": ["electricity", "gas", "water", "internet", "tv"],
            },
            "account_no": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["biller", "account_no", "amount"],
    },
)

# --- Healthcare ---
SEARCH_DOCTORS = ToolDef(
    name="search_doctors",
    description="Search doctors by specialty and location.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "specialty": {"type": "string"},
            "location": {"type": "string"},
        },
        "required": ["specialty"],
    },
)
GET_AVAILABLE_SLOTS = ToolDef(
    name="get_available_slots",
    description="Get available appointment slots for a doctor.",
    parameters_json_schema={
        "type": "object",
        "properties": {"doctor_id": {"type": "string"}, "date": {"type": "string"}},
        "required": ["doctor_id"],
    },
)
BOOK_APPOINTMENT = ToolDef(
    name="book_appointment",
    description="Book an appointment.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "doctor_id": {"type": "string"},
            "slot_id": {"type": "string"},
            "patient_name": {"type": "string"},
        },
        "required": ["doctor_id", "slot_id", "patient_name"],
    },
)

# --- E-commerce ---
SEARCH_PRODUCTS = ToolDef(
    name="search_products",
    description="Search products by query with optional price range.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "min_price": {"type": "number", "description": "Minimum price in BDT"},
            "max_price": {"type": "number", "description": "Maximum price in BDT"},
        },
        "required": ["query"],
    },
)
GET_ORDER_STATUS = ToolDef(
    name="get_order_status",
    description="Get status of an order.",
    parameters_json_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
REQUEST_RETURN = ToolDef(
    name="request_return",
    description="Request a return for an order item.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["order_id", "reason"],
    },
)

# --- Govt services ---
GET_SERVICE_STATUS = ToolDef(
    name="get_service_status",
    description="Check status of a government service application.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "service_type": {
                "type": "string",
                "enum": ["passport", "nid", "birth_certificate"],
            },
            "app_id": {"type": "string"},
        },
        "required": ["service_type", "app_id"],
    },
)
WEB_SEARCH = ToolDef(
    name="web_search",
    description="Search the web for public information, government rules, toll rates, procedures, schedules, or general knowledge in Bangladesh when no specialized tool exists.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query in Bangla or English, e.g. 'পদ্মা সেতু টোল' or 'metro rail friday schedule'"},
        },
        "required": ["query"],
    },
)

# --- Agri + weather ---
GET_WEATHER = ToolDef(
    name="get_weather",
    description="Get weather forecast for a location.",
    parameters_json_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["location"],
    },
)
GET_CROP_ADVISORY = ToolDef(
    name="get_crop_advisory",
    description="Get crop advisory for a region.",
    parameters_json_schema={
        "type": "object",
        "properties": {"region": {"type": "string"}, "crop": {"type": "string"}},
        "required": ["region"],
    },
)

# --- News ---
SEARCH_NEWS = ToolDef(
    name="search_news",
    description="Search news articles.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "source": {"type": "string"},
            "date_range": {"type": "string", "enum": ["today", "week", "month"]},
        },
        "required": ["topic"],
    },
)

# --- Telecom ---
RECHARGE_MOBILE = ToolDef(
    name="recharge_mobile",
    description="Recharge a mobile number.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "number": {"type": "string"},
            "amount": {"type": "number"},
            "operator": {
                "type": "string",
                "enum": ["grameenphone", "robi", "banglalink", "teletalk"],
            },
        },
        "required": ["number", "amount", "operator"],
    },
)
BUY_DATA_PACK = ToolDef(
    name="buy_data_pack",
    description="Purchase a data pack for a number.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "number": {"type": "string"},
            "pack_id": {
                "type": "string",
                "enum": ["1gb_3day", "5gb_7day", "15gb_30day", "50gb_30day"],
            },
        },
        "required": ["number", "pack_id"],
    },
)

# =============================================================================
# New domains (Bangladesh-focused). Tool names/keys stay ASCII; values Bangla.
# =============================================================================

# --- Sports scores (cricket + football) ---
GET_LIVE_SCORE = ToolDef(
    name="get_live_score",
    description="Get the live score of an ongoing cricket or football match.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sport": {"type": "string", "enum": ["cricket", "football"]},
            "match_id": {"type": "string", "description": "Match identifier, e.g. BPL-2026-014"},
        },
        "required": ["sport", "match_id"],
    },
)
GET_MATCH_SCHEDULE = ToolDef(
    name="get_match_schedule",
    description="Get upcoming match schedule for a sport (BPL, national team, football league).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sport": {"type": "string", "enum": ["cricket", "football"]},
            "days": {"type": "integer", "description": "Number of days ahead to look"},
        },
        "required": ["sport"],
    },
)
GET_PLAYER_STATS = ToolDef(
    name="get_player_stats",
    description="Get career statistics for a Bangladeshi player.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "player_name": {"type": "string", "description": "e.g. তামিম ইকবাল"},
            "format": {"type": "string", "enum": ["odi", "t20", "test", "all"]},
        },
        "required": ["player_name"],
    },
)
GET_STANDINGS = ToolDef(
    name="get_standings",
    description="Get league/tournament standings table.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "sport": {"type": "string", "enum": ["cricket", "football"]},
            "tournament": {"type": "string", "description": "e.g. BPL, bangladesh_football_league"},
        },
        "required": ["sport", "tournament"],
    },
)

# --- Food delivery ---
SEARCH_RESTAURANTS = ToolDef(
    name="search_restaurants",
    description="Search restaurants by cuisine and area (Foodpanda/Pathao Food style).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "cuisine": {"type": "string", "description": "e.g. বিরিয়ানি, ফাস্টফুড, চাইনিজ"},
            "area": {"type": "string", "description": "e.g. ধানমন্ডি, গুলশান, মিরপুর"},
        },
        "required": ["area"],
    },
)
GET_MENU = ToolDef(
    name="get_menu",
    description="Get the menu of a restaurant.",
    parameters_json_schema={
        "type": "object",
        "properties": {"restaurant_id": {"type": "string"}},
        "required": ["restaurant_id"],
    },
)
PLACE_ORDER = ToolDef(
    name="place_order",
    description="Place a food order.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "restaurant_id": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["item_id", "quantity"],
                },
            },
            "address": {"type": "string"},
        },
        "required": ["restaurant_id", "items", "address"],
    },
)
TRACK_DELIVERY = ToolDef(
    name="track_delivery",
    description="Track the status of a food delivery order.",
    parameters_json_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)

# --- Travel planning ---
SEARCH_DESTINATIONS = ToolDef(
    name="search_destinations",
    description="Search tourist destinations in Bangladesh (Cox's Bazar, সেন্টমার্টিন, সুন্দরবন, etc.).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "e.g. সমুদ্রতীর, পাহাড়, বন"},
            "budget_bdt": {"type": "number"},
        },
        "required": [],
    },
)
GET_PACKAGES = ToolDef(
    name="get_packages",
    description="Get tour packages for a destination.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "duration_days": {"type": "integer"},
        },
        "required": ["destination"],
    },
)
CHECK_HOTEL_AVAILABILITY = ToolDef(
    name="check_hotel_availability",
    description="Check hotel availability for dates.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "check_in": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "check_out": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "guests": {"type": "integer"},
        },
        "required": ["destination", "check_in", "check_out"],
    },
)
BOOK_TOUR = ToolDef(
    name="book_tour",
    description="Book a tour package.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "package_id": {"type": "string"},
            "traveler_name": {"type": "string"},
            "travelers": {"type": "integer"},
        },
        "required": ["package_id", "traveler_name", "travelers"],
    },
)
BOOK_HOTEL = ToolDef(
    name="book_hotel",
    description="Book a hotel room in a destination.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "hotel_id": {"type": "string", "description": "e.g. H1744"},
            "guest_name": {"type": "string"},
            "check_in": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "check_out": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "room_type": {"type": "string", "enum": ["single", "deluxe", "suite"]},
        },
        "required": ["hotel_id", "guest_name", "check_in", "check_out"],
    },
)

# --- Education ---
SEARCH_UNIVERSITIES = ToolDef(
    name="search_universities",
    description="Search universities in Bangladesh (ঢাকা বিশ্ববিদ্যালয়, BUET, etc.).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "e.g. প্রকৌশল, চিকিৎসা, ব্যবসা"},
            "city": {"type": "string"},
        },
        "required": [],
    },
)
GET_ADMISSION_INFO = ToolDef(
    name="get_admission_info",
    description="Get admission circular and requirements for a university.",
    parameters_json_schema={
        "type": "object",
        "properties": {"university_id": {"type": "string"}},
        "required": ["university_id"],
    },
)
CHECK_RESULT = ToolDef(
    name="check_result",
    description="Check an exam result (SSC, HSC, university admission).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "exam": {"type": "string", "enum": ["ssc", "hsc", "admission", "semester"]},
            "roll": {"type": "string"},
            "year": {"type": "integer"},
        },
        "required": ["exam", "roll"],
    },
)
GET_SCHOLARSHIP_LIST = ToolDef(
    name="get_scholarship_list",
    description="List available scholarships for Bangladeshi students.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["hsc", "undergraduate", "postgraduate"]},
        },
        "required": [],
    },
)

# --- Bus + launch booking ---
SEARCH_BUS_ROUTES = ToolDef(
    name="search_bus_routes",
    description="Search intercity bus routes (ঢাকা-চট্টগ্রাম, ঢাকা-সিলেট, etc.).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["from", "to", "date"],
    },
)
SEARCH_LAUNCH_ROUTES = ToolDef(
    name="search_launch_routes",
    description="Search launch/ferry routes (ঢাকা-বরিশাল, ঢাকা-পটুয়াখালী, etc.).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["from", "to", "date"],
    },
)
GET_TRANSPORT_FARE = ToolDef(
    name="get_transport_fare",
    description="Get fare for a bus or launch route by cabin/deck class.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "transport_type": {"type": "string", "enum": ["bus", "launch"]},
            "route_id": {"type": "string"},
            "seat_class": {"type": "string"},
        },
        "required": ["transport_type", "route_id", "seat_class"],
    },
)
BOOK_TRANSPORT_TICKET = ToolDef(
    name="book_transport_ticket",
    description="Book a bus or launch ticket.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "transport_type": {"type": "string", "enum": ["bus", "launch"]},
            "route_id": {"type": "string"},
            "seat_class": {"type": "string"},
            "passenger_name": {"type": "string"},
            "seat_count": {"type": "integer"},
        },
        "required": ["transport_type", "route_id", "seat_class", "passenger_name", "seat_count"],
    },
)

# --- Flight booking ---
SEARCH_FLIGHTS = ToolDef(
    name="search_flights",
    description="Search domestic flights (বিমান, US-Bangla, নোভোএয়ার).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "IATA-style, e.g. ঢাকা"},
            "to": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["from", "to", "date"],
    },
)
GET_FLIGHT_FARE = ToolDef(
    name="get_flight_fare",
    description="Get fare for a flight.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "flight_id": {"type": "string"},
            "cabin_class": {"type": "string", "enum": ["economy", "business"]},
        },
        "required": ["flight_id", "cabin_class"],
    },
)
BOOK_FLIGHT = ToolDef(
    name="book_flight",
    description="Book a flight ticket.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "flight_id": {"type": "string"},
            "cabin_class": {"type": "string"},
            "passenger_name": {"type": "string"},
            "seat_count": {"type": "integer"},
        },
        "required": ["flight_id", "cabin_class", "passenger_name", "seat_count"],
    },
)

# --- Real estate ---
SEARCH_PROPERTIES = ToolDef(
    name="search_properties",
    description="Search properties for rent or buy (ভাড়া/ক্রয়).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "listing_type": {"type": "string", "enum": ["rent", "buy"]},
            "property_type": {"type": "string", "description": "e.g. ফ্ল্যাট, বাড়ি"},
            "area": {"type": "string", "description": "e.g. উত্তরা, বনানী, ধানমন্ডি"},
            "max_budget_bdt": {"type": "number"},
        },
        "required": ["listing_type"],
    },
)
GET_PROPERTY_DETAILS = ToolDef(
    name="get_property_details",
    description="Get full details of a property listing.",
    parameters_json_schema={
        "type": "object",
        "properties": {"property_id": {"type": "string"}},
        "required": ["property_id"],
    },
)
SCHEDULE_VISIT = ToolDef(
    name="schedule_visit",
    description="Schedule a visit to a property.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "property_id": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "visitor_name": {"type": "string"},
        },
        "required": ["property_id", "date", "visitor_name"],
    },
)

# --- Job search ---
SEARCH_JOBS = ToolDef(
    name="search_jobs",
    description="Search jobs (BDJobs-style) by title and location.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "location": {"type": "string"},
            "job_type": {"type": "string", "enum": ["full_time", "part_time", "internship"]},
        },
        "required": ["query"],
    },
)
GET_JOB_DETAILS = ToolDef(
    name="get_job_details",
    description="Get full details of a job posting.",
    parameters_json_schema={
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
)
APPLY_JOB = ToolDef(
    name="apply_job",
    description="Apply to a job posting.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "applicant_name": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["job_id", "applicant_name", "email"],
    },
)
GET_COMPANY_INFO = ToolDef(
    name="get_company_info",
    description="Get information about a hiring company.",
    parameters_json_schema={
        "type": "object",
        "properties": {"company_id": {"type": "string"}},
        "required": ["company_id"],
    },
)

# --- Movie / cinema ---
SEARCH_MOVIES = ToolDef(
    name="search_movies",
    description="Search currently showing movies in cinemas.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": ["bangla", "hindi", "english"]},
            "city": {"type": "string"},
        },
        "required": [],
    },
)
GET_SHOWTIMES = ToolDef(
    name="get_showtimes",
    description="Get showtimes for a movie at cinemas.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "movie_id": {"type": "string"},
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["movie_id"],
    },
)
BOOK_CINEMA_TICKETS = ToolDef(
    name="book_cinema_tickets",
    description="Book cinema tickets.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "showtime_id": {"type": "string"},
            "seats": {"type": "integer"},
            "viewer_name": {"type": "string"},
        },
        "required": ["showtime_id", "seats", "viewer_name"],
    },
)

# --- City transport ---
ESTIMATE_RIDE = ToolDef(
    name="estimate_ride",
    description="Estimate ride fare and ETA (Pathao/Uber/CNG/rickshaw).",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "Pickup area, e.g. গুলশান"},
            "to": {"type": "string", "description": "Drop-off area, e.g. ধানমন্ডি"},
            "vehicle": {"type": "string", "enum": ["car", "bike", "cng", "rickshaw"]},
        },
        "required": ["from", "to", "vehicle"],
    },
)
BOOK_RIDE = ToolDef(
    name="book_ride",
    description="Book a ride.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "vehicle": {"type": "string", "enum": ["car", "bike", "cng", "rickshaw"]},
        },
        "required": ["from", "to", "vehicle"],
    },
)
GET_RIDE_FARE = ToolDef(
    name="get_ride_fare",
    description="Get fare details for a booked ride.",
    parameters_json_schema={
        "type": "object",
        "properties": {"ride_id": {"type": "string"}},
        "required": ["ride_id"],
    },
)


DOMAIN_TOOLS: dict[str, list[ToolDef]] = {
    "railway_booking": [SEARCH_TRAINS, GET_TRAIN_FARE, BOOK_TICKET],
    "mobile_finance": [SEND_MONEY, CHECK_BALANCE, PAY_BILL],
    "healthcare": [SEARCH_DOCTORS, GET_AVAILABLE_SLOTS, BOOK_APPOINTMENT],
    "ecommerce": [SEARCH_PRODUCTS, GET_ORDER_STATUS, REQUEST_RETURN],
    "govt_services": [GET_SERVICE_STATUS, WEB_SEARCH],
    "agri_weather": [GET_WEATHER, GET_CROP_ADVISORY],
    "news": [SEARCH_NEWS],
    "telecom": [RECHARGE_MOBILE, BUY_DATA_PACK],
    # New Bangladesh-focused domains
    "sports_scores": [GET_LIVE_SCORE, GET_MATCH_SCHEDULE, GET_PLAYER_STATS, GET_STANDINGS],
    "food_delivery": [SEARCH_RESTAURANTS, GET_MENU, PLACE_ORDER, TRACK_DELIVERY],
    "travel_planning": [SEARCH_DESTINATIONS, GET_PACKAGES, CHECK_HOTEL_AVAILABILITY, BOOK_TOUR, BOOK_HOTEL],
    "education": [SEARCH_UNIVERSITIES, GET_ADMISSION_INFO, CHECK_RESULT, GET_SCHOLARSHIP_LIST],
    "bus_launch_booking": [
        SEARCH_BUS_ROUTES, SEARCH_LAUNCH_ROUTES, GET_TRANSPORT_FARE, BOOK_TRANSPORT_TICKET,
    ],
    "flight_booking": [SEARCH_FLIGHTS, GET_FLIGHT_FARE, BOOK_FLIGHT],
    "real_estate": [SEARCH_PROPERTIES, GET_PROPERTY_DETAILS, SCHEDULE_VISIT],
    "job_search": [SEARCH_JOBS, GET_JOB_DETAILS, APPLY_JOB, GET_COMPANY_INFO],
    "movie_cinema": [SEARCH_MOVIES, GET_SHOWTIMES, BOOK_CINEMA_TICKETS],
    "city_transport": [ESTIMATE_RIDE, BOOK_RIDE, GET_RIDE_FARE],
}

ALL_TOOLS: list[ToolDef] = [t for ts in DOMAIN_TOOLS.values() for t in ts]
