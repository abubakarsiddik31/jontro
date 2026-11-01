"""Multi-tool, multi-turn scenario blueprints across 18 Bangladesh domains.

Contains 60 distinct scenario blueprints covering:
  - Transport (train, bus, launch, flight, ride-hailing)
  - Healthcare & Doctor Consultations
  - Mobile Finance (bKash/Nagad P2P, utilities)
  - Food Delivery & Restaurant Ordering
  - Telecom & Data Packs
  - E-Commerce & Retail Returns
  - Cinema & Entertainment
  - Real Estate & Housing Visits
  - Jobs, Universities & Education
  - Agriculture, Crops & Weather Advisories
  - Web Search & Citizen Public Info (Padma bridge tolls, Metro rail, e-passport)
  - Real-World Friction (slot negotiation, missing parameters, sold out alternatives)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from bangla_datasets.schema import Persona, ToolDef
from bangla_datasets.tools.catalog import (
    APPLY_JOB,
    BOOK_APPOINTMENT,
    BOOK_CINEMA_TICKETS,
    BOOK_FLIGHT,
    BOOK_HOTEL,
    BOOK_RIDE,
    BOOK_TICKET,
    BOOK_TOUR,
    BOOK_TRANSPORT_TICKET,
    BUY_DATA_PACK,
    CHECK_BALANCE,
    CHECK_HOTEL_AVAILABILITY,
    CHECK_RESULT,
    ESTIMATE_RIDE,
    GET_ADMISSION_INFO,
    GET_AVAILABLE_SLOTS,
    GET_COMPANY_INFO,
    GET_CROP_ADVISORY,
    GET_FLIGHT_FARE,
    GET_JOB_DETAILS,
    GET_LIVE_SCORE,
    GET_MATCH_SCHEDULE,
    GET_MENU,
    GET_ORDER_STATUS,
    GET_PACKAGES,
    GET_PLAYER_STATS,
    GET_PROPERTY_DETAILS,
    GET_RIDE_FARE,
    GET_SCHOLARSHIP_LIST,
    GET_SERVICE_STATUS,
    GET_SHOWTIMES,
    GET_STANDINGS,
    GET_TRAIN_FARE,
    GET_TRANSPORT_FARE,
    GET_WEATHER,
    PAY_BILL,
    PLACE_ORDER,
    RECHARGE_MOBILE,
    REQUEST_RETURN,
    SCHEDULE_VISIT,
    SEARCH_BUS_ROUTES,
    SEARCH_DESTINATIONS,
    SEARCH_DOCTORS,
    SEARCH_FLIGHTS,
    SEARCH_JOBS,
    SEARCH_LAUNCH_ROUTES,
    SEARCH_MOVIES,
    SEARCH_NEWS,
    SEARCH_PRODUCTS,
    SEARCH_PROPERTIES,
    SEARCH_RESTAURANTS,
    SEARCH_TRAINS,
    SEARCH_UNIVERSITIES,
    SEND_MONEY,
    TRACK_DELIVERY,
    WEB_SEARCH,
)


@dataclass
class ScenarioBlueprint:
    name: str
    primary_domain: str
    tools: list[ToolDef]
    goals_bn: list[str]
    goals_banglish: list[str]
    min_turns: int = 2
    max_turns: int = 4


SCENARIO_BLUEPRINTS: list[ScenarioBlueprint] = [
    # =========================================================================
    # 1. TRANSPORT & TRAVEL (1-12)
    # =========================================================================
    ScenarioBlueprint(
        name="railway_intercity_booking",
        primary_domain="railway_booking",
        tools=[SEARCH_TRAINS, GET_TRAIN_FARE, BOOK_TICKET],
        goals_bn=[
            "ঢাকা থেকে চট্টগ্রাম যাওয়ার ট্রেনের সময়সূচী এবং ভাড়া জেনে পছন্দের ট্রেনের স্নিগ্ধা টিকিট বুক করা।",
            "রাজশাহী থেকে ঢাকা আগামী সপ্তাহের ট্রেনের শিডিউল দেখা, শোভন চেয়ারের ভাড়া জানা এবং ১টি টিকিট কাটা।",
            "ঢাকা থেকে সিলেট ট্রেনের টিকিট বুক করা, আগে ট্রেনের সময় ও ভাড়া যাচাই করা।",
        ],
        goals_banglish=[
            "Dhaka theke Chittagong train schedule o fare jene Snigdha ticket book kora.",
            "Rajshahi theke Dhaka trainer shomoy o bhara dekhe ekta ticket book korte hobe.",
        ],
    ),
    ScenarioBlueprint(
        name="railway_stay_payment",
        primary_domain="railway_booking",
        tools=[SEARCH_TRAINS, BOOK_TICKET, CHECK_HOTEL_AVAILABILITY, BOOK_HOTEL, PAY_BILL],
        goals_bn=[
            "ঢাকা থেকে চট্টগ্রাম ট্রেনের টিকিট বুক করা এবং আগ্রাবাদ এলাকায় হোটেল রুম বুকিং ও পেমেন্ট সম্পন্ন করা।",
            "সিলেট যাওয়ার ট্রেনের টিকিট কেটে দরগাহ রোডের কাছে হোটেলের সন্ধান করা ও রুম বুকিং করা।",
        ],
        goals_banglish=[
            "Dhaka theke Chittagong train ticket book kora ebong hotel room check kore booking confirm kora.",
        ],
    ),
    ScenarioBlueprint(
        name="bus_intercity_ac_booking",
        primary_domain="bus_launch_booking",
        tools=[SEARCH_BUS_ROUTES, GET_TRANSPORT_FARE, BOOK_TRANSPORT_TICKET],
        goals_bn=[
            "ঢাকা থেকে রংপুরগামী হানিফ বা শ্যামলী বাসের এসি রুট ও ভাড়া জেনে ১টি টিকিট বুক করা।",
            "ঢাকা থেকে বগুড়া যাওয়ার বিকেলের এসি বাসের ভাড়া দেখে টিকিট নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Dhaka theke Rangpur Hanif ba Shyamoli AC bus-er schedule o fare dekhe ticket book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="launch_cabin_booking",
        primary_domain="bus_launch_booking",
        tools=[SEARCH_LAUNCH_ROUTES, GET_TRANSPORT_FARE, BOOK_TRANSPORT_TICKET],
        goals_bn=[
            "ঢাকা সদরঘাট থেকে বরিশাল যাওয়ার লঞ্চের তালিকা ও কেবিনের ভাড়া জেনে সিঙ্গেল কেবিন বুক করা।",
            "ঢাকা থেকে ভোলা যাওয়ার সুন্দরবন বা প্রিন্স অব সাউথ লঞ্চের ডাবল কেবিন টিকিট বুকিং করা।",
        ],
        goals_banglish=[
            "Dhaka theke Barisal launch-er cabin bhara jene ticket confirm kora.",
        ],
    ),
    ScenarioBlueprint(
        name="flight_domestic_booking",
        primary_domain="flight_booking",
        tools=[SEARCH_FLIGHTS, GET_FLIGHT_FARE, BOOK_FLIGHT, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "ঢাকা থেকে কক্সবাজারের ফ্লাইটের সময় ও ভাড়া দেখে টিকিট বুক করা এবং শাহজালাল বিমানবন্দরে যাওয়ার রাইড ডাকা।",
            "চট্টগ্রাম থেকে ঢাকা ফ্লাইটের টিকিট কেটে গন্তব্যে পৌঁছানোর জন্য রাইড নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Dhaka theke Cox's Bazar flight ticket book kora ebong airport jaoar ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="flight_return_schedule",
        primary_domain="flight_booking",
        tools=[SEARCH_FLIGHTS, GET_FLIGHT_FARE, BOOK_FLIGHT],
        goals_bn=[
            "সিলেট থেকে ঢাকা ফেরত আসার সকালের ফ্লাইটের সময়সূচী এবং বিমান বাংলাদেশ বা ইউএস-বাংলার ভাড়া দেখে টিকিট বুক করা।",
            "কক্সবাজার থেকে ঢাকা সান্ধ্যকালীন ফ্লাইটের ভাড়া দেখে ১টি আসন নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Sylhet theke Dhaka return flight-er shomoy o bhara dekhe ticket book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="city_ride_hailing",
        primary_domain="city_transport",
        tools=[ESTIMATE_RIDE, BOOK_RIDE, GET_RIDE_FARE],
        goals_bn=[
            "ধানমন্ডি থেকে মতিঝিল যাওয়ার জন্য বাইক বা কার রাইডের ভাড়া অনুমান করে দ্রুত রাইড বুক করা।",
            "উত্তরা থেকে গুলশান-২ যাওয়ার জন্য গাড়ির ভাড়া জেনে রাইড নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Dhanmondi theke Motijheel jaoar jonno bike ba car ride estimate dekhe book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="vacation_tour_package",
        primary_domain="travel_planning",
        tools=[SEARCH_DESTINATIONS, GET_PACKAGES, CHECK_HOTEL_AVAILABILITY, BOOK_TOUR],
        goals_bn=[
            "সাজেখ ভ্যালি বা শ্রীমঙ্গলের ৩ দিনের ট্যুর প্যাকেজ খোঁজা, হোটেল সুবিধা দেখে প্যাকেজ বুক করা।",
            "কক্সবাজারের কাপল ট্যুর প্যাকেজ যাচাই করে ভ্রমণ প্যাকেজ নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Sajek ba Sreemangal-er 3 diner tour package dekhe booking confirm kora.",
        ],
    ),
    ScenarioBlueprint(
        name="hotel_booking_and_amenities",
        primary_domain="travel_planning",
        tools=[CHECK_HOTEL_AVAILABILITY, BOOK_HOTEL],
        goals_bn=[
            "কক্সবাজার সমুদ্র সৈকতের কাছে ভালো হোটেলের খালি রুমের তালিকা দেখে ডিলাক্স রুম বুক করা।",
            "সিলেটে থাকার জন্য ৩ তারকা হোটেলের রুমের ভাড়া জেনে ২ রাতের জন্য রুম নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Cox's Bazar hotel-er faka room o vara dekhe deluxe room book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="multimodal_trip_planner",
        primary_domain="flight_booking",
        tools=[SEARCH_FLIGHTS, BOOK_FLIGHT, CHECK_HOTEL_AVAILABILITY, BOOK_HOTEL, BOOK_RIDE],
        goals_bn=[
            "ঢাকা থেকে চট্টগ্রাম ফ্লাইট বুক করা, জিইসি মোড়ে হোটেল বুক করা এবং বিমানবন্দর থেকে হোটেলে যাওয়ার গাড়ি ঠিক করা।",
        ],
        goals_banglish=[
            "Dhaka theke Chittagong flight book kora, hotel room confirm kora ebong ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="bus_return_journey",
        primary_domain="bus_launch_booking",
        tools=[SEARCH_BUS_ROUTES, GET_TRANSPORT_FARE, BOOK_TRANSPORT_TICKET],
        goals_bn=[
            "রাজশাহী থেকে ঢাকা ফেরত আসার রাতের এসি বাসের ভাড়া জেনে টিকিট বুক করা।",
            "খুলনা থেকে ঢাকা ফেরার জন্য শ্যামলী বা গ্রীনলাইনের আসন সংরক্ষণ করা।",
        ],
        goals_banglish=[
            "Rajshahi theke Dhaka return AC bus-er ticket bhara jene book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="train_round_trip",
        primary_domain="railway_booking",
        tools=[SEARCH_TRAINS, GET_TRAIN_FARE, BOOK_TICKET],
        goals_bn=[
            "ঢাকা থেকে চট্টগ্রাম গিয়ে পরশুদিন ফিরে আসার ট্রেনের টিকিট ও ভাড়া যাচাই করে কনফার্ম করা।",
        ],
        goals_banglish=[
            "Dhaka to Chittagong round trip train ticket o fare dekhe booking kora.",
        ],
    ),

    # =========================================================================
    # 2. HEALTHCARE & MEDICAL SERVICES (13-18)
    # =========================================================================
    ScenarioBlueprint(
        name="doctor_specialist_appointment",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, GET_AVAILABLE_SLOTS, BOOK_APPOINTMENT],
        goals_bn=[
            "ঢাকায় একজন হৃদরোগ বা মেডিসিন বিশেষজ্ঞ ডাক্তারের সন্ধান করা, ফাঁকা সময় দেখে অ্যাপয়েন্টমেন্ট বুক করা।",
            "চট্টগ্রামে শিশু বিশেষজ্ঞ ডাক্তার খুঁজে নিয়ে উপযুক্ত তারিখে অ্যাপয়েন্টমেন্ট নেওয়া।",
        ],
        goals_banglish=[
            "Dhaka te specialist doctor khuje tar faka slot dekhe appointment book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="doctor_slot_negotiation",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, GET_AVAILABLE_SLOTS, BOOK_APPOINTMENT],
        goals_bn=[
            "ঢাকায় বিশেষজ্ঞ ডাক্তারের অ্যাপয়েন্টমেন্ট নেওয়া, নিজের পছন্দের সময় না পেয়ে সহকারীর প্রস্তাবিত খালি স্লটে সিরিয়াল নেওয়া।",
        ],
        goals_banglish=[
            "Doctor appointment chaoya, pochonder slot faka na thakle alternative slot book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="healthcare_ambulance_ride",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, BOOK_APPOINTMENT, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "জরুরি চিকিৎসার জন্য ডাক্তারের অ্যাপয়েন্টমেন্ট নেওয়া এবং হাসপাতালে দ্রুত যাওয়ার জন্য গাড়ি বুক করা।",
        ],
        goals_banglish=[
            "Doctor appointment book kora ebong hospital jaoar ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="doctor_second_opinion_search",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, GET_AVAILABLE_SLOTS, BOOK_APPOINTMENT],
        goals_bn=[
            "চর্মরোগ বিশেষজ্ঞ ডাঃ রহিমা এবং ডাঃ করিমের ফাঁকা সময় তুলনা করে সুবিধাজনক স্লট বুক করা।",
        ],
        goals_banglish=[
            "Dujon doctor er available time compare kore bhalo slot-e serial book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="telemedicine_telecom_bundle",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, BOOK_APPOINTMENT, RECHARGE_MOBILE],
        goals_bn=[
            "ডাক্তারের পরামর্শের জন্য অ্যাপয়েন্টমেন্ট নেওয়া এবং টেলিমেডিসিন এসএমএস ও কলের জন্য ফোনে ১০০ টাকা রিচার্জ করা।",
        ],
        goals_banglish=[
            "Doctor appointment book kora ebong mobile-e 100 taka recharge kora.",
        ],
    ),
    ScenarioBlueprint(
        name="eye_specialist_consultation",
        primary_domain="healthcare",
        tools=[SEARCH_DOCTORS, GET_AVAILABLE_SLOTS, BOOK_APPOINTMENT],
        goals_bn=[
            "চোখের সমস্যার জন্য চক্ষু বিশেষজ্ঞ খুঁজে আগামীকালের সকালের স্লটে অ্যাপয়েন্টমেন্ট নিশ্চিত করা।",
        ],
        goals_banglish=[
            "Eye specialist doctor khuje sokaler slot-e appointment book kora.",
        ],
    ),

    # =========================================================================
    # 3. FINANCIAL SERVICES & UTILITIES (19-25)
    # =========================================================================
    ScenarioBlueprint(
        name="mfs_p2p_transfer",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, SEND_MONEY],
        goals_bn=[
            "বিকাশ/নগদ ব্যালেন্স দেখে বন্ধুর নম্বরে ২০০০ টাকা পাঠানো এবং ট্রানজেকশন আইডি নিশ্চিত করা।",
            "অ্যাকাউন্টে পর্যাপ্ত ব্যালেন্স আছে কিনা দেখে পরিবারের সদস্যকে ১০০০ টাকা পাঠানো।",
        ],
        goals_banglish=[
            "Balance check kore bondhur nombore taka send kora ebong TxID confirm kora.",
        ],
    ),
    ScenarioBlueprint(
        name="mfs_missing_slots",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, SEND_MONEY, PAY_BILL],
        goals_bn=[
            "টাকা পাঠানো দরকার কিন্তু শুরুতে তথ্য অসম্পূর্ণ রাখা; সহকারী নম্বর ও পরিমাণ চাইলে ০১৭০০১২৩৪৫৬ নম্বরে ৫০০ টাকা পাঠানো।",
        ],
        goals_banglish=[
            "Taka send korte chai kintu details na bola, assistant chaile number o amount provide kora.",
        ],
    ),
    ScenarioBlueprint(
        name="utility_electricity_bill",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, PAY_BILL],
        goals_bn=[
            "ডেসকো/ডিপিডিসি বিদ্যুৎ বিলের একাউন্ট নম্বর দিয়ে ১২০০ টাকা বিল পরিশোধ করা ও রসিদ নেওয়া।",
        ],
        goals_banglish=[
            "Electricity bill account number diye 1200 taka pay kora ebong receipt number collect kora.",
        ],
    ),
    ScenarioBlueprint(
        name="utility_missing_account",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, PAY_BILL],
        goals_bn=[
            "গ্যাস বিল দিতে চাওয়া, শুরুতে একাউন্ট নম্বর না বলা; সহকারী চাইলে ৬৬৭৭৮৮ এবং ৬৫০ টাকা দিয়ে পরিশোধ করা।",
        ],
        goals_banglish=[
            "Gas bill dite chaoa, assistant account number chaile provide kore bill pay kora.",
        ],
    ),
    ScenarioBlueprint(
        name="utility_gas_water_bill",
        primary_domain="mobile_finance",
        tools=[PAY_BILL],
        goals_bn=[
            "তিতাস গ্যাস বিল ৮৫০ টাকা এবং ঢাকা ওয়াসা পানির বিল ৪০০ টাকা পরিশোধ করে রসিদ সংরক্ষণ করা।",
        ],
        goals_banglish=[
            "Titas gas bill ebong WASA water bill porishodh kora.",
        ],
    ),
    ScenarioBlueprint(
        name="utility_broadband_payment",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, PAY_BILL],
        goals_bn=[
            "ব্যালেন্স যাচাই করে মাসিক ব্রডব্যান্ড ইন্টারনেট বিল ১০০০ টাকা পরিশোধ করা।",
        ],
        goals_banglish=[
            "Balance check kore broadband internet bill pay kora.",
        ],
    ),
    ScenarioBlueprint(
        name="mfs_multi_bill_settlement",
        primary_domain="mobile_finance",
        tools=[CHECK_BALANCE, PAY_BILL, SEND_MONEY],
        goals_bn=[
            "ব্যালেন্স দেখে বিদ্যুৎ বিল পরিশোধ করা এবং অতিরিক্ত ৫০০ টাকা ভাইকে পাঠানো।",
        ],
        goals_banglish=[
            "Electricity bill pay kore baki taka bhai-er nombore send kora.",
        ],
    ),

    # =========================================================================
    # 4. FOOD DELIVERY & DINING (26-30)
    # =========================================================================
    ScenarioBlueprint(
        name="food_kacchi_biryani_order",
        primary_domain="food_delivery",
        tools=[SEARCH_RESTAURANTS, GET_MENU, PLACE_ORDER, TRACK_DELIVERY],
        goals_bn=[
            "ধানমন্ডি এলাকার রেস্তোরাঁ থেকে হাজী বিরিয়ানির মেন্যু দেখে ২ প্লেট কাচ্চি অর্ডার করা এবং ডেলিভারি ট্র্যাক করা।",
        ],
        goals_banglish=[
            "Dhanmondi restaurant theke kacchi biryani order kora ebong delivery track kora.",
        ],
    ),
    ScenarioBlueprint(
        name="food_fastfood_delivery",
        primary_domain="food_delivery",
        tools=[SEARCH_RESTAURANTS, GET_MENU, PLACE_ORDER, TRACK_DELIVERY],
        goals_bn=[
            "গুলশান এলাকার রেস্তোরাঁ থেকে বার্গার ও ফ্রাইস অর্ডার করা এবং পৌঁছানোর সময় জানা।",
        ],
        goals_banglish=[
            "Gulshan theke burger order kora ebong delivery time track kora.",
        ],
    ),
    ScenarioBlueprint(
        name="food_order_change_constraint",
        primary_domain="food_delivery",
        tools=[SEARCH_RESTAURANTS, GET_MENU, PLACE_ORDER, TRACK_DELIVERY],
        goals_bn=[
            "রেস্তোরাঁর মেন্যু দেখে প্রথমে বার্গার নিতে চেয়ে পরে সিদ্ধান্ত বদলে কাচ্চি বিরিয়ানি অর্ডার করা।",
        ],
        goals_banglish=[
            "Burger order korte cheye pore decision change kore biryani order kora.",
        ],
    ),
    ScenarioBlueprint(
        name="food_dinner_and_ride",
        primary_domain="food_delivery",
        tools=[SEARCH_RESTAURANTS, GET_MENU, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "রেস্তোরাঁর মেন্যু দেখে ডিনার করার জন্য রেস্টুরেন্টে সরাসরি যাওয়ার গাড়ি বুক করা।",
        ],
        goals_banglish=[
            "Restaurant menu dekhe sekhane dinner-e jaoar jonno ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="food_track_existing_order",
        primary_domain="food_delivery",
        tools=[TRACK_DELIVERY],
        goals_bn=[
            "পূর্বের অর্ডারের ডেলিভারি স্ট্যাটাস এবং রাইডারের অবস্থান ট্র্যাক করা।",
        ],
        goals_banglish=[
            "Ager food order-er delivery status track kora.",
        ],
    ),

    # =========================================================================
    # 5. TELECOM & CONNECTIVITY (31-35)
    # =========================================================================
    ScenarioBlueprint(
        name="telecom_recharge_and_pack",
        primary_domain="telecom",
        tools=[RECHARGE_MOBILE, BUY_DATA_PACK, CHECK_BALANCE],
        goals_bn=[
            "গ্রামীণফোন বা রবি নম্বরে টাকা রিচার্জ করা এবং ৩০ দিনের ৫০ জিবি ইন্টারনেট প্যাক চালু করা।",
        ],
        goals_banglish=[
            "Grameenphone ba Robi nombore taka recharge kore 30 diner data pack activate kora.",
        ],
    ),
    ScenarioBlueprint(
        name="telecom_emergency_recharge",
        primary_domain="telecom",
        tools=[CHECK_BALANCE, RECHARGE_MOBILE],
        goals_bn=[
            "মোবাইল ব্যালেন্স চেক করে জরুরি প্রয়োজনে ১০০ টাকা রিচার্জ করা।",
        ],
        goals_banglish=[
            "Mobile balance check kore 100 taka recharge kora.",
        ],
    ),
    ScenarioBlueprint(
        name="telecom_short_term_pack",
        primary_domain="telecom",
        tools=[BUY_DATA_PACK, CHECK_BALANCE],
        goals_bn=[
            "অনলাইন ক্লাসের জন্য ৭ দিনের ৫ জিবি ইন্টারনেট প্যাক কেনা এবং ব্যালেন্স যাচাই করা।",
        ],
        goals_banglish=[
            "Online class-er jonno 7 diner 5GB internet pack kena.",
        ],
    ),
    ScenarioBlueprint(
        name="telecom_operator_bundle",
        primary_domain="telecom",
        tools=[RECHARGE_MOBILE, BUY_DATA_PACK],
        goals_bn=[
            "বাংলালিংক নম্বরে ৩০০ টাকা রিচার্জ করে ১৫ জিবির মাসিক প্যাক কেনা।",
        ],
        goals_banglish=[
            "Banglalink nombore 300 taka recharge kore 15GB pack kena.",
        ],
    ),
    ScenarioBlueprint(
        name="telecom_balance_check_only",
        primary_domain="telecom",
        tools=[CHECK_BALANCE],
        goals_bn=[
            "সিম অ্যাকাউন্টে কত টাকা ব্যালেন্স অবশিষ্ট আছে তা যাচাই করা।",
        ],
        goals_banglish=[
            "Mobile SIM balance koto ache check kora.",
        ],
    ),

    # =========================================================================
    # 6. E-COMMERCE & SHOPPING (36-40)
    # =========================================================================
    ScenarioBlueprint(
        name="ecommerce_gadget_search",
        primary_domain="ecommerce",
        tools=[SEARCH_PRODUCTS, GET_ORDER_STATUS],
        goals_bn=[
            "স্মার্টওয়াচ বা হেডফোন খুঁজে নিয়ে দাম ও স্পেসিফিকেশন দেখা এবং পূর্বের অর্ডারের স্ট্যাটাস জানা।",
        ],
        goals_banglish=[
            "Smartwatch ba headphone search kore dam dekha ebong order status check kora.",
        ],
    ),
    ScenarioBlueprint(
        name="ecommerce_order_tracking",
        primary_domain="ecommerce",
        tools=[GET_ORDER_STATUS],
        goals_bn=[
            "অনলাইন অর্ডারের (ORD-9912345) বর্তমান ডেলিভারি অবস্থা এবং আনুমানিক পৌঁছানোর সময় জানা।",
        ],
        goals_banglish=[
            "Online order ORD-9912345 er delivery status check kora.",
        ],
    ),
    ScenarioBlueprint(
        name="ecommerce_product_return",
        primary_domain="ecommerce",
        tools=[GET_ORDER_STATUS, REQUEST_RETURN],
        goals_bn=[
            "ত্রুটিপূর্ণ বা ক্ষতিগ্রস্ত পণ্যের অর্ডারের জন্য রিটার্ন রিকোয়েস্ট তৈরি করা।",
        ],
        goals_banglish=[
            "Khoti-grosto product-er jonno return request create kora.",
        ],
    ),
    ScenarioBlueprint(
        name="ecommerce_appliance_search",
        primary_domain="ecommerce",
        tools=[SEARCH_PRODUCTS],
        goals_bn=[
            "ওয়ালটন বা স্যামসাংয়ের ওয়াশিং মেশিন ও ফ্রিজের মডেল ও দাম যাচাই করা।",
        ],
        goals_banglish=[
            "Walton ba Samsung washing machine search kore dam dekha.",
        ],
    ),
    ScenarioBlueprint(
        name="ecommerce_order_and_pay",
        primary_domain="ecommerce",
        tools=[SEARCH_PRODUCTS, SEND_MONEY],
        goals_bn=[
            "পণ্য অনুসন্ধান করে দাম জেনে মার্চেন্ট নম্বরে বিকাশ পেমেন্ট সম্পন্ন করা।",
        ],
        goals_banglish=[
            "Product search kore merchant nombore bKash payment kora.",
        ],
    ),

    # =========================================================================
    # 7. CINEMA & ENTERTAINMENT (41-44)
    # =========================================================================
    ScenarioBlueprint(
        name="cinema_blockbuster_tickets",
        primary_domain="movie_cinema",
        tools=[SEARCH_MOVIES, GET_SHOWTIMES, BOOK_CINEMA_TICKETS],
        goals_bn=[
            "স্টার সিনেপ্লেক্সে আজকের নতুন সিনেমার শো-টাইম দেখে ২টা টিকিট বুক করা।",
        ],
        goals_banglish=[
            "Star Cineplex-e notun movie-r showtime dekhe 2 ta ticket book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="cinema_movie_and_ride",
        primary_domain="movie_cinema",
        tools=[SEARCH_MOVIES, GET_SHOWTIMES, BOOK_CINEMA_TICKETS, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "সিনেপ্লেক্সের টিকিট কাটা এবং সিনেপ্লেক্সে যাওয়ার জন্য রাইড ডাকা।",
        ],
        goals_banglish=[
            "Movie ticket book kora ebong sekhane jaoar ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="cinema_sold_out_negotiation",
        primary_domain="movie_cinema",
        tools=[SEARCH_MOVIES, GET_SHOWTIMES, BOOK_CINEMA_TICKETS],
        goals_bn=[
            "সন্ধ্যার শো-এর টিকিট না পেয়ে বিকেলের শো-তে টিকিট কনফার্ম করা।",
        ],
        goals_banglish=[
            "Evening show faka na thakle bikel-er show-te ticket book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="cinema_weekend_planning",
        primary_domain="movie_cinema",
        tools=[SEARCH_MOVIES, GET_SHOWTIMES],
        goals_bn=[
            "শুক্রবারের বসুন্ধরা সিটি ও সীমান্ত সম্ভারের সিনেমা তালিকা ও আসন ফাঁকা আছে কিনা দেখা।",
        ],
        goals_banglish=[
            "Friday-te movie showtime o available seat check kora.",
        ],
    ),

    # =========================================================================
    # 8. REAL ESTATE & HOUSING (45-48)
    # =========================================================================
    ScenarioBlueprint(
        name="real_estate_flat_rent",
        primary_domain="real_estate",
        tools=[SEARCH_PROPERTIES, GET_PROPERTY_DETAILS, SCHEDULE_VISIT],
        goals_bn=[
            "উত্তরা বা ধানমন্ডিতে ৩ রুমের ফ্ল্যাটের খোঁজ করা, বিবরণ দেখে পরিদর্শনের সময় নির্ধারণ করা।",
        ],
        goals_banglish=[
            "Uttara ba Dhanmondi-te 3 bed flat khuje visit schedule kora.",
        ],
    ),
    ScenarioBlueprint(
        name="real_estate_commercial_office",
        primary_domain="real_estate",
        tools=[SEARCH_PROPERTIES, GET_PROPERTY_DETAILS, SCHEDULE_VISIT],
        goals_bn=[
            "গুলশান বা বনানীতে বাণিজ্যিক অফিস স্পেসের ভাড়া জেনে ভিজিট শিডিউল করা।",
        ],
        goals_banglish=[
            "Gulshan ba Banani-te commercial office space dekhe visit schedule kora.",
        ],
    ),
    ScenarioBlueprint(
        name="real_estate_buy_flat",
        primary_domain="real_estate",
        tools=[SEARCH_PROPERTIES, GET_PROPERTY_DETAILS],
        goals_bn=[
            "বসুন্ধরা আবাসিক এলাকায় ক্রয়ের জন্য ফ্ল্যাটের মূল্য ও আয়তন জানা।",
        ],
        goals_banglish=[
            "Bashundhara area-te flat kroy korar jonno details check kora.",
        ],
    ),
    ScenarioBlueprint(
        name="real_estate_bachelor_mess",
        primary_domain="real_estate",
        tools=[SEARCH_PROPERTIES, SCHEDULE_VISIT],
        goals_bn=[
            "মিরপুর এলাকায় ব্যাচেলর বা ছোট পরিবারের জন্য ২ রুমের ফ্ল্যাটের খোঁজ করা।",
        ],
        goals_banglish=[
            "Mirpur-e 2 bed flat khuje dekhar shomoy fix kora.",
        ],
    ),

    # =========================================================================
    # 9. JOBS, EDUCATION & CAREER (49-52)
    # =========================================================================
    ScenarioBlueprint(
        name="job_software_engineer_apply",
        primary_domain="job_search",
        tools=[SEARCH_JOBS, GET_JOB_DETAILS, APPLY_JOB, GET_COMPANY_INFO],
        goals_bn=[
            "সফটওয়্যার ইঞ্জিনিয়ার বা ওয়েব ডেভেলপার পদের চাকরি খুঁজে অনলাইনে আবেদন করা।",
        ],
        goals_banglish=[
            "Software engineer job circular khuje online-e apply kora.",
        ],
    ),
    ScenarioBlueprint(
        name="job_bank_circular_apply",
        primary_domain="job_search",
        tools=[SEARCH_JOBS, GET_JOB_DETAILS, APPLY_JOB],
        goals_bn=[
            "ব্যাংক বা বহুজাতিক কোম্পানির চাকরির সার্কুলার দেখে আবেদন প্রক্রিয়া সম্পন্ন করা।",
        ],
        goals_banglish=[
            "Bank job circular dekhe qualifications jene apply kora.",
        ],
    ),
    ScenarioBlueprint(
        name="education_university_admission",
        primary_domain="education",
        tools=[SEARCH_UNIVERSITIES, GET_ADMISSION_INFO],
        goals_bn=[
            "বুয়েট বা ঢাকা বিশ্ববিদ্যালয়ের ভর্তি পরীক্ষার যোগ্যতা ও আসন সংখ্যা জানা।",
        ],
        goals_banglish=[
            "BUET ba DU admission requirements o eligibility jante chaoa.",
        ],
    ),
    ScenarioBlueprint(
        name="education_scholarship_search",
        primary_domain="education",
        tools=[SEARCH_UNIVERSITIES, GET_SCHOLARSHIP_LIST, CHECK_RESULT],
        goals_bn=[
            "বিশ্ববিদ্যালয়ের মেধাভিত্তিক স্কলারশিপের তালিকা দেখা এবং ফলাফলের জিপিএ যাচাই করা।",
        ],
        goals_banglish=[
            "University scholarship list dekha ebong result GPA check kora.",
        ],
    ),

    # =========================================================================
    # 10. AGRICULTURE, CROPS & WEATHER (53-55)
    # =========================================================================
    ScenarioBlueprint(
        name="agri_weather_crop_advisory",
        primary_domain="agri_weather",
        tools=[GET_WEATHER, GET_CROP_ADVISORY],
        goals_bn=[
            "রাজশাহী বা রংপুর অঞ্চলে আগামী ৭ দিনের আবহাওয়া জেনে ধান বা আলুর রোগবালাই প্রতিরোধের কৃষি পরামর্শ নেওয়া।",
        ],
        goals_banglish=[
            "Rajshahi ba Rangpur elakar weather dekhe crop advisory neoa.",
        ],
    ),
    ScenarioBlueprint(
        name="agri_flood_cyclone_alert",
        primary_domain="agri_weather",
        tools=[GET_WEATHER, GET_CROP_ADVISORY, SEARCH_NEWS],
        goals_bn=[
            "উপকূলীয় অঞ্চলে ঘূর্ণিঝড় বা বৃষ্টির পূর্বাভাস জেনে ফসলের সুরক্ষা পরামর্শ ও সংবাদ দেখা।",
        ],
        goals_banglish=[
            "Cyclone/rain forecast dekhe foshol rokkhar poramorsho o news check kora.",
        ],
    ),
    ScenarioBlueprint(
        name="agri_subsidy_status",
        primary_domain="agri_weather",
        tools=[GET_WEATHER, GET_SERVICE_STATUS],
        goals_bn=[
            "আবহাওয়ার পূর্বাভাস দেখার পাশাপাশি সরকারি কৃষি সহায়তা আবেদনের অবস্থা জানা।",
        ],
        goals_banglish=[
            "Weather check kora ebong krishi subsidy status check kora.",
        ],
    ),

    # =========================================================================
    # 11. HYBRID WEB SEARCH & CITIZEN KNOWLEDGE (56-60)
    # =========================================================================
    ScenarioBlueprint(
        name="hybrid_metro_schedule_ride",
        primary_domain="city_transport",
        tools=[WEB_SEARCH, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "ঢাকা মেট্রোরেলের শুক্রবারে চলাচলের সময়সূচী ওয়েব সার্চে জানা এবং বাসা থেকে মেট্রোরেল স্টেশনে যাওয়ার রাইড বুক করা।",
        ],
        goals_banglish=[
            "Metro rail friday schedule web search kore jene station jaoar ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="hybrid_padma_toll_and_ride",
        primary_domain="city_transport",
        tools=[WEB_SEARCH, ESTIMATE_RIDE, BOOK_RIDE],
        goals_bn=[
            "পদ্মা সেতু বা এলিভেটেড এক্সপ্রেসওয়ের টোল রেট ও নিয়মাবলি ওয়েব সার্চে জেনে নিয়ে সেখানে যাওয়ার গাড়ি ভাড়া অনুমান করে রাইড বুক করা।",
        ],
        goals_banglish=[
            "Padma bridge ba Expressway toll rates web search-e dekhe ride book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="hybrid_passport_rules_and_status",
        primary_domain="govt_services",
        tools=[WEB_SEARCH, GET_SERVICE_STATUS],
        goals_bn=[
            "ই-পাসপোর্ট করার বর্তমান সরকারি ফি ও প্রয়োজনীয় কাগজপত্র ওয়েব সার্চে জেনে নেওয়া এবং নিজের পাসপোর্ট আবেদন স্ট্যাটাস চেক করা।",
        ],
        goals_banglish=[
            "E-passport fee o documents web search kore jene application status check kora.",
        ],
    ),
    ScenarioBlueprint(
        name="hybrid_sajek_escort_and_tour",
        primary_domain="travel_planning",
        tools=[WEB_SEARCH, GET_PACKAGES, BOOK_TOUR],
        goals_bn=[
            "সাজেক ভ্যালির বাঘাইহাট আর্মি এসকর্টের সময়সূচী ওয়েব সার্চে জানা এবং ৩ দিনের ট্যুর প্যাকেজ বুক করা।",
        ],
        goals_banglish=[
            "Sajek army escort timing web search kore jene tour package book kora.",
        ],
    ),
    ScenarioBlueprint(
        name="hybrid_tunnel_toll_and_flight",
        primary_domain="flight_booking",
        tools=[WEB_SEARCH, SEARCH_FLIGHTS, BOOK_FLIGHT],
        goals_bn=[
            "কর্ণফুলী বঙ্গবন্ধু টানেলের টোল রেট ওয়েব সার্চে জানা এবং চট্টগ্রাম থেকে ঢাকা ফ্লাইটের টিকিট কাটা।",
        ],
        goals_banglish=[
            "Bangabandhu tunnel toll rates web search-e dekhe Chittagong flight book kora.",
        ],
    ),
]


def sample_multi_tool_scenario(seed: int, persona: Persona) -> tuple[ScenarioBlueprint, str]:
    """Deterministically samples a scenario blueprint and a matching goal."""
    rng = random.Random(seed)
    blueprint = rng.choice(SCENARIO_BLUEPRINTS)
    script = getattr(persona, "script", "bengali") or "bengali"

    if script == "banglish" and blueprint.goals_banglish:
        goal = rng.choice(blueprint.goals_banglish)
    else:
        goal = rng.choice(blueprint.goals_bn)

    return blueprint, goal
