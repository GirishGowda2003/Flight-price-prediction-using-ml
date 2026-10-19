"""
Flight Fare Predictor – Streamlit UI
====================================
Uses `gradient_boosting.pkl` (trained in Flight_Fare_Prediction.ipynb).

The model was trained on 15 engineered/encoded columns. This UI only asks for 7 plain inputs
and rebuilds the rest exactly like the notebook did:

  Airline, Source, Destination  -> dropdowns (label-encoded internally)
  Route                         -> dropdown, filtered by Source/Destination (Total_Stops comes with it)
  Date of journey               -> Journey_Day / Journey_Month / Journey_Year
  Departure time                -> Dep_Hour, Dep_Period, Dep_Time_Minutes (min-max scaled)
  Duration (h + m)              -> Duration_Minutes (min-max scaled)
  Arrival time & arrival date   -> derived = departure + duration (matches the dataset in 99.97% of rows)

The model predicts log(price), so the output is exp(prediction) in Indian rupees.

Run:
    pip install streamlit pandas numpy scikit-learn==1.6.1
    streamlit run app.py
Keep gradient_boosting.pkl in the same folder as this script.
"""
from datetime import date, datetime, time, timedelta
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("gradient_boosting.pkl")

# --------------------------------------------------------------------------------------
# Constants re-created from the notebook's preprocessing on the training data
# (LabelEncoder = sorted classes, code = position; MinMaxScaler ranges)
# --------------------------------------------------------------------------------------
AIRLINES = ["Air Asia", "Air India", "GoAir", "IndiGo", "Jet Airways", "Jet Airways Business",
            "Multiple carriers", "Multiple carriers Premium economy", "SpiceJet", "Trujet",
            "Vistara", "Vistara Premium economy"]
SOURCES = ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
DESTINATIONS = ["Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"]
STOPS = ["1 stop", "2 stops", "3 stops", "4 stops", "non-stop"]
DEP_PERIODS = ["Afternoon", "Early Morning", "Evening", "Morning"]
ARRIVAL_DATES = ["02", "03", "04", "05", "07", "08", "10", "11", "13", "16", "17", "19", "20", "22",
                 "23", "25", "26", "28", "29", None]          # None (last) = arrives the same day
ARRIVAL_MONTHS = ["Apr", "Jun", "Mar", "May", None]           # None (last) = arrives the same day
MONTH_ABBR = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}

DURATION_RANGE = (5.0, 2860.0)       # minutes, MinMaxScaler
DEP_MINUTES_RANGE = (20.0, 1435.0)   # minutes since midnight, MinMaxScaler

# Training data covers journeys from 1 Mar to 27 Jun 2019 only
DATE_MIN, DATE_MAX = date(2019, 3, 1), date(2019, 6, 27)

# Valid (Source, Destination) -> routes (most common first) with their Total_Stops value
ROUTES = {
    ("Banglore", "Delhi"): [
        ("BLR → DEL", "non-stop"),
    ],
    ("Banglore", "New Delhi"): [
        ("BLR → BOM → DEL", "1 stop"),
        ("BLR → DEL", "non-stop"),
        ("BLR → HYD → DEL", "1 stop"),
        ("BLR → MAA → DEL", "1 stop"),
        ("BLR → AMD → DEL", "1 stop"),
        ("BLR → COK → DEL", "1 stop"),
        ("BLR → CCU → DEL", "1 stop"),
        ("BLR → CCU → GAU → DEL", "2 stops"),
        ("BLR → BOM → BHO → DEL", "2 stops"),
        ("BLR → HYD → VGA → DEL", "2 stops"),
        ("BLR → VGA → DEL", "1 stop"),
        ("BLR → BOM → NAG → DEL", "2 stops"),
        ("BLR → BDQ → DEL", "1 stop"),
        ("BLR → GOI → DEL", "1 stop"),
        ("BLR → BOM → JDH → DEL", "2 stops"),
        ("BLR → PNQ → DEL", "1 stop"),
        ("BLR → BOM → IDR → DEL", "2 stops"),
        ("BLR → BBI → DEL", "1 stop"),
        ("BLR → CCU → BBI → DEL", "2 stops"),
        ("BLR → IDR → DEL", "1 stop"),
        ("BLR → BOM → AMD → DEL", "2 stops"),
        ("BLR → VGA → HYD → DEL", "2 stops"),
        ("BLR → NAG → DEL", "1 stop"),
        ("BLR → LKO → DEL", "1 stop"),
        ("BLR → GAU → DEL", "1 stop"),
        ("BLR → BOM → UDR → DEL", "2 stops"),
        ("BLR → CCU → BBI → HYD → DEL", "3 stops"),
        ("BLR → BOM → IDR → GWL → DEL", "3 stops"),
        ("BLR → TRV → COK → DEL", "2 stops"),
        ("BLR → VGA → VTZ → DEL", "2 stops"),
        ("BLR → STV → DEL", "1 stop"),
        ("BLR → HBX → BOM → BHO → DEL", "3 stops"),
        ("BLR → HBX → BOM → AMD → DEL", "3 stops"),
        ("BLR → HBX → BOM → NAG → DEL", "3 stops"),
        ("BLR → BOM → IXC → DEL", "2 stops"),
        ("BLR → CCU → BBI → HYD → VGA → DEL", "4 stops"),
    ],
    ("Chennai", "Kolkata"): [
        ("MAA → CCU", "non-stop"),
    ],
    ("Delhi", "Cochin"): [
        ("DEL → BOM → COK", "1 stop"),
        ("DEL → HYD → COK", "1 stop"),
        ("DEL → BLR → COK", "1 stop"),
        ("DEL → COK", "non-stop"),
        ("DEL → JAI → BOM → COK", "2 stops"),
        ("DEL → MAA → COK", "1 stop"),
        ("DEL → AMD → BOM → COK", "2 stops"),
        ("DEL → IDR → BOM → COK", "2 stops"),
        ("DEL → HYD → MAA → COK", "2 stops"),
        ("DEL → BHO → BOM → COK", "2 stops"),
        ("DEL → HYD → BOM → COK", "2 stops"),
        ("DEL → CCU → BOM → COK", "2 stops"),
        ("DEL → NAG → BOM → COK", "2 stops"),
        ("DEL → GOI → BOM → COK", "2 stops"),
        ("DEL → PNQ → COK", "1 stop"),
        ("DEL → LKO → BOM → COK", "2 stops"),
        ("DEL → ATQ → BOM → COK", "2 stops"),
        ("DEL → JDH → BOM → COK", "2 stops"),
        ("DEL → BDQ → BOM → COK", "2 stops"),
        ("DEL → AMD → COK", "1 stop"),
        ("DEL → UDR → BOM → COK", "2 stops"),
        ("DEL → LKO → COK", "1 stop"),
        ("DEL → TRV → COK", "1 stop"),
        ("DEL → RPR → NAG → BOM → COK", "3 stops"),
        ("DEL → IXU → BOM → COK", "2 stops"),
        ("DEL → MAA → BOM → COK", "2 stops"),
        ("DEL → IXC → BOM → COK", "2 stops"),
        ("DEL → GWL → IDR → BOM → COK", "3 stops"),
        ("DEL → BBI → COK", "1 stop"),
        ("DEL → DED → BOM → COK", "2 stops"),
    ],
    ("Kolkata", "Banglore"): [
        ("CCU → BOM → BLR", "1 stop"),
        ("CCU → BLR", "non-stop"),
        ("CCU → DEL → BLR", "1 stop"),
        ("CCU → MAA → BLR", "1 stop"),
        ("CCU → HYD → BLR", "1 stop"),
        ("CCU → BBI → BLR", "1 stop"),
        ("CCU → IXR → DEL → BLR", "2 stops"),
        ("CCU → BBI → BOM → BLR", "2 stops"),
        ("CCU → GAU → BLR", "1 stop"),
        ("CCU → GAU → DEL → BLR", "2 stops"),
        ("CCU → BOM → COK → BLR", "2 stops"),
        ("CCU → DEL → COK → BLR", "2 stops"),
        ("CCU → DEL → AMD → BLR", "2 stops"),
        ("CCU → IXB → BLR", "1 stop"),
        ("CCU → PNQ → BLR", "1 stop"),
        ("CCU → JAI → BOM → BLR", "2 stops"),
        ("CCU → BOM → GOI → BLR", "2 stops"),
        ("CCU → AMD → BLR", "1 stop"),
        ("CCU → VNS → DEL → BLR", "2 stops"),
        ("CCU → PAT → BLR", "1 stop"),
        ("CCU → KNU → BLR", "1 stop"),
        ("CCU → NAG → BLR", "1 stop"),
        ("CCU → BOM → AMD → BLR", "2 stops"),
        ("CCU → BBI → HYD → BLR", "2 stops"),
        ("CCU → IXR → BBI → BLR", "2 stops"),
        ("CCU → JAI → DEL → BLR", "2 stops"),
        ("CCU → BOM → HBX → BLR", "2 stops"),
        ("CCU → GAU → IMF → DEL → BLR", "3 stops"),
        ("CCU → BOM → PNQ → BLR", "2 stops"),
        ("CCU → DEL → VGA → BLR", "2 stops"),
        ("CCU → DEL → COK → TRV → BLR", "3 stops"),
        ("CCU → BBI → IXR → DEL → BLR", "3 stops"),
        ("CCU → BOM → TRV → BLR", "2 stops"),
        ("CCU → RPR → HYD → BLR", "2 stops"),
        ("CCU → VTZ → BLR", "1 stop"),
        ("CCU → IXZ → MAA → BLR", "2 stops"),
        ("CCU → IXA → BLR", "1 stop"),
        ("CCU → IXB → DEL → BLR", "2 stops"),
    ],
    ("Mumbai", "Hyderabad"): [
        ("BOM → HYD", "non-stop"),
        ("BOM → DEL → HYD", "1 stop"),
        ("BOM → BHO → DEL → HYD", "2 stops"),
        ("BOM → GOI → PNQ → HYD", "2 stops"),
        ("BOM → AMD → ISK → HYD", "2 stops"),
        ("BOM → JDH → DEL → HYD", "2 stops"),
        ("BOM → MAA → HYD", "1 stop"),
        ("BOM → BLR → HYD", "1 stop"),
        ("BOM → BDQ → DEL → HYD", "2 stops"),
        ("BOM → IDR → DEL → HYD", "2 stops"),
        ("BOM → GOI → HYD", "1 stop"),
        ("BOM → DED → DEL → HYD", "2 stops"),
        ("BOM → JLR → HYD", "1 stop"),
        ("BOM → JDH → JAI → DEL → HYD", "3 stops"),
        ("BOM → COK → MAA → HYD", "2 stops"),
        ("BOM → NDC → HYD", "1 stop"),
        ("BOM → CCU → HYD", "1 stop"),
        ("BOM → BLR → CCU → BBI → HYD", "3 stops"),
        ("BOM → UDR → DEL → HYD", "2 stops"),
        ("BOM → RPR → VTZ → HYD", "2 stops"),
        ("BOM → JAI → DEL → HYD", "2 stops"),
        ("BOM → VNS → DEL → HYD", "2 stops"),
        ("BOM → BBI → HYD", "1 stop"),
    ],
}

# All 128 routes the encoder knows (sorted = LabelEncoder order)
ALL_ROUTES = sorted({r for lst in ROUTES.values() for r, _ in lst})
ROUTE_STOPS = {r: s for lst in ROUTES.values() for r, s in lst}

CITY_LABEL = {"Banglore": "Bengaluru", "Cochin": "Kochi"}
AIRPORTS = {"BLR": "Bengaluru", "DEL": "Delhi", "BOM": "Mumbai", "HYD": "Hyderabad", "COK": "Kochi",
            "MAA": "Chennai", "CCU": "Kolkata", "AMD": "Ahmedabad", "ATQ": "Amritsar", "BBI": "Bhubaneswar",
            "BDQ": "Vadodara", "BHO": "Bhopal", "DED": "Dehradun", "GAU": "Guwahati", "GOI": "Goa",
            "GWL": "Gwalior", "HBX": "Hubli", "IDR": "Indore", "IMF": "Imphal", "ISK": "Nashik",
            "IXA": "Agartala", "IXB": "Bagdogra", "IXC": "Chandigarh", "IXR": "Ranchi", "IXU": "Aurangabad",
            "IXZ": "Port Blair", "JAI": "Jaipur", "JDH": "Jodhpur", "JLR": "Jabalpur", "KNU": "Kanpur",
            "LKO": "Lucknow", "NAG": "Nagpur", "NDC": "Nanded", "PAT": "Patna", "PNQ": "Pune",
            "RPR": "Raipur", "STV": "Surat", "TRV": "Thiruvananthapuram", "UDR": "Udaipur",
            "VGA": "Vijayawada", "VNS": "Varanasi", "VTZ": "Visakhapatnam"}


# --------------------------------------------------------------------------------------
# Feature building + prediction
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _nearest(classes, value: int):
    """Closest known (numeric-string) class, for arrival days the training data never contained."""
    known = [c for c in classes if c is not None]
    return min(known, key=lambda c: abs(int(c) - value))


def build_features(airline, source, destination, route, journey_date: date, dep: time, dur_minutes: int):
    dep_dt = datetime.combine(journey_date, dep)
    arr_dt = dep_dt + timedelta(minutes=dur_minutes)
    same_day = arr_dt.date() == dep_dt.date()

    # Arrival date/month: None when the flight lands on the departure day (as in the dataset)
    if same_day:
        arr_date_code = len(ARRIVAL_DATES) - 1
        arr_month_code = len(ARRIVAL_MONTHS) - 1
    else:
        day = _nearest(ARRIVAL_DATES, arr_dt.day)
        arr_date_code = ARRIVAL_DATES.index(day)
        month_key = min(MONTH_ABBR, key=lambda m: abs(m - arr_dt.month))
        arr_month_code = ARRIVAL_MONTHS.index(MONTH_ABBR[month_key])

    h = dep.hour
    period = "Early Morning" if h < 6 else "Morning" if h < 12 else "Afternoon" if h < 18 else "Evening"
    dep_min = h * 60 + dep.minute

    features = {
        "Airline": AIRLINES.index(airline),
        "Source": SOURCES.index(source),
        "Destination": DESTINATIONS.index(destination),
        "Route": ALL_ROUTES.index(route),
        "Total_Stops": STOPS.index(ROUTE_STOPS[route]),
        "Journey_Day": journey_date.day,
        "Journey_Month": journey_date.month,
        "Journey_Year": journey_date.year,
        "Dep_Hour": h,
        "Dep_Period": DEP_PERIODS.index(period),
        "Duration_Minutes": (dur_minutes - DURATION_RANGE[0]) / (DURATION_RANGE[1] - DURATION_RANGE[0]),
        "Dep_Time_Minutes": (dep_min - DEP_MINUTES_RANGE[0]) / (DEP_MINUTES_RANGE[1] - DEP_MINUTES_RANGE[0]),
        "Arrival_Time_Minutes": arr_dt.hour * 60 + arr_dt.minute,   # not scaled in the notebook
        "Arrival_Date": arr_date_code,
        "Arrival_Month": arr_month_code,
    }
    return features, arr_dt, same_day


def predict_fare(model, features: dict) -> float:
    X = pd.DataFrame([features], columns=list(model.feature_names_in_))
    return float(np.exp(model.predict(X)[0]))          # model was trained on log(Price)


def inr(x: float) -> str:
    """Format with Indian digit grouping, e.g. 1,23,456."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return f"₹{s}"
    head, last3 = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join(groups + [last3])


def route_label(route: str) -> str:
    pretty = " → ".join(AIRPORTS.get(c.strip(), c.strip()) for c in route.split("→"))
    return f"{route}   ({ROUTE_STOPS[route]})  ·  {pretty}"


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Flight Fare Predictor")
st.caption("Gradient Boosting model trained on Indian domestic flights (Mar–Jun 2019). "
           "Fares are in Indian rupees (₹).")

model = load_model()
result_box = st.container()

st.subheader("Flight details")
c1, c2, c3 = st.columns(3)
airline = c1.selectbox("Airline", AIRLINES, index=AIRLINES.index("Jet Airways"))
source = c2.selectbox("Source", list(dict.fromkeys(s for s, _ in ROUTES)),
                      index=2, format_func=lambda v: CITY_LABEL.get(v, v))
dest_options = [d for s, d in ROUTES if s == source]
destination = c3.selectbox("Destination", dest_options, format_func=lambda v: CITY_LABEL.get(v, v))
if destination == "New Delhi" or destination == "Delhi":
    c3.caption("Dataset lists both “Delhi” and “New Delhi”.")

route = st.selectbox("Route (with number of stops)", [r for r, _ in ROUTES[(source, destination)]],
                     format_func=route_label)

st.subheader("Schedule")
d1, d2, d3 = st.columns(3)
journey_date = d1.date_input("Date of journey", value=date(2019, 5, 15), min_value=DATE_MIN, max_value=DATE_MAX,
                             help="The model only saw journeys between 1 Mar and 27 Jun 2019.")
dep_time = d2.time_input("Departure time", value=time(9, 0), step=300)
with d3:
    hh, mm = st.columns(2)
    dur_h = hh.number_input("Duration (h)", 0, 47, 7, 1)
    dur_m = mm.number_input("(min)", 0, 59, 30, 5)
duration_minutes = int(dur_h) * 60 + int(dur_m)
duration_ok = duration_minutes >= 5
if not duration_ok:
    st.warning("Duration must be at least 5 minutes.")

features, arrival_dt, same_day = build_features(airline, source, destination, route, journey_date,
                                                dep_time, max(duration_minutes, 5))
st.caption(f"🛬 Arrival (calculated): **{arrival_dt:%d %b %Y, %H:%M}**"
           f"{'' if same_day else '  (next day or later)'}  ·  Stops: **{ROUTE_STOPS[route]}**")

with result_box:
    if duration_ok:
        fare = predict_fare(model, features)
        left, right = st.columns([1, 2])
        left.metric("Estimated fare", inr(fare))
        right.caption("Typical error of the model on unseen flights is about ±12%. "
                      "This is an estimate, not a booking price.")
    with st.expander("Show model input (15 features)"):
        st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "value"}))
