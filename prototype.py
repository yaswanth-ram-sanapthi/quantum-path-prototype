# fleet_optimization.py
import streamlit as st
import numpy as np
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Fleet Optimization", page_icon="🚚", layout="wide")
st.title("🚚 Fleet Optimization: Shortest Route Prototype")

# --- Inputs ---
api_key = st.text_input("Google Maps API Key", type="password", help="Enable Distance Matrix + Geocoding APIs in GCP console.")

st.subheader("Stops (Addresses or lat,lng)")
default_stops = [
    "Vijayawada Railway Station",
    "MG Road, Vijayawada",
    "Krishna River Park",
    "SRM University, AP",
    "Amaravati Secretariat"
]
stop_text = st.text_area("One stop per line", value="\n".join(default_stops), height=150)
solve_roundtrip = st.checkbox("Return to start (round trip)", value=True)
start_index = st.number_input("Start at index", min_value=0, value=0, step=1)

# --- Functions ---
def geocode_stop(s):
    s = s.strip()
    if "," in s:
        lat, lng = map(float, s.split(","))
        return lat, lng, s
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": s, "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if data.get("results"):
        loc = data["results"][0]["geometry"]["location"]
        name = data["results"][0].get("formatted_address", s)
        return loc["lat"], loc["lng"], name
    else:
        st.error(f"Geocoding failed: {s}")
        st.stop()

def distance_matrix(coords):
    origins = "|".join([f"{lat},{lng}" for lat, lng, _ in coords])
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {"origins": origins, "destinations": origins, "mode": "driving", "units": "metric", "key": api_key}
    r = requests.get(url, params=params, timeout=30)
    dm = r.json()
    n = len(coords)
    D = np.zeros((n, n))
    for i, row in enumerate(dm.get("rows", [])):
        for j, elt in enumerate(row.get("elements", [])):
            if elt.get("status") == "OK":
                D[i, j] = elt["distance"]["value"] / 1000.0
            else:
                D[i, j] = float("inf")
    return D

def solve_tsp_ortools(D, start=0, roundtrip=True):
    n = len(D)
    manager = pywrapcp.RoutingIndexManager(n, 1, start)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(D[f, t] * 1000)

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_params)
    if solution:
        index = routing.Start(0)
        order = []
        total = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            order.append(node)
            next_index = solution.Value(routing.NextVar(index))
            total += routing.GetArcCostForVehicle(index, next_index, 0)
            index = next_index
        order.append(manager.IndexToNode(index))
        return order, total / 1000.0
    return None, None

# --- Main ---
if st.button("Compute Route"):
    stops = [s.strip() for s in stop_text.splitlines() if s.strip()]
    if len(stops) < 3:
        st.error("Provide at least 3 stops.")
        st.stop()
    if not api_key:
        st.warning("No API key provided. Google API may fail.")

    with st.spinner("Geocoding stops..."):
        coords = [geocode_stop(s) for s in stops]

    st.success(f"Geocoded {len(coords)} stops.")
    names = [c[2] for c in coords]

    with st.spinner("Building distance matrix..."):
        D = distance_matrix(coords)
        st.dataframe(np.round(D, 2))

    with st.spinner("Solving TSP..."):
        route, total_km = solve_tsp_ortools(D, start=start_index, roundtrip=solve_roundtrip)

    if route:
        st.success(f"Route length ≈ {total_km:.2f} km")
        st.write("Order:", route)

        m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=11)
        for idx, (lat, lng, name) in enumerate(coords):
            folium.Marker([lat, lng], popup=f"{idx}: {name}").add_to(m)
        latlng_route = [[coords[i][0], coords[i][1]] for i in route]
        folium.PolyLine(latlng_route, weight=5, opacity=0.8, color="blue").add_to(m)
        st_folium(m, height=500)

        ordered_names = [names[i] for i in route]
        st.markdown("**Stop order:**")
        st.write(" → ".join(ordered_names))
    else:
        st.warning("TSP solver failed.")
