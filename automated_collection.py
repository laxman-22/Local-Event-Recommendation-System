import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PLACES_API")
CITY = "Durham, NC"

def text_search(query):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{query} in {CITY}", "key": API_KEY}
    resp = requests.get(url, params=params).json()
    if resp.get("results"):
        return resp["results"][0]
    return None

def place_details(place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join([
        "geometry/location",
        "name", "formatted_address",
        "rating", "user_ratings_total",
        "price_level", "opening_hours"
    ])
    params = {"place_id": place_id, "fields": fields, "key": API_KEY}
    return requests.get(url, params=params).json().get("result", {})

activities = [
"Halloween Events",
"Thanksgiving Market",
"Santa Paws",
"Senior Holiday Party",
"Holiday Parade",
"Solstice: A Winter Circus Experience",
"Durham Farmers Market (Winter)",
"First Day Hike (Eno River State Park)",
"NC Science Festival Events",
"Easter at the Jug",
"20th Annual Strawberry Festival",
"Geek and Grub Market (May the 4th Be With You Edition)",
"Taste of Soul NC",
"Durham Craft Market (Saturdays)",
"Art Market Durham (Saturdays)",
"South Durham Farmers Market (Saturdays)"
]

rows = []
for name in activities:
    res = text_search(name)
    if not res:
        continue
    pid = res["place_id"]
    det = place_details(pid)
    row = {
        "Activity Name": det.get("name"),
        "Location": det["geometry"]["location"],
        "Price Level": det.get("price_level"),  
        "Num Reviews": det.get("user_ratings_total"),
        "Hours of Operations": det.get("opening_hours", {}).get("periods", {}),
        "Address": det.get("formatted_address"),
        "Tags": [],
        "Min Age": None, 
        "Max Age": None,
        "Min Group Size": None,
        "Max Group Size": None,
        "Event Duration": None,
    }
    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("temp.csv", index=False)
    time.sleep(0.1)





