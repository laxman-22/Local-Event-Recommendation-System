from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from difflib import get_close_matches
from time import sleep
import json

load_dotenv(find_dotenv())

def main():
    """ Used to interpolate some data in the auto_events.csv dataset using gemini"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    activities = [
    "Wine Tasting at The Wine Feed",
    "Wine Tasting at Uncorked",
    "Cooking Class at Durham Co-op Market",
    "Cooking Class at Southern Season (Chapel Hill)",
    "Art Workshop at The Carrack Modern Art",
    "Art Workshop at Durham Arts Council",
    "Pottery Night at Claymakers",
    "Pottery Class at The Durham Studio School",
    "Guided Hike at Eno River State Park",
    "Guided Hike with Triangle Land Conservancy",
    "Ghost Tour of Downtown Durham",
    "Tobacco Road Tours - Haunted Durham",
    "Wine & Design (Paint & Sip)",
    "AR Workshop Durham (DIY Crafts)",
    "Bull City Craft - Workshops",
    "Cooking at The Kitchen Specialist",
    "Eno River Kayak Tours (Guided)",
    "Fullsteam Brewery Tours & Tastings"
    ]

    df = pd.read_csv("../data/auto_events.csv", encoding='utf-8')
    df["Activity Name"] = df["Activity Name"].astype(str).str.strip().str.lower()

    count = 0
    for activity in activities:
        activity = activity.strip().lower()
        matches = get_close_matches(activity, df["Activity Name"], n=1, cutoff=0.7)
        if matches:
            best = matches[0]
            address = df.loc[df["Activity Name"] == best, "Address"].values

            res = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"Can you look up and accurately fill in the following details for this activity? \
                Activity Name: {activity}   \
                Location: Durham, NC   \
                \
                The tag structure should be as follows: \
                Experience Type: Adventure, Entertainment, Social, Relaxing, Cultural, Educational \
                Event Type: Festival, Concert, Game, Meetup, Workshop, Seasonal \
                Setting: Indoor, Outdoor, Mixed \
                Audience: Family-Friendly, Kids, Couples, Adults-Only \
                Physical Activity: Low, Medium, High \
                Budget: Free, Budget-Friendly, Expensive \
                Seasonality: Summer, Winter, Fall, Spring Year-Round \
                \
                Please search online to provide: \
                \
                - Price Level ($-$$$$) \
                - Tags (one tag for each category) \
                - Min Age \
                - Max Age \
                - Min Group Size \
                - Max Group Size \
                - Event Duration (in hours) \
                - Location (as a dict containing keys lat and long) \
                - Num Reviews \
                - Address \
                                            \
                Return results in JSON format. Event Duration should be in hours."
            )
            row = json.loads(res.text.split('json')[1].split('```')[0])

            price_level = len(row["Price Level"])
            tags = row["Tags"]
            tags_list = []
            
            tags_list.append(tags["Experience Type"])
            tags_list.append(tags["Event Type"])
            tags_list.append(tags["Setting"])
            tags_list.append(tags["Audience"])
            tags_list.append(tags["Physical Activity"])
            tags_list.append(tags["Budget"])
            tags_list.append(tags["Seasonality"])

            min_age = row["Min Age"]
            max_age = row["Max Age"]
            min_group_size = row["Min Group Size"]
            max_group_size = row["Max Group Size"]
            print(row)
            event_duration = row["Event Duration"]
            location = row["Location"]
            num_reviews = row["Num Reviews"]
            address = row["Address"]

            df.loc[df["Activity Name"] == best, "Price Level"] = price_level
            idx = df.loc[df["Activity Name"] == best, "Tags"].index[0]
            df.at[idx, "Tags"] = tags_list
            df.loc[df["Activity Name"] == best, "Min Age"] = min_age
            df.loc[df["Activity Name"] == best, "Max Age"] = max_age
            df.loc[df["Activity Name"] == best, "Min Group Size"] = min_group_size
            df.loc[df["Activity Name"] == best, "Max Group Size"] = max_group_size
            df.loc[df["Activity Name"] == best, "Event Duration"] = event_duration
            df.loc[df["Activity Name"] == best, "Location"] = location
            df.loc[df["Activity Name"] == best, "Num Reviews"] = num_reviews
            df.loc[df["Activity Name"] == best, "Address"] = address

            df.to_csv("../auto_events.csv", index=False, encoding='utf-8')
        
        if count == 13:
            sleep(60)
            count += 1
        else:
            count = 0
            sleep(0.5)

if __name__ == "__main__":
    main()