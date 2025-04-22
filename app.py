import streamlit as st
from models.non_deep_learning import recommend, cold_start_recommend, update_model
from rapidfuzz import process
import pandas as pd
from time import sleep

activity_df = pd.read_csv("./data/processed/Cleaned_Activity_Labels.csv")
true_activity_names = activity_df["Cleaned"].tolist()

def match_activity_name(input_name):
    match, _, _ = process.extractOne(input_name, true_activity_names)
    return match

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

user_map = {"Alice": 1, "Bob": 2, "Cold Start User": 9999}
st.set_page_config(layout="wide")
st.title("⭐ Local Activity Recommender")

with st.sidebar:
    with st.form("user_selection_form"):
        st.header("Select User")
        selected_user = st.selectbox("Choose a user", list(user_map.keys()))

        col_enter, col_clear = st.columns([1, 1])

        with col_enter:
            enter_clicked = st.form_submit_button("Enter")

        with col_clear:
            clear_clicked = st.form_submit_button("Clear")

        if enter_clicked:
            user_id = user_map[selected_user]
            st.session_state.ratings = {}
            st.session_state.current_user = selected_user
            if user_id == 9999:
                st.session_state.recommendations = None
            else:
                st.session_state.recommendations = recommend(user_id)

        if clear_clicked:
            st.session_state.current_user = None
            st.session_state.recommendations = []
            st.session_state.ratings = {}
            st.rerun()

# Display recommendations if a user has been selected
if st.session_state.current_user != "Cold Start User" and st.session_state.current_user is not None:
    st.subheader(f"🔍 Recommendations for {st.session_state.current_user}")
    recommendations = st.session_state.recommendations[:5]
    print(recommendations)

    for activity, score in recommendations:
        key = activity.replace(" ", "_")
        slider_key = f"slider_{key}"
        submit_key = f"submit_{key}"

        col1, col2, col3 = st.columns([1, 2, 1])

        activity = match_activity_name(activity)
        with col2:
            with st.container():
                st.markdown(
                    f"""
                    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #f9f9f9;'>
                        <h4 style='color: black; margin-bottom: 10px; text-align: center;'>{activity}</h4>
                        <a style='text-align: center;' href="https://www.google.com/search?q={activity}" target="_blank">
                                Click here to learn more!
                        </a>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("**Drag to rate:**")

                rating = st.slider(
                    "",
                    0.0,
                    5.0,
                    step=0.5,
                    key=slider_key,
                    label_visibility="collapsed"
                )

                full_stars = int(rating)
                half_star = 1 if rating - full_stars >= 0.5 else 0
                empty_stars = 5 - full_stars - half_star
                star_display = "⭐" * full_stars + "🌓" * half_star + "☆" * empty_stars
                st.markdown(
                    f"<div style='text-align: center; margin: 10px 0;'><span style='font-size: 30px;'>{star_display}</span></div>",
                    unsafe_allow_html=True
                )
            if st.button("Submit", key=submit_key):
                st.session_state.ratings[activity] = rating
                st.success(f"✅ You rated **{activity}**: {rating} {star_display}")
                user_id = user_map[st.session_state.current_user]
                with st.spinner("Updating model..."):
                    update_model(user_id, activity, rating)
                    sleep(2)
                st.session_state.recommendations = recommend(user_id)
                st.rerun()


            st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.current_user == "Cold Start User" and st.session_state.current_user is not None:
    if st.session_state.recommendations:
        st.subheader("🎯 Personalized Results Based on Your Answers")
        recommendations = st.session_state.recommendations[:5]

        for activity, score in recommendations:
            key = activity.replace(" ", "_")
            slider_key = f"slider_{key}"
            submit_key = f"submit_{key}"

            col1, col2, col3 = st.columns([1, 2, 1])
            activity = match_activity_name(activity)
            with col2:
                with st.container():
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #f9f9f9;'>
                            <h4 style='color: black; margin-bottom: 10px; text-align: center;'>{activity}</h4>
                            <a style='text-align: center;' href="https://www.google.com/search?q={activity}" target="_blank">
                                Click here to learn more!
                            </a>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("**Drag to rate:**")

                    rating = st.slider("", 0.0, 5.0, step=0.5, key=slider_key, label_visibility="collapsed")
                    full_stars = int(rating)
                    half_star = 1 if rating - full_stars >= 0.5 else 0
                    empty_stars = 5 - full_stars - half_star
                    star_display = "⭐" * full_stars + "🌓" * half_star + "☆" * empty_stars
                    st.markdown(
                        f"<div style='text-align: center; margin: 10px 0;'><span style='font-size: 30px;'>{star_display}</span></div>",
                        unsafe_allow_html=True
                    )

                if st.button("Submit", key=submit_key):
                    st.session_state.ratings[activity] = rating
                    st.success(f"✅ You rated **{activity}**: {rating} {star_display}")
                    user_id = user_map[st.session_state.current_user]
                    with st.spinner("Updating model..."):
                        update_model(user_id, activity, rating)
                        sleep(2)
                    st.session_state.recommendations = recommend(user_id)
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.subheader("🧭 Tell us about yourself to get personalized activity suggestions")
        with st.form("cold_start_form"):
            selected_tags = st.multiselect(
                "What types of activities do you enjoy?",
                options=[
                    "Adventure", "Creative", "Cultural", "Educational", "Entertainment", "Festival",
                    "Food", "Free", "Game", "Outdoor", "Indoor", "Relaxing", "Social", "Workshop",
                    "Seasonal", "Family-Friendly", "Adults-Only"
                ],
            )

            price_level = st.slider("Preferred price level", 0, 4, value=2)
            min_age = st.slider("Minimum age in your group", 0, 80, value=18)
            max_age = st.slider("Maximum age in your group", 0, 100, value=35)
            group_size = st.slider("Preferred group size", 1, 10, value=2)
            event_duration = st.slider("Preferred event duration (hrs)", 1, 6, value=2)

            submitted = st.form_submit_button("Get Recommendations")

            if submitted:
                TAGS_ALL = [
                    "Adults-Only", "Adventure", "Budget-Friendly", "Concert", "Creative", "Cultural", "Educational",
                    "Entertainment", "Expensive", "Fall", "Family-Friendly", "Festival", "Free", "Game", "High",
                    "Indoor", "Low", "Market", "Medium", "Meetup", "Mixed", "Outdoor", "Relaxing", "Seasonal",
                    "Social", "Spring", "Summer", "Winter", "Workshop", "Year-Round"
                ]

                profile_dict = {
                    "selected_tags": selected_tags,
                    "tags_all_possible": TAGS_ALL,
                    "price_level": price_level,
                    "min_age": min_age,
                    "max_age": max_age,
                    "min_group_size": group_size,
                    "max_group_size": group_size,
                    "event_duration": event_duration
                }

                st.session_state.recommendations = cold_start_recommend(profile_dict)
                print(st.session_state.recommendations)
                st.rerun()
else:
    st.info("👈 Select a user and press **Enter** to see recommendations.")
