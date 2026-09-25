import pandas as pd
import streamlit as st

#Load the SG data
sg_data = pd.read_csv("data/PGA_data_tourlevel.csv")

#Get the unique players from the SG data
rory_sg = sg_data[
    sg_data["Player_initial_last"]
    .astype(str)
    .str.contains("McIlroy", case=False, na=False)
].copy()

collin_sg = sg_data[
    sg_data["Player_initial_last"]
    .astype(str)
    .str.contains("Morikawa", case=False, na=False)
].copy()

#player selection
st.title("Golf Player Performance Analysis")

player = st.selectbox(
    "Select Player",
    ["Rory McIlroy", "Collin Morikawa"]
)

#select the player data based on the selection
if player == "Rory McIlroy":
   player_sg = rory_sg
else:
   player_sg = collin_sg

#Create the profile
profile = {
    "SG Off-the-Tee": player_sg["sg_ott"].mean(),
    "SG Approach": player_sg["sg_app"].mean(),
    "SG Around-the-Green": player_sg["sg_arg"].mean(),
    "SG Putting": player_sg["sg_putt"].mean(),
    "SG Tee-to-Green": player_sg["sg_t2g"].mean(),
    "SG Total": player_sg["sg_total"].mean()
}
# Create visualization data


sg_chart = pd.DataFrame({
    "Statistic": [
        "SG Off-the-Tee",
        "SG Approach",
        "SG Around-the-Green",
        "SG Putting",
        "SG Tee-to-Green",
        "SG Total"
    ],
    "Value": [
        profile["SG Off-the-Tee"],
        profile["SG Approach"],
        profile["SG Around-the-Green"],
        profile["SG Putting"],
        profile["SG Tee-to-Green"],
        profile["SG Total"]
    ]
})


# SG Profile Chart

st.header("Strokes Gained Profile")

st.bar_chart(
    sg_chart,
    x="Statistic",
    y="Value"
)

# Rory vs Collin comparison
comparison = pd.DataFrame({
    "Statistic": [
        "SG Off-the-Tee",
        "SG Approach",
        "SG Around-the-Green",
        "SG Putting",
        "SG Tee-to-Green",
        "SG Total"
    ],

    "Rory McIlroy": [
        rory_sg["sg_ott"].mean(),
        rory_sg["sg_app"].mean(),
        rory_sg["sg_arg"].mean(),
        rory_sg["sg_putt"].mean(),
        rory_sg["sg_t2g"].mean(),
        rory_sg["sg_total"].mean()
    ],

    "Collin Morikawa": [
        collin_sg["sg_ott"].mean(),
        collin_sg["sg_app"].mean(),
        collin_sg["sg_arg"].mean(),
        collin_sg["sg_putt"].mean(),
        collin_sg["sg_t2g"].mean(),
        collin_sg["sg_total"].mean()
    ]
})

st.header("Rory vs. Collin")

st.bar_chart(
    comparison,
    x="Statistic",
    y=["Rory McIlroy", "Collin Morikawa"]
)

#display the statistics
st.header (f"{player} — Stroked Gained Statistics")
for stat, value in profile.items():
   st.metric(stat, f"{value:.3f}")