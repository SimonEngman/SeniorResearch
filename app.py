import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from playerProfile import data_distance_collin, data_distance_rory

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

fig, ax = plt.subplots()

ax.bar(sg_chart["Statistic"], sg_chart["Value"])

ax.set_xlabel("Statistic")
ax.set_ylabel("Strokes Gained")
ax.set_title(f"{player} Strokes Gained Profile")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)
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

fig, ax = plt.subplots()

x = range(len(comparison["Statistic"]))
width = 0.35

ax.bar(
    [i - width/2 for i in x],
    comparison["Rory McIlroy"],
    width,
    label="Rory McIlroy"
)

ax.bar(
    [i + width/2 for i in x],
    comparison["Collin Morikawa"],
    width,
    label="Collin Morikawa"
)

ax.set_xlabel("Statistic")
ax.set_ylabel("Strokes Gained")
ax.set_title("Rory vs. Collin")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Statistic"], rotation=45, ha="right")
ax.legend()

plt.tight_layout()

st.pyplot(fig)

#display the statistics
st.header (f"{player} — Stroked Gained Statistics")
for stat, value in profile.items():
   st.metric(stat, f"{value:.3f}")


#Displaying the distance data for the selected player
if player == "Rory McIlroy":
    distance_data = data_distance_rory
else:
    distance_data = data_distance_collin

distance_chart = pd.DataFrame({
   "Club": list(distance_data["Club"]),
   "Distance": list(distance_data["Distance"])
})

st.header(f"{player} — Average Distance by Club")
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    distance_chart["Club"],
    distance_chart["Distance"]
)

ax.set_xlabel("Average Yardage")
ax.set_ylabel("Club")
ax.set_title(f"{player} Average Club Yardages")

# Put the highest-lofted/shortest club at the bottom
ax.invert_yaxis()

# Display the yardage number at the end of each bar
for i, value in enumerate(distance_chart["Distance"]):
    ax.text(
        value + 2,
        i,
        f"{value} yds",
        va="center"
    )

plt.tight_layout()

st.pyplot(fig)
