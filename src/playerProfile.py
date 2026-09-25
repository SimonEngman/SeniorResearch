import pandas as pd

#Load the PGA_data_tourlevel.csv 
sg_data = pd.read_csv("data/PGA_data_tourlevel.csv")

#Load Rory Mcilroy's data
rory_shots = pd.read_csv("data/rory_2025_shots.csv")

#Load Collin Morikawa's data
collin_shots = pd.read_csv("data/collin_2025_shots.csv")

print("Rory Shots:", len(rory_shots))
print("Collin Shots:", len(collin_shots))
print("SG Rows:", len(sg_data))

#Collect Rory's SG data from sg_data
rory_sg = sg_data[
    sg_data["Player_initial_last"]
    .astype(str)
    .str.contains("McIlroy", case=False, na=False)
].copy()

#Collect Collin's SG data from sg_data
collin_sg = sg_data[
    sg_data["Player_initial_last"]
    .astype(str)
    .str.contains("Morikawa", case=False, na=False)
].copy()

print("\nRory SG rows:", len(rory_sg))
print("Collin SG rows:", len(collin_sg))

#Calculate average SG for Rory McIlroy
rory_average_sg = rory_sg[
    [
        "sg_ott",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "sg_t2g",
        "sg_total"
    ]
].mean()

print("\nRory McIlroy — 2025 Average Strokes Gained")
print(rory_average_sg)

#Calculate average SG for Collin Morikawa
collin_average_sg = collin_sg[
    [
        "sg_ott",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "sg_t2g",
        "sg_total"
    ]
].mean()

print("\nCollin Morikawa — 2025 Average Strokes Gained")
print(collin_average_sg)

#Combining the two players' average SG into a single DataFrame for comparison
#This will help with creating visualizations and further analysis
player_sg_profiles = pd.DataFrame({
    "Player": ["Rory McIlroy", "Collin Morikawa"],
    "sg_ott": [rory_average_sg["sg_ott"], collin_average_sg["sg_ott"]],
    "sg_app": [rory_average_sg["sg_app"], collin_average_sg["sg_app"]],
    "sg_arg": [rory_average_sg["sg_arg"], collin_average_sg["sg_arg"]],
    "sg_putt": [rory_average_sg["sg_putt"], collin_average_sg["sg_putt"]],
    "sg_t2g": [rory_average_sg["sg_t2g"], collin_average_sg["sg_t2g"]],
    "sg_total": [rory_average_sg["sg_total"], collin_average_sg["sg_total"]]
})  
print("\nPlayer SG Profiles:")
print(player_sg_profiles)