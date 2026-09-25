import pgatourpy as pga
import pandas as pd
import time

players = pga.pga_players()

rory_players = players[players['first_name'] == 'Rory']

print(rory_players.head())
print("\nColumns:")
print(players.columns.tolist())


tournament_id = "R2026014" #The masters Tournament 2025 ID
roryMcilroy_id = "28237" #Rory McIlroy Player ID
season = 2025

# Get Rory's 2025 tournament results
results = pga.pga_player_results(
    roryMcilroy_id,
    season=season
)

print("Found", len(results), "tournaments")

ludvigAberg = players[players['first_name'] == 'Ludvig']
print(ludvigAberg.head())
print("\nColumns:")
print(ludvigAberg.columns.tolist())


ludvigAberg_id = "52955" #Get result for Ludvig Aberg
results = pga.pga_player_results(
    ludvigAberg_id,
    season=season
)

collinMorikawa = players[players['first_name'] == 'Collin']
print(collinMorikawa.head())
print("\nColumns:")
print(collinMorikawa.columns.tolist())

shotsgained_shots = pd.read_csv("data/PGA_data_tourlevel.csv")
print("Columns:", shotsgained_shots.columns.tolist())

rory_stats = shotsgained_shots[ #I am trying to find Rory McIlroy's stats in the shotsgained_shots dataframe
    shotsgained_shots["Player_initial_last"]
    .astype(str)
    .str.contains("McIlroy", case=False, na=False)
].copy()

print(rory_stats.head())


rory_average_sg = rory_stats[
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

collin_stats = shotsgained_shots[ #I am trying to find Collin Morikawa's stats in the shotsgained_shots dataframe
    shotsgained_shots["Player_initial_last"]
    .astype(str)
    .str.contains("Morikawa", case=False, na=False)
].copy()

print(collin_stats.head())

collin_average_sg = collin_stats[
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
