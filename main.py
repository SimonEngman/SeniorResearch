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

shotsgained_shots = pd.read_csv("data/PGA_data_tourlevel.csv")
print("Columns:", shotsgained_shots.columns.tolist())

#find Rory Mcilroy
rory_players = shotsgained_shots[shotsgained_shots['first_name'] == 'Rory']