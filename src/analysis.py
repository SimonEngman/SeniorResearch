import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

shots = pd.read_csv("data/ludvig_2025_shots.csv")

print("Total shots:", len(shots))
print("Columns:", shots.columns.tolist())

tee_shots = shots[
    (shots["stroke_number"] == 1) &
    (shots["par"] != 3)
].copy()

print("Total tee shots:", len(tee_shots))
print(tee_shots.head())



# Coordinate columns
from_x = "leftToRightCoords.fromCoords.x"
from_y = "leftToRightCoords.fromCoords.y"

to_x = "leftToRightCoords.toCoords.x"
to_y = "leftToRightCoords.toCoords.y"

# Remove shots with missing coordinates
tee_shots = tee_shots.dropna(
    subset=[from_x, from_y, to_x, to_y]
)

# Get all shots
all_shots = shots[
    shots["par"] != 3
].copy()

# Find the final position of each hole
hole_ends = (
    all_shots[all_shots["final_stroke"] == True]
    .sort_values(
        ["tournament_id", "hole_number", "stroke_number"]
    )
    .groupby(
        ["tournament_id", "hole_number"]
    )
    .tail(1)
)

# Keep endpoint coordinates
hole_ends = hole_ends[
    ["tournament_id", "hole_number", to_x, to_y]
].rename(
    columns={
        to_x: "hole_end_x",
        to_y: "hole_end_y"
    }
)

# Attach hole endpoint to each tee shot
tee_shots = tee_shots.merge(
    hole_ends,
    on=["tournament_id", "hole_number"],
    how="left"
)

# Remove shots where we couldn't find the hole endpoint
tee_shots = tee_shots.dropna(
    subset=["hole_end_x", "hole_end_y"]
)

# Tee position
tee_x = tee_shots[from_x].values
tee_y = tee_shots[from_y].values

# Hole endpoint
end_x = tee_shots["hole_end_x"].values
end_y = tee_shots["hole_end_y"].values

# Direction from tee → hole
hole_dx = end_x - tee_x
hole_dy = end_y - tee_y

# Length of hole in coordinate units
hole_length = np.sqrt(
    hole_dx**2 + hole_dy**2
)

# Unit vector down the hole
unit_x = hole_dx / hole_length
unit_y = hole_dy / hole_length

# Tee shot landing position
shot_x = tee_shots[to_x].values
shot_y = tee_shots[to_y].values

shot_dx = shot_x - tee_x
shot_dy = shot_y - tee_y

# Distance DOWN the hole
down_hole_coordinate = (
    shot_dx * unit_x +
    shot_dy * unit_y
)

# Distance OFFLINE
offline_coordinate = (
    shot_dx * (-unit_y) +
    shot_dy * unit_x
)

# Convert coordinate units → yards
yardage = (
    tee_shots["yardage"]
    .astype(str)
    .str.replace(" yds", "", regex=False)
    .astype(float)
    .values
)

scale = yardage / hole_length

tee_shots["down_hole_yards"] = (
    down_hole_coordinate * scale
)

tee_shots["offline_yards"] = (
    offline_coordinate * scale
)

# Below is figure 1, total dispersion included
plt.figure(
    figsize=(8, 10),
    facecolor="#cccccc"
)

plt.scatter(
    tee_shots["offline_yards"],
    tee_shots["down_hole_yards"],
    alpha=0.5
)

# Centerline
plt.axvline(
    0,
    linewidth=1
)

plt.xlabel("Offline distance (yards)")
plt.ylabel("Distance down hole (yards)")
plt.title(
    "Ludvig Åberg — Tee Shot Dispersion (Par 4 & 5)"
)

plt.grid(True)

plt.show()

#Below is figure 2, removing the outside 30% of the dispersion

# Center point
center_x = 0
center_y = 317

# Calculate distance from the center point
tee_shots["distance_from_center"] = np.sqrt(
    (tee_shots["offline_yards"] - center_x) ** 2 +
    (tee_shots["down_hole_yards"] - center_y) ** 2
)

# Find the distance that contains the closest 70% of shots
radius = tee_shots["distance_from_center"].quantile(0.70)

# Keep the closest 70%
middle_70 = tee_shots[
    tee_shots["distance_from_center"] <= radius
].copy()

print("Total shots:", len(tee_shots))
print("Central 70%:", len(middle_70))
print("Dispersion radius:", round(radius, 1), "yards")


# Plot
plt.figure(
    figsize=(8, 8),
    facecolor="#cccccc"
)

plt.scatter(
    middle_70["offline_yards"],
    middle_70["down_hole_yards"],
    alpha=0.5
)

# Centerline
plt.axvline(
    0,
    linewidth=1
)

# Average driving distance
plt.axhline(
    317,
    linewidth=1
)

# Center point
plt.scatter(
    0,
    317,
    s=100,
    marker="x"
)

plt.xlabel("Offline distance (yards)")
plt.ylabel("Distance down hole (yards)")

plt.title(
    "Ludvig Åberg — Central 70% Driver Dispersion"
)

plt.grid(True)

plt.axis("equal")

plt.show()