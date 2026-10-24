
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shapely import affinity

from analysis import (
    filter_tee_dispersion_by_distance,
    load_player_data,
    calculate_dispersion_typdir
)

# Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Path to the QGIS GeoPackage
COURSE_FILE = PROJECT_DIR / "data" / "course" / "hole13_final.gpkg"

# Import the course polygons
zones = gpd.read_file(COURSE_FILE, layer="course_zones")

# Remove water areas that overlap with out of bounds

ob = zones[zones["zone_type"] == "ob"]
water_mask = zones["zone_type"] == "water"

# Combine all OB polygons into one geometry
ob_area = ob.geometry.union_all()

# Subtract OB areas from water polygons
zones.loc[water_mask, "geometry"] = zones.loc[
    water_mask, "geometry"
].apply(lambda geom: geom.difference(ob_area))

print("OB overlaps removed from water polygons!")

zones = zones.to_crs(epsg=32617) #cordinates to meters

METERS_TO_YARDS = 1.09361

zones_yards = zones.copy() #creata a copy for plotting in yards

# Convert geographic coordinates to meters
zones = zones.to_crs(epsg=32617)

# Find the bottom-left corner of the course map
min_x, min_y, max_x, max_y = zones.total_bounds

# Convert meters to yards and move origin to (0, 0)
METERS_TO_YARDS = 1.0936133

zones_yards = zones.copy()

zones_yards["geometry"] = zones.geometry.apply(     #this function makes the cordinates start from 0 and not based on geographic coordinates
    lambda geom: affinity.scale(
        affinity.translate(
            geom,
            xoff=-min_x,
            yoff=-min_y
        ),
        xfact=METERS_TO_YARDS,
        yfact=METERS_TO_YARDS,
        origin=(0, 0)
    )
)

def load_player_dispersion(file_name):
  # Load the player's shot data
    shots = load_player_data(file_name)

    # Calculate dispersion using typical direction
    tee_shots = calculate_dispersion_typdir(shots)

    # Apply the distance filter from analysis.py
    tee_shots = filter_tee_dispersion_by_distance(tee_shots)

    # Calculate average landing position
    center_x = tee_shots["offline_yards"].mean()
    center_y = tee_shots["down_hole_yards"].mean()

    # Calculate distance from each shot to the average
    tee_shots["distance_from_center"] = np.sqrt(
        (tee_shots["offline_yards"] - center_x)**2 +
        (tee_shots["down_hole_yards"] - center_y)**2
    )

    # Find the radius containing 90% of shots
    radius = tee_shots["distance_from_center"].quantile(0.90)

    # Keep the central 90%
    tee_shots = tee_shots[
        tee_shots["distance_from_center"] <= radius
    ].copy()

    print("Final dispersion shots:", len(tee_shots))
    print("Central 90% radius:", round(radius, 1), "yards")

    return tee_shots[
        ["offline_yards", "down_hole_yards"]
    ].copy()

#Function to position the disperson on the course map
def position_dispersion(dispersion, aim_x, aim_y, angle):
    #Center the dispersion around the average langing position
    offline = (
        dispersion["offline_yards"] - dispersion["offline_yards"].mean()
    )

    downhole = (
        dispersion["down_hole_yards"] - dispersion["down_hole_yards"].mean()
    )
    
    #rotate the dispersion to match the aim direction
    rotated = np.radians(angle)
    landing_x = aim_x + offline * np.cos(rotated) - downhole * np.sin(rotated)
    landing_y = aim_y + offline * np.sin(rotated) + downhole * np.cos(rotated)
    return landing_x, landing_y




print("Course coordinates converted to yards!")


# Print information about the imported data
print("Course zones imported successfully!")
print("Coordinate system:", zones.crs)
print("\nZone counts:")
print(zones["zone_type"].value_counts())

# Define colors for each course zone
zone_colors = {
    "fairway": "#50A83B",
    "green": "#176B35",
    "bunker": "#E6D2A0",
    "water": "#2878B5",
    "trees": "#E6C229",
    "ob": "#D62828",
    "tee": "#7ED957",
}

# Plot the imported polygons
fig, ax = plt.subplots(figsize=(12, 8))

for zone_type, group in zones_yards.groupby("zone_type"): #show in yards
    group.plot(
        ax=ax,
        color=zone_colors.get(zone_type, "gray"),
        edgecolor="black",
        linewidth=0.5,
        label=zone_type
    )

#I am finging the center of the tee box
tee_box = zones_yards[zones_yards["zone_type"] == "tee"]
tee_center = tee_box.geometry.centroid

# Plot the tee center as a red dot
ax.scatter(tee_center.x, tee_center.y, color="red", marker="o", label="Tee Center") 

#save the cordinates for future calculations
tee_center_coords = (tee_center.x.values[0], tee_center.y.values[0])

#print the coordinates of the tee center for clarity
print("Tee Center Coordinates (yards):", tee_center_coords[0], tee_center_coords[1])

#find the center of the green
green = zones_yards[zones_yards["zone_type"] == "green"]
green_center = green.geometry.centroid

#save the coordinates for future calculations
green_center_coords = (green_center.x.values[0], green_center.y.values[0])

#print the coordinates of the green center for clarity
print("Green Center Coordinates (yards):", green_center_coords[0], green_center_coords[1])

# Plot the green center as a blue dot
ax.scatter(green_center.x, green_center.y, color="blue", marker="o", label="Green Center")

#EXPERIMENT
#Trying to plot dispersion on course map
# Load Rory's tee-shot dispersion
dispersion = load_player_dispersion("rory_2025_shots.csv")

# Set an experimental aim point
aim_x = tee_center_coords[0]
aim_y = tee_center_coords[1] + 310

# Position the dispersion around the aim point
landing_x, landing_y = position_dispersion(
    dispersion,
    aim_x,
    aim_y,
    angle=0
)

# Plot the landing positions
ax.scatter(
    landing_x,
    landing_y,
    color="red",
    s=10,
    alpha=0.35,
    label="Rory Dispersion"
)

# Plot the aim point
ax.scatter(
    aim_x,
    aim_y,
    color="black",
    marker="x",
    s=100,
    label="Aim Point"
)

#calculating the distance from tee to green (as a straight line)
tee_to_green_distance_straight = tee_center_coords[0] - green_center_coords[0], tee_center_coords[1] - green_center_coords[1]
print("Distance from Tee to Green (yards):", tee_to_green_distance_straight)

ax.set_title("Augusta National - Hole 13 Course Zones")
ax.set_aspect("equal")
ax.legend(title="Course Zones")
ax.set_xlabel("Easting (yards)")
ax.set_ylabel("Northing (yards)")

plt.tight_layout()
plt.show()



