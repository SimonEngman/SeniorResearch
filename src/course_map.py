
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shapely import affinity
from shapely.geometry import Point
from matplotlib.patches import Ellipse

from analysis import (
    filter_tee_dispersion_by_distance,
    load_player_data,
    calculate_dispersion_typdir
)

# Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Path to the QGIS GeoPackage
COURSE_FILE = PROJECT_DIR / "data" / "course" / "hole13_final.gpkg"

#Load the course zones base on QGIS
def load_course_zones():
    # Import the course polygons
    zones = gpd.read_file(COURSE_FILE, layer="course_zones")

    #Make sure that no areas overlap
    # Remove water areas that overlap with out of bounds
    ob = zones[zones["zone_type"] == "ob"]
    water_mask = zones["zone_type"] == "water"

    # Combine all OB polygons into one geometry
    ob_area = ob.geometry.union_all()

    # Subtract OB areas from water polygons
    zones.loc[water_mask, "geometry"] = zones.loc[
        water_mask, "geometry"
    ].apply(lambda geom: geom.difference(ob_area))

    #fairway that overlaps with water should be cosidered fairway
    fairway = zones[zones["zone_type"] == "fairway"]
    #combine all fairway zones as one
    fairway_area = fairway.geometry.union_all()
    #Subtract water area from fairway
    zones.loc[water_mask, "geometry"] = zones.loc[
        water_mask, "geometry"
    ].apply(lambda geom: geom.difference(fairway_area))

    #Green that ovrlaps fairway should be considered Green
    green = zones[zones["zone_type"] == "green"]
    green_area = green.geometry.union_all()
    #Select the fairway polygons
    fairway_mask = zones["zone_type"] == "fairway"
    zones.loc[fairway_mask, "geometry"] = zones.loc[
        fairway_mask, "geometry"
    ].apply(lambda geom: geom.difference(green_area))

    #Bunker that overlaps fairway should be considered bunker
    bunker = zones[zones["zone_type"] == "bunker"]
    bunker_area = bunker.geometry.union_all()
    zones.loc[fairway_mask, "geometry"] = zones.loc[
        fairway_mask, "geometry"
    ].apply(lambda geom: geom.difference(bunker_area))

    #Fairway overlapping rough should be considered fairway
    rough = zones[zones["zone_type"] == "rough"]
    rough_area = rough.geometry.union_all()
    rough_mask = zones["zone_type"] == "rough"
    zones.loc[rough_mask, "geometry"] = zones.loc[
        rough_mask, "geometry"
    ].apply(lambda geom: geom.difference(fairway_area))

    #Trees overlapping rough should be considered trees
    trees = zones[zones["zone_type"] == "trees"]
    trees_area = trees.geometry.union_all()
    zones.loc[rough_mask, "geometry"] = zones.loc[
        rough_mask, "geometry"
    ].apply(lambda geom: geom.difference(trees_area))
    
    #Trees overlapping OB should be considered trees
    ob_mask = zones["zone_type"] == "ob"
    zones.loc[ob_mask, "geometry"] = zones.loc[
        ob_mask, "geometry"
    ].apply(lambda geom: geom.difference(trees_area))

    #Fairway overlapping OB should be considered fairway
    zones.loc[ob_mask, "geometry"] = zones.loc[
        ob_mask, "geometry"
    ].apply(lambda geom: geom.difference(fairway_area))

    #Green overlapping water should be considered green
    zones.loc[water_mask, "geometry"] = zones.loc[
        water_mask, "geometry"
    ].apply(lambda geom: geom.difference(green_area))


    print("Polygon overlapping have been removed")

    return zones
# Check whether any course polygons still overlap
def check_overlaps(zones):

    overlap_count = 0

    # Compare every polygon with the polygons after it
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):

            polygon1 = zones.geometry.iloc[i]
            polygon2 = zones.geometry.iloc[j]

            # Check if the polygons share an actual area
            overlap = polygon1.intersection(polygon2)

            if overlap.area > 0.000001:
                zone1 = zones["zone_type"].iloc[i]
                zone2 = zones["zone_type"].iloc[j]

                print(f"Overlap found: {zone1} and {zone2}")
                print(f"Overlap area: {overlap.area:.2f}")

                overlap_count += 1

    if overlap_count == 0:
        print("No overlapping polygon areas found")
    else:
        print(f"\nTotal overlapping polygon pairs: {overlap_count}")

#Convert from meters to yards
def convert_to_yards(zones):
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

    print("Coordinates have been convertede to yards")

    return zones_yards

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

#Count where all tee shots ended up
def classify_landing_zones(landing_x, landing_y, zones_yards):
    #List that will store where each shot land
    landing_results = []
    #Ranking priority of the zones
    #These priorities match how the overlapping of polygons is handeled
    priority = [
    "trees",
    "green",
    "bunker",
    "fairway",
    "ob",
    "water",
    "rough",
    "tee"
    ]

    #Check each shot
    for x, y in zip(landing_x, landing_y):
        point = Point(x, y)
        landing_zone = "other" #Default landing zone if no matching course polygon is found

        #loop through the course zones in priority order
        for zone_type in priority:
            #Select only the polygons that belong to the course zone
            polygons = zones_yards[
                zones_yards["zone_type"] == zone_type
            ]

            #Check if the shot lands in a zone
            #If it is, assign the landing zone the zone type
            if polygons.geometry.covers(point).any():
                landing_zone = zone_type
                break
            
        #Add the shot to the result list
        landing_results.append(landing_zone)

    return landing_results

# Find the center of a specific course zone
def get_zone_center(zones_yards, zone_type):

    # Select the polygons belonging to the zone
    selected_zone = zones_yards[
        zones_yards["zone_type"] == zone_type
    ]

    # Calculate the center of the zone
    center = selected_zone.geometry.union_all().centroid

    # Return the x and y coordinates
    return center.x, center.y

#Plot an ellipse around the 90% dispersion
def plot_dispersion_elipse(ax, landing_x, landing_y):
    #Find the center of the shot dispersion
    center_x = np.mean(landing_x)
    center_y = np.mean(landing_y)

        # Calculate how the shots are spread in X and Y
    covariance = np.cov(landing_x, landing_y)

    # Calculate the main directions of the dispersion
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort the directions from largest spread to smallest
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate the angle of the ellipse
    angle = np.degrees(
        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )

    # Scale for a theoretical 90% dispersion ellipse
    #4.605 comes from chi-square distribution
    #I am using a two dimensional distribution and want to include 90%
    #For 90%, the probability is 4.605 or below
    scale = np.sqrt(4.605) 

    # Calculate the width and height
    width = 2 * scale * np.sqrt(eigenvalues[0])
    height = 2 * scale * np.sqrt(eigenvalues[1])

    # Create the ellipse
    ellipse = Ellipse(
        xy=(center_x, center_y),
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        facecolor="none",
        linewidth=1,
        linestyle="--",
        label="Dispersion Ellipse"
    )

    # Add the ellipse to the course map
    ax.add_patch(ellipse)



# Plot the course map and player dispersion
def plot_course_map(
    zones_yards,
    tee_center_coords,
    green_center_coords,
    landing_x,
    landing_y,
    aim_x,
    aim_y,
    player_name
):

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

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the course polygons
    for zone_type, group in zones_yards.groupby("zone_type"):
        group.plot(
            ax=ax,
            color=zone_colors.get(zone_type, "gray"),
            edgecolor="black",
            linewidth=0.5,
            label=zone_type
        )

    # Plot the tee center
    ax.scatter(
        tee_center_coords[0],
        tee_center_coords[1],
        color="red",
        marker="o",
        label="Tee Center"
    )

    # Plot the green center
    ax.scatter(
        green_center_coords[0],
        green_center_coords[1],
        color="blue",
        marker="o",
        label="Green Center"
    )

    # Plot the landing positions
    ax.scatter(
        landing_x,
        landing_y,
        color="red",
        s=10,
        alpha=0.35,
        label=f"{player_name} Dispersion"
    )

    #Draw the ellipse around the dispersion
    plot_dispersion_elipse(ax, landing_x, landing_y)

    # Plot the target landing center
    ax.scatter(
        aim_x,
        aim_y,
        color="black",
        marker="x",
        s=100,
        label="Aim Point"
    )

    ax.set_title("Augusta National - Hole 13 Course Zones")
    ax.set_aspect("equal")
    ax.legend(title="Course Zones")
    ax.set_xlabel("Easting (yards)")
    ax.set_ylabel("Northing (yards)")

    plt.tight_layout()
    plt.show()


# Main function to run the course map analysis
def main():

    # Load the course zones from QGIS
    zones = load_course_zones()

    #Check for overlapping
    check_overlaps(zones)
    # Convert course coordinates to yards
    zones_yards = convert_to_yards(zones)

    # Print information about the course zones
    print("\nCourse zones imported successfully!")
    print("Coordinate system:", zones.crs)
    print("\nZone counts:")
    print(zones["zone_type"].value_counts())

    # Find the center of the tee box and green
    tee_center_coords = get_zone_center(zones_yards, "tee")
    green_center_coords = get_zone_center(zones_yards, "green")

    print("\nTee Center Coordinates (yards):", tee_center_coords)
    print("Green Center Coordinates (yards):", green_center_coords)

    # Select the player
    player_name = "Rory McIlroy"
    file_name = "rory_2025_shots.csv"

    # Load the player's central 90% tee-shot dispersion
    dispersion = load_player_dispersion(file_name)

    # Set an experimental landing center
    aim_x = tee_center_coords[0]
    aim_y = tee_center_coords[1] + 310

    # Position the dispersion on the course map
    landing_x, landing_y = position_dispersion(
        dispersion,
        aim_x,
        aim_y,
        angle=0
    )

    # Classify where each tee shot lands
    landing_zones = classify_landing_zones(
        landing_x,
        landing_y,
        zones_yards
    )

    # Count the shots in each landing zone
    from collections import Counter

    zone_counts = Counter(landing_zones)

    print("\n--- Tee Shot Landing Zones ---")

    for zone, count in zone_counts.items():
        percentage = count / len(landing_zones) * 100
        print(f"{zone}: {count} shots ({percentage:.2f}%)")

    # Calculate straight-line distance from tee to green
    tee_to_green_distance = np.sqrt(
        (tee_center_coords[0] - green_center_coords[0])**2 +
        (tee_center_coords[1] - green_center_coords[1])**2
    )

    print(
        "\nStraight-line Tee to Green Distance:",
        round(tee_to_green_distance, 2),
        "yards"
    )

    # Plot the course and tee-shot dispersion
    plot_course_map(
        zones_yards,
        tee_center_coords,
        green_center_coords,
        landing_x,
        landing_y,
        aim_x,
        aim_y,
        player_name
    )

if __name__ == "__main__":
    main()
