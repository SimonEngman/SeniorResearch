
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

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


from shapely import affinity

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

ax.set_title("Augusta National - Hole 13 Course Zones")
ax.set_aspect("equal")
ax.legend(title="Course Zones")
ax.set_xlabel("Easting (yards)")
ax.set_ylabel("Northing (yards)")

plt.tight_layout()
plt.show()



