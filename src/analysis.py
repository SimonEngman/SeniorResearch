import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

#Player data
DATA_DIR = Path(__file__).parent.parent / "data"

players = {
    "Rory McIlroy": "rory_2025_shots.csv",
    "Collin Morikawa": "collin_2025_shots.csv"
}

#Load the player data
def load_player_data(file_name):
    shots = pd.read_csv(DATA_DIR / file_name)
    print("\n--- DATASET COLUMNS ---")
    print(shots.columns.tolist())

    print("\n--- POSSIBLE ROUND COLUMNS ---")
    print([
        col for col in shots.columns
        if "round" in col.lower()
    ])

    print("\n--- POSSIBLE DISTANCE COLUMNS ---")
    print([
        col for col in shots.columns
        if "distance" in col.lower()
        or "yard" in col.lower()
    ])
    print (f"Loaded {len(shots)} shots from {file_name}")
    print("Total shots:", len(shots))
    return shots

#Filter tee shots (remove par 3s and shots that are not the first shot of the hole)
def filter_tee_shots(shots):
    tee_shots = shots[
        (shots["stroke_number"] == 1) &
        (shots["par"] != 3)
    ].copy()

    return tee_shots

#Calculate dispersion
def calculate_dispersion(shots):
    tee_shots = filter_tee_shots(shots)

    #Organize colums
    from_x = "leftToRightCoords.fromCoords.x"
    from_y = "leftToRightCoords.fromCoords.y"

    to_x = "leftToRightCoords.toCoords.x"
    to_y = "leftToRightCoords.toCoords.y"

    #Remove all shots with missing coordinates
    tee_shots = tee_shots.dropna(subset=[from_x, from_y, to_x, to_y])

    #get all shots on par 4 and par 5 holes
    all_shots = shots[shots["par"] != 3]

    #identify each hole separately
    group_cols = ["tournament_id", "hole_number", "round_number"]

    # Find the final position of each hole
    hole_ends = (
        all_shots[all_shots["final_stroke"] == True]
        .sort_values(group_cols + ["stroke_number"])
        .groupby(group_cols)
        .tail(1)
    )

    # Keep endpoint coordinates
    hole_ends = hole_ends[
        group_cols + [to_x, to_y]
    ].rename(
        columns={
            to_x: "hole_end_x",
            to_y: "hole_end_y"
        }
    )

    # Attach the hole endpoint to each tee shot
    tee_shots = tee_shots.merge(
        hole_ends,
        on=group_cols,
        how="left",
        validate="many_to_one"
    )

    # Remove shots without hole endpoint
    tee_shots = tee_shots.dropna(
        subset=["hole_end_x", "hole_end_y"]
    )

    # Tee position
    tee_x = tee_shots[from_x].to_numpy()
    tee_y = tee_shots[from_y].to_numpy()

    # Hole endpoint
    end_x = tee_shots["hole_end_x"].to_numpy()
    end_y = tee_shots["hole_end_y"].to_numpy()

    # Direction from tee to hole
    hole_dx = end_x - tee_x
    hole_dy = end_y - tee_y

    # Hole length in coordinate units
    hole_length = np.sqrt(
        hole_dx**2 + hole_dy**2
    )

    # Remove invalid hole directions
    valid = np.isfinite(hole_length) & (hole_length > 0)

    tee_shots = tee_shots.iloc[np.flatnonzero(valid)].copy()
    hole_dx = hole_dx[valid]
    hole_dy = hole_dy[valid]
    hole_length = hole_length[valid]

    # Unit vector down the hole
    unit_x = hole_dx / hole_length
    unit_y = hole_dy / hole_length

    # Diagnostic: inspect hole direction
    tee_shots["hole_direction_degrees"] = np.degrees(
        np.arctan2(unit_y, unit_x)
    )

    problem = tee_shots[
        (tee_shots["tournament_id"] == "R2025009") &
        (tee_shots["hole_number"] == 6)
    ]

    print("\n--- HOLE 6 DIRECTION CHECK ---")

    print(
        problem[
            [
                "round_number",
                "hole_direction_degrees"
            ]
        ].to_string(index=False)
    )

    # Tee shot landing position
    shot_x = tee_shots[to_x].to_numpy()
    shot_y = tee_shots[to_y].to_numpy()

    shot_dx = shot_x - tee_shots[from_x].to_numpy()
    shot_dy = shot_y - tee_shots[from_y].to_numpy()

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

    # Convert coordinate units to yards
    # Use actual recorded shot distance
    shot_distance = pd.to_numeric(
        tee_shots["distance"]
        .astype(str)
        .str.replace(" yds", "", regex=False),
        errors="coerce"
    ).to_numpy()

    # Length of tee shot vector in coordinate units
    shot_length = np.sqrt(
        shot_dx**2 + shot_dy**2
    )

    # Convert coordinates into yards using actual shot distance
    scale = np.divide(
        shot_distance,
        shot_length,
        out=np.full_like(shot_length, np.nan, dtype=float),
        where=shot_length > 0
    )

    tee_shots["down_hole_yards"] = (
        down_hole_coordinate * scale
    )

    tee_shots["offline_yards"] = (
        offline_coordinate * scale
    )

    # Remove invalid results
    tee_shots = tee_shots.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["down_hole_yards", "offline_yards"]
    )

    print("Valid tee shots:", len(tee_shots))
    print("\n--- LARGEST OFFLINE SHOTS ---")

    print(
        tee_shots.assign(
            absolute_offline=tee_shots["offline_yards"].abs()
        )
        .sort_values("absolute_offline", ascending=False)
        [
            [
                "tournament_id",
                "hole_number",
                "yardage",
                "down_hole_yards",
                "offline_yards"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return tee_shots

#NEXPERIMENT function on typical direction
def calculate_typical_direction(tee_shots, from_x, from_y, to_x, to_y):
    tee_shots = tee_shots.copy()
    #calculate the direction of each tee shot
    tee_shots["shot_dx"] = tee_shots[to_x] - tee_shots[from_x]
    tee_shots["shot_dy"] = tee_shots[to_y] - tee_shots[from_y]

    # Group shots from the same tournament and hole
    group_cols = ["tournament_id", "hole_number"]

    #calculate the average direction for each hole
    tee_shots["avg_dx"] = tee_shots.groupby(group_cols)["shot_dx"].transform("mean") 
    tee_shots["avg_dy"] = tee_shots.groupby(group_cols)["shot_dy"].transform("mean")

    # Calculate length of average direction
    avg_length = np.sqrt(
        tee_shots["avg_dx"]**2 +
        tee_shots["avg_dy"]**2
    )

    #convert the average direction into a unit vector
    tee_shots["unit_x"] = tee_shots["avg_dx"] / avg_length
    tee_shots["unit_y"] = tee_shots["avg_dy"] / avg_length

    return tee_shots

# Calculate dispersion using typical shot direction - EXPERIMENT
def calculate_dispersion_typdir(shots):

    tee_shots = filter_tee_shots(shots).copy()

    # Coordinate columns
    from_x = "leftToRightCoords.fromCoords.x"
    from_y = "leftToRightCoords.fromCoords.y"
    to_x = "leftToRightCoords.toCoords.x"
    to_y = "leftToRightCoords.toCoords.y"

    # Remove shots with missing coordinates
    tee_shots = tee_shots.dropna(
        subset=[from_x, from_y, to_x, to_y]
    )

    # Calculate typical direction using our separate function
    tee_shots = calculate_typical_direction(
        tee_shots, from_x, from_y, to_x, to_y
    )

    # Exclude holes with fewer than two tee shots
    group_cols = ["tournament_id", "hole_number"]

    group_size = tee_shots.groupby(group_cols)["shot_dx"].transform("count")
    tee_shots = tee_shots[group_size >= 2].copy()

    # Remove invalid direction vectors
    tee_shots = tee_shots.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["unit_x", "unit_y"])

    # Calculate distance down the typical direction
    down_hole_coordinate = (
        tee_shots["shot_dx"] * tee_shots["unit_x"] +
        tee_shots["shot_dy"] * tee_shots["unit_y"]
    )

    # Calculate distance offline from the typical direction
    offline_coordinate = (
        -tee_shots["shot_dx"] * tee_shots["unit_y"] +
        tee_shots["shot_dy"] * tee_shots["unit_x"]
    )

    # Get recorded shot distance in yards
    shot_distance = pd.to_numeric(
        tee_shots["distance"]
        .astype(str)
        .str.replace(" yds", "", regex=False),
        errors="coerce"
    )

    # Calculate shot vector length
    shot_length = np.sqrt(
        tee_shots["shot_dx"]**2 +
        tee_shots["shot_dy"]**2
    )

    # Convert coordinate units to yards
    scale = shot_distance / shot_length.replace(0, np.nan)

    tee_shots["down_hole_yards"] = down_hole_coordinate * scale
    tee_shots["offline_yards"] = offline_coordinate * scale

    # Remove invalid results
    tee_shots = tee_shots.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["down_hole_yards", "offline_yards"]
    )

    print("Valid tee shots (typical direction):", len(tee_shots))

    return tee_shots


# Calculate player statistics
def calculate_statistics(tee_shots, player_name):

    avg_down_hole = tee_shots["down_hole_yards"].mean()
    avg_offline = tee_shots["offline_yards"].mean()

    std_down_hole = tee_shots["down_hole_yards"].std()
    std_offline = tee_shots["offline_yards"].std()

    print(f"\n--- {player_name} ---")
    print("Average down-hole distance:",
          round(avg_down_hole, 1), "yards")
    print("Average offline distance:",
          round(avg_offline, 1), "yards")
    print("Down-hole standard deviation:",
          round(std_down_hole, 1), "yards")
    print("Offline standard deviation:",
          round(std_offline, 1), "yards")
    
#I added this function so that I can import the variable to optimization.py
def calculate_average_distance(tee_shots):
    return tee_shots["down_hole_yards"].mean()


# Plot full tee-shot dispersion
def plot_full_dispersion(tee_shots, player_name):

    plt.figure(figsize=(8, 10))

    plt.scatter(
        tee_shots["offline_yards"],
        tee_shots["down_hole_yards"],
        alpha=0.5
    )

    # Hole centerline
    plt.axvline(0, color="black", linewidth=1)

    plt.xlabel("Offline distance (yards)")
    plt.ylabel("Distance down hole (yards)")
    plt.title(f"{player_name} - Tee Shot Dispersion")

    plt.grid(True)
    plt.show()


# Plot central 90% tee-shot dispersion
def plot_central_dispersion(tee_shots, player_name):

    # Average landing position
    center_x = tee_shots["offline_yards"].mean()
    center_y = tee_shots["down_hole_yards"].mean()

    data = tee_shots.copy()

    # Distance from average landing position
    data["distance_from_center"] = np.sqrt(
        (data["offline_yards"] - center_x)**2 +
        (data["down_hole_yards"] - center_y)**2
    )

    # Find radius containing closest 70%
    radius = data["distance_from_center"].quantile(0.90)

    # Keep closest 70%
    middle_70 = data[
        data["distance_from_center"] <= radius
    ]

    print(
        f"{player_name} - Central 70% radius:",
        round(radius, 1), "yards"
    )

    # Plot
    plt.figure(figsize=(8, 8))

    plt.scatter(
        middle_70["offline_yards"],
        middle_70["down_hole_yards"],
        alpha=0.5
    )

    # Centerline
    plt.axvline(0, color="black", linewidth=1)

    # Average landing position
    plt.scatter(
        center_x,
        center_y,
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        label="Average landing position"
    )

    plt.xlabel("Offline distance (yards)")
    plt.ylabel("Distance down hole (yards)")
    plt.title(f"{player_name} - Central 90% Dispersion")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def filter_tee_dispersion_by_distance(tee_shots):
    filtered = tee_shots[
        (tee_shots["down_hole_yards"] >= 280) &
        (tee_shots["down_hole_yards"] <= 340) &
        (tee_shots["offline_yards"].abs() <= 70)
    ]
    print(f"Filtered tee shots: {len(filtered)}")
    print(f"shots after filtering: {len(filtered)}")
    print("shots excluded: ", len(tee_shots) - len(filtered))

    return filtered

#All functions below are for creating dispersion on approach shots
def filter_approach_shots(shots):
    approach_shots = shots.copy()

    # Convert distances to yards
    def to_yards(value):
        if pd.isna(value):
            return np.nan

        value = str(value).lower().strip()

        try:
            if "yd" in value:
                return float(value.split()[0])
            elif "ft" in value:
                return float(value.split()[0]) / 3
            elif "in" in value:
                return float(value.split()[0]) / 36
            else:
                return float(value)
        except ValueError:
            return np.nan

    approach_shots["remaining_yards"] = (
        approach_shots["distance_remaining"].apply(to_yards)
    )

    approach_shots["hole_yards"] = (
        approach_shots["yardage"].apply(to_yards)
    )

    # Identify each hole in each round
    group_cols = [
        "tournament_id",
        "round_number",
        "hole_number"
    ]

    # Sort shots in playing order
    approach_shots = approach_shots.sort_values(
        group_cols + ["stroke_number"]
    )

    # Previous shot's remaining distance = current starting distance
    approach_shots["starting_distance"] = (
        approach_shots.groupby(group_cols)["remaining_yards"].shift(1)
    )

    # Par 3 tee shots start approximately at hole yardage
    par3_tee = (
        (approach_shots["par"] == 3) &
        (approach_shots["stroke_number"] == 1)
    )

    approach_shots.loc[par3_tee, "starting_distance"] = (
        approach_shots.loc[par3_tee, "hole_yards"]
    )

    # Keep approach shots and par 3 tee shots
    approach_shots = approach_shots[
        (approach_shots["stroke_number"] > 1) | par3_tee
    ].copy()

    # Keep starting distances from 50 to 200 yards
    approach_shots = approach_shots[
        (approach_shots["starting_distance"] >= 50) &
        (approach_shots["starting_distance"] < 200)
    ].copy()

    return approach_shots

def group_approach_shots(approach_shots):

    bins = list(range(50, 201, 25))

    approach_shots = approach_shots.copy()

    approach_shots["distance_group"] = pd.cut(
        approach_shots["starting_distance"],
        bins=bins,
        right=False
    )

    group_counts = (
        approach_shots["distance_group"]
        .value_counts()
        .sort_index()
    )

    return group_counts

def get_hole_target(shots):
    shots = shots.copy()

    #Identify each individual hole played
    group_cols = [
        "tournament_id",
        "round_number",
        "hole_number"
    ]

    #coordinate columns
    to_x = "leftToRightCoords.toCoords.x"
    to_y = "leftToRightCoords.toCoords.y"

        # Find the final stroke on each hole
    final_shots = shots[
        shots["final_stroke"] == True
    ].copy()

    # Keep the final stroke if duplicates exist
    final_shots = (
        final_shots
        .sort_values(group_cols + ["stroke_number"])
        .groupby(group_cols)
        .tail(1)
    )
        # Save the estimated target coordinates
    targets = final_shots[
        group_cols + [to_x, to_y]
    ].rename(columns={
        to_x: "target_x",
        to_y: "target_y"
    })

    return targets
def test_approach_targets(approach_shots, targets):

    group_cols = [
        "tournament_id",
        "round_number",
        "hole_number"
    ]

    # Match each approach shot with its hole target
    test = approach_shots.merge(
        targets,
        on=group_cols,
        how="left",
        validate="many_to_one"
    )

    from_x = "leftToRightCoords.fromCoords.x"
    from_y = "leftToRightCoords.fromCoords.y"

    # Calculate coordinate distance to target
    test["coordinate_distance"] = np.sqrt(
        (test["target_x"] - test[from_x])**2 +
        (test["target_y"] - test[from_y])**2
    )

    print("APPROACH TARGET CHECK:")
    print(
        test[
            [
                "starting_distance",
                "coordinate_distance",
                "target_x",
                "target_y"
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nMissing targets:",
        test["target_x"].isna().sum()
    )

    return test

#this function converst string values to numeric values in yards
def convert_to_yards(value):
    if pd.isna(value):
        return np.nan
    
    text = str(value).lower().strip()

    yards = re.search(r"([\d.]+)\s*yd", text)
    feet = re.search(r"([\d.]+)\s*ft", text)
    inches = re.search(r"([\d.]+)\s*in", text)

    #converts to yards regardless of the unit
    if yards or feet or inches:
        return (
            (float(yards.group(1)) if yards else 0)
            + (float(feet.group(1)) / 3 if feet else 0)
            + (float(inches.group(1)) / 36 if inches else 0)
        )
    
    try:
        return float(text)
    except ValueError:
        return np.nan
    
#Calculate approach shot dispersion relative to the estimated target position
def calculate_approach_dispersion(approach_shots, targets):

    group_cols = [
        "tournament_id",
        "round_number",
        "hole_number"
    ]

    # Match each shot to its estimated flag position
    data = approach_shots.merge(
        targets,
        on=group_cols,
        how="left",
        validate="many_to_one"
    )

    # Convert distances to yards
    data["shot_yards"] = data["distance"].apply(convert_to_yards)
    data["remaining_yards"] = data["distance_remaining"].apply(convert_to_yards)

    # Triangle sides:
    # a = starting distance to flag
    # b = shot distance
    # c = remaining distance to flag
    a = data["starting_distance"]
    b = data["shot_yards"]
    c = data["remaining_yards"]

    # Distance traveled along the target line
    down_target = (a**2 + b**2 - c**2) / (2 * a)

    # Positive = long, negative = short
    data["distance_error_yards"] = down_target - a

    # Perpendicular distance from the target line
    offline_squared = b**2 - down_target**2

    #This could lead to some shots being straight even though they are offline
    #Therefore, I have commented it out for now
    """data["offline_yards"] = np.sqrt(
        np.maximum(offline_squared, 0)
    ) """
    #The offline distance is now calculated in the next step, but it will be flagged if the triangle is invalid
    # Flag shots with inconsistent distance measurements
    data["invalid_triangle"] = offline_squared < -1e-6

    # Keep invalid measurements as "not a number" instead of forcing zero
    data["offline_yards"] = np.sqrt(
        offline_squared.where(offline_squared >= -1e-6).clip(lower=0)
    )

    # Determine which side of the target line the shot finished
    from_x = "leftToRightCoords.fromCoords.x"
    from_y = "leftToRightCoords.fromCoords.y"
    to_x = "leftToRightCoords.toCoords.x"
    to_y = "leftToRightCoords.toCoords.y"

    target_dx = data["target_x"] - data[from_x]
    target_dy = data["target_y"] - data[from_y]

    shot_dx = data[to_x] - data[from_x]
    shot_dy = data[to_y] - data[from_y]

    # Cross product determines left or right
    cross = target_dx * shot_dy - target_dy * shot_dx

    # Positive = left, negative = right in a standard x/y coordinate system
    data["offline_yards"] *= np.sign(cross)

    # Remove rows with missing measurements
    data = data.replace([np.inf, -np.inf], np.nan)

    print("Invalid triangles:", data["invalid_triangle"].sum())

    data = data.dropna(
        subset=[
            "offline_yards",
            "distance_error_yards"
        ]
    )

    return data

def classify_approach_shots(approach_shots):

    data = approach_shots.copy()

    # Standardize text to make comparisons easier
    data["from_location_clean"] = (
        data["from_location"].fillna("").astype(str).str.lower()
    )

    data["to_location_clean"] = (
        data["to_location"].fillna("").astype(str).str.lower()
    )

    data["stroke_type_clean"] = (
        data["stroke_type"].fillna("").astype(str).str.lower()
    )

    def classify(row):

        start = row["from_location_clean"]
        end = row["to_location_clean"]
        stroke = row["stroke_type_clean"]

        # Explicitly identified recovery shots
        if "recovery" in stroke:
            return "Recovery"

        # Shots that finished on the green
        if "green" in end:
            return "Green approach"

        # Shots that finished in a bunker
        if "bunker" in end or "sand" in end:
            return "Bunker finish"

        # Shots that finished in the rough
        if "rough" in end:
            return "Rough finish"

        # Shots that finished on the fairway
        if "fairway" in end:
            return "Fairway finish"

        return "Other"

    data["shot_category"] = data.apply(classify, axis=1)

    return data
def classify_shot_intent(approach_shots):

    data = approach_shots.copy()

    # Rule 1: Shot finishes more than 50% of starting distance offline
    offline_ratio = (
        data["offline_yards"].abs()
        / data["starting_distance"]
    )

    # Rule 2: Shot makes less than 70% progress toward the target
    # Progress = starting distance - remaining distance
    progress_ratio = (
        (data["starting_distance"] - data["remaining_yards"])
        / data["starting_distance"]
    )

    # Save ratios for future analysis
    data["offline_ratio"] = offline_ratio
    data["progress_ratio"] = progress_ratio

    # Identify unusual shots
    data["potential_recovery"] = (
        (offline_ratio > 0.50)
        | (progress_ratio < 0.70)
    )

    # Assign classification
    data["shot_intent"] = np.where(
        data["potential_recovery"],
        "Potential recovery/layup",
        "Standard approach candidate"
    )

    return data

#EXPERIMENT - start to plot for group 150 yds - 175yds
def plot_approach_dispersion(approach_dispersion, player_name, min_distance=150, max_distance=175):

   #Filter the selected distance range
   #remove potential recovery shots from the analysis
    data = approach_dispersion[
        (approach_dispersion["starting_distance"] >= min_distance) &
        (approach_dispersion["starting_distance"] < max_distance) &
        (~approach_dispersion["potential_recovery"])
    ].copy()

    if data.empty:
        print(f"No approach shots found for {player_name}")
        return

    # Create scatter plot
    plt.figure(figsize=(8, 8))

    plt.scatter(
        data["offline_yards"],
        data["distance_error_yards"],
        alpha=0.5,
        s=25,
        color="steelblue"
    )

    # Mark the aim at (0, 0)
    plt.scatter(
        0, 0,
        color="red",
        marker="x",
        s=120,
        linewidths=3,
        label="Aim"
    )

    # Reference lines
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.title(
        f"{player_name} - Approach Dispersion "
        f"({min_distance}-{max_distance} yards)"
    )

    plt.xlabel("Offline Distance (yards)")
    plt.ylabel("Distance Error (yards)")

    # Use the same scale on both axes
    plt.axis("equal")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Function that plots the central 90% of approach shots for a given distance range
def plot_central_approach_dispersion(approach_dispersion, player_name, min_distance, max_distance):

    # Filter the selected distance range
    #remove the potential recovery shots from the analysis
    data = approach_dispersion[
        (approach_dispersion["starting_distance"] >= min_distance) &
        (approach_dispersion["starting_distance"] < max_distance) &
        (~approach_dispersion["potential_recovery"])
    ].copy()

    #Look if there are any shots in the selected range
    if data.empty:
        print(f"No approach shots found for {player_name}")
        return
    
        # Calculate distance of each shot from the hole
    data["total_error"] = np.sqrt(
        data["offline_yards"]**2 +
        data["distance_error_yards"]**2
    )

    # Find the 90th percentile
    threshold = data["total_error"].quantile(0.90)

    # Keep central 90% for visualization
    central = data[
        data["total_error"] <= threshold
    ]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        central["offline_yards"],
        central["distance_error_yards"],
        alpha=0.5,
        s=25,
        color="steelblue"
    )

    #Mark aim at 0, 0
    plt.scatter(
        0, 0,
        color="red",
        marker="x",
        s=120,
        linewidths=3,
        label="Aim"
    )

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.title(
        f"{player_name} - Central 90% Approach Dispersion "
        f"({min_distance}-{max_distance} yards)"
    )

    plt.xlabel("Offline Distance (yards)")
    plt.ylabel("Distance Error (yards)")

    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n--- {player_name} Central 90% ---")
    print("Original shots:", len(data))
    print("Shots in central 90%:", len(central))
    print("90% radius:", round(threshold, 2), "yards")

#Calculate statistics on the dispersions for each approach group
def calculate_approach_group_statistics(approach_dispersion):
    # Keep standard approach candidates
    data = approach_dispersion[
        ~approach_dispersion["potential_recovery"]
    ].copy()

    # Calculate actual proximity to the hole
    data["proximity_yards"] = data["remaining_yards"]

    # Create 25-yard distance groups
    bins = list(range(50, 201, 25))

    data["distance_group"] = pd.cut(
        data["starting_distance"],
        bins=bins,
        right=False
    )

    # Calculate statistics for each group
    statistics = data.groupby(
        "distance_group",
        observed=True
    ).agg(
        shots=("offline_yards", "count"),
        mean_offline=("offline_yards", "mean"),
        offline_sd=("offline_yards", "std"),
        mean_distance_error=("distance_error_yards", "mean"),
        distance_sd=("distance_error_yards", "std"),
        average_proximity=("proximity_yards", "mean")
    )

    return statistics.round(2)

# Main program
def main():
    for player_name, file_name in players.items():

        shots = load_player_data(file_name)
        """
        # Calculate dispersion using typical direction
        tee_shots = calculate_dispersion_typdir(shots)

        tee_shots_filtered = filter_tee_dispersion_by_distance(tee_shots)

        if len(tee_shots) < 2:
            print(f"Not enough valid tee shots for {player_name}")
            continue

        # Calculate statistics
        print("Standard tee shot dispersion statistics:")
        calculate_statistics(tee_shots, player_name)

        print("Filtered tee shot dispersion statistics:")
        calculate_statistics(tee_shots_filtered, player_name)

        # Plot full dispersion
        plot_full_dispersion(tee_shots_filtered, player_name)

        # Plot central 90% dispersion on the filtered data
        plot_central_dispersion(tee_shots_filtered, player_name)
        """
        # Approach shot analysis
        approach_shots = filter_approach_shots(shots)

        group_counts = group_approach_shots(approach_shots)

        print(f"\n--- {player_name} Approach Shot Groups ---")
        print(group_counts)

        print("\nExample approach shots:")
        print(
            approach_shots[
                [
                    "hole_number",
                    "par",
                    "stroke_number",
                    "starting_distance",
                    "distance",
                    "distance_remaining"
                ]
            ].head(10).to_string(index=False)
        )
        targets = get_hole_target(shots)

        print(f"\n--- {player_name} Hole Targets ---")
        print(targets.head(10).to_string(index=False))

        print("Total hole targets:", len(targets))

        test = test_approach_targets(approach_shots, targets)

        approach_dispersion = calculate_approach_dispersion(
            approach_shots,
            targets
        )             

        print(f"\n--- {player_name} Approach Dispersion ---")

        print(
            approach_dispersion[
                [
                    "starting_distance",
                    "shot_yards",
                    "remaining_yards",
                    "offline_yards",
                    "distance_error_yards"
                ]
            ].head(15).to_string(index=False)
        )

        print("\nDispersion statistics:")
        print(
            approach_dispersion[
                ["offline_yards", "distance_error_yards"]
            ].describe()
        )

        print("Shots analyzed:", len(approach_dispersion))

        approach_dispersion = calculate_approach_dispersion(
        approach_shots, targets
        )

        # Classify the shots
        approach_dispersion = classify_approach_shots(
            approach_dispersion
        )

        print(f"\n--- {player_name} Shot Categories ---")
        print(approach_dispersion["shot_category"].value_counts())
        approach_dispersion = calculate_approach_dispersion(
        approach_shots, targets
            )

        approach_dispersion = classify_shot_intent(
            approach_dispersion
        )
        # Calculate statistics for each approach distance group
        group_statistics = calculate_approach_group_statistics(
            approach_dispersion
        )

        print(f"\n--- {player_name} Approach Group Statistics ---")
        print(group_statistics.to_string())

        #This is just an experiment to see if we can identify potential recovery shots based on distance and offline metrics
        print(f"\n--- {player_name} Shot Classification ---")
        print(approach_dispersion["shot_intent"].value_counts())
        print("POTENTIAL RECOVERY SHOTS:")

        print(
            approach_dispersion[
                approach_dispersion["potential_recovery"]
            ][
                [
                    "starting_distance",
                    "shot_yards",
                    "remaining_yards",
                    "offline_yards",
                    "progress_ratio"
                ]
            ].head(15).to_string(index=False)
        )
        # Compare dispersion with and without flagged shots
        standard_approaches = approach_dispersion[
        ~approach_dispersion["potential_recovery"]
        ]

        print(f"\n--- {player_name} Standard Approach Statistics ---")

        print(
            standard_approaches[
                ["offline_yards", "distance_error_yards"]
            ].describe()
        )
        # Classify potential recovery shots
        approach_dispersion = classify_shot_intent(
            approach_dispersion
        )

        # Plot approach dispersion from 150-175 yards
        """plot_approach_dispersion(
            approach_dispersion,
            player_name,
            min_distance=150,
            max_distance=175
        ) """

        # Plot central 90% approach dispersion from 150-175 yards
        """plot_central_approach_dispersion(
            approach_dispersion,
            player_name,
            min_distance=150,
            max_distance=175
        )"""

        #Plot all different distance groups
        #6 groups from 50 - 200 yds, 25 yds in between each
        for min_distance in range(50, 200, 25):
            max_distance = min_distance + 25
            #Plot the central 90% dispersion for each group
            plot_central_approach_dispersion(approach_dispersion, player_name, min_distance, max_distance)



if __name__ == "__main__":
    main()


