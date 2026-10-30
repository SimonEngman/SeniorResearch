from collections import Counter
import numpy as np
import pandas as pd

from course_map import (
    position_dispersion,
    classify_landing_zones,
    load_course_zones,
    convert_to_yards,
    get_zone_center,
    load_player_dispersion,
    plot_course_map
)

from analysis import(
    calculate_average_distance,
    load_player_data,
    filter_tee_dispersion_by_distance,
    calculate_dispersion_typdir,
    filter_approach_shots,
    group_approach_shots,
    calculate_approach_dispersion,
    classify_shot_intent,
    get_hole_target
)

#Evaluate the landing zone for a given aimpoint
def evaluate_aimpoint(dispersion, aim_x, aim_y, angle, zones_yards):
    #Position the players dispersion on the course
    landing_x, landing_y = position_dispersion(dispersion, aim_x, aim_y, angle)
    
    #Determine in what zone the shots land
    landing_zones = classify_landing_zones(landing_x, landing_y, zones_yards)

    # Count shots in each landing zone
    zone_counts = Counter(landing_zones)

    # Calculate percentage of shots in each zone
    total_shots = len(landing_zones)

    zone_percentages = {}

    for zone, count in zone_counts.items():
        zone_percentages[zone] = count / total_shots * 100

    return zone_percentages

#Function that creates a weigthed score to evaluate a tee shot aimpoint
def evaluate_score(zones_percentages):
    fairway = zones_percentages.get("fairway", 0)
    rough = zones_percentages.get("rough", 0)
    trees = zones_percentages.get("trees", 0)
    water = zones_percentages.get("water", 0)
    ob = zones_percentages.get("ob", 0)

    #Calculate the score with risk-adjustements
    score = (
        fairway
        -0.2 * rough
        -0.4 * trees #Trees on augusta hole 13 is not overly bad
        -2 * water # water is a huge disadvantage
        -3.5 * ob #Ob should almost never occur
    )

    return score

#Calculate the remaining distance to the hole afte tee shot
def calculate_remaining_distance(landing_x, landing_y, green_x, green_y):
    remaining_distances = np.sqrt(
        (green_x - landing_x)**2
        + (green_y - landing_y)**2
    )
    return remaining_distances

#Returns taht approach can be played only if ball is in rough ot fairway
def can_play_approach(ladning_zone):
    if ladning_zone in ["fairway", "rough"]:
        return True #Approach can be played

    return False #Approach can't be played

def select_approach_group(remaining_distance):
    #We are using these groupings in analysis.py
    bins = [
        (50, 75),
        (75, 100), 
        (100, 125), 
        (125, 150), 
        (150, 175), 
        (175, 200), 
        (200, 225), 
        (225, 250), 
    ]

    for lower, upper in bins:
        if lower <= remaining_distance < upper:
            return(lower, upper) #returns the lower and upper limit of the group
    return None #shot could not be classified to a group

#Returns the approach situation
def get_approach_situations(landing_x, landing_y, landing_zones, remaining_distances):
    approach_situations = []

    for i in range(len(landing_x)):
        zone = landing_zones[i]
        distance = remaining_distances.iloc[i]

        if can_play_approach(zone): #make sure the shot can be played towards the green
            approach_group = select_approach_group(distance)

            if approach_group is not None: #make sure the zone exist
                approach_situations.append({ #adds the sitation to the array
                    "landing_x": landing_x.iloc[i],
                    "landing_y": landing_y.iloc[i],
                    "remaining_distance": distance,
                    "landing_zone": zone,
                    "approach_group": approach_group
                })

    return approach_situations

#Function is simulation the approach shots
def simulate_approach(
    landing_x,
    landing_y,
    approach_group,
    approach_dispersion,
    remaining_distance,
    green_x,
    green_y
):
    # Select historical approach shots from the correct distance group
    lower, upper = approach_group

    group_data = approach_dispersion[
        (approach_dispersion["starting_distance"] >= lower) &
        (approach_dispersion["starting_distance"] < upper) &
        (~approach_dispersion["potential_recovery"])
    ]

    # Calculate direction from the ball to the green
    dx = green_x - landing_x
    dy = green_y - landing_y

    # Convert direction into a unit vector
    forward_x = dx / remaining_distance
    forward_y = dy / remaining_distance

    # Create a vector pointing right
    right_x = forward_y
    right_y = -forward_x

    # Get historical approach errors for the selected group
    # Negative sign converts left-positive errors to right-positive
    lateral = -group_data["offline_yards"].to_numpy()
    longitudinal = group_data["distance_error_yards"].to_numpy()

    # Position historical dispersion around the green target
    approach_x = green_x + lateral * right_x + longitudinal * forward_x
    approach_y = green_y + lateral * right_y + longitudinal * forward_y

    return approach_x, approach_y

def main():
    zones = load_course_zones()
    zones_yards = convert_to_yards(zones)

    tee_x, tee_y = get_zone_center(zones_yards, "tee")

    dispersion = load_player_dispersion("rory_2025_shots.csv")

    # Original experimental landing center
    # Use Rory's tee-shot data
    shots = load_player_data("rory_2025_shots.csv")

    tee_shots = calculate_dispersion_typdir(shots)
    tee_shots = filter_tee_dispersion_by_distance(tee_shots)

    average_distance = calculate_average_distance(tee_shots)

    print(f"Rory's average distance: {average_distance:.2f} yards")

    # Test different horizontal offsets
    #THIS OFFSET IS USED FOR COMPLETE CALCULATION
    #offsets = list(range(-40, 41)) 
    #THIS OFFSET IS USED DURING DEVELOPMENT FOR FASTER RUN TIME
    offsets = [-30, -20, -10, 10, 20, 30]

    #initialize most_fairwy_percentage to 0%
    most_fairway_percentage = -1

    #This variable will store the offset that produced the highest fairway percentage
    best_offset_fairway_hit = None

    #Varible for the angle with the best aim (fairway_percentage)
    best_aim_angle_fairway = None

    #variable for best eval score
    best_eval_score = -1000 #Can't be a lower score than this

    #Variable for best eval_score offset
    best_evalscore_offset = None

    # Varible for the angle with the best aim (eval_score)
    best_aim_angle_eval = None


    for offset in offsets:
        current_aim_x = tee_x + offset

        #Calculating aim_y so the total distance off the tee eqauls the average driving distance
        current_aim_y = tee_y + np.sqrt(
            average_distance**2 - offset**2
        )
        #Calculate angle from tee to landing center
        #the angle need to be changed when we have an aim that is offseted form the dispersion
        #Otherwise we would get a faulty dispersion
        angle = -np.degrees(
            np.arctan2(
                current_aim_x - tee_x,
                current_aim_y - tee_y
            )
        )
        results = evaluate_aimpoint(
            dispersion,
            current_aim_x,
            current_aim_y,
            angle,
            zones_yards=zones_yards
        )
        #get the eval score with risk-adjustments
        print()
        score = evaluate_score(results)
        print ("Weighted Score: ", score)
        print()

        #find the best eval_score and its offset
        if score > best_eval_score:
            #Store the best eval_score
            best_eval_score = score
            #Store the offset for the best score
            best_evalscore_offset = offset
            #Store the angle for the best score
            best_aim_angle_eval = angle

        #get fairway percentage for the offset
        fairway_percentage = results.get("fairway", 0)

        #compare it to most_faiway_percentage
        if fairway_percentage > most_fairway_percentage:
            #store that fairway as most fairway percentage
            most_fairway_percentage = fairway_percentage
            #remember the offset
            best_offset_fairway_hit = offset
            #remember the angle
            best_aim_angle_fairway = angle

        print(f"\nLanding center offset: {offset} yards")
        print(results)
    
    print("Best offset:", best_offset_fairway_hit)
    print("Highest fairway percentage:", most_fairway_percentage)
    print()
    print("Best Eval-Score:", best_eval_score)
    print("Best Offset (EVAL-SCORE):", best_evalscore_offset)

    #Plot the highest fairway percentage map
    best_aim_x = tee_x + best_offset_fairway_hit
    best_aim_y = tee_y + np.sqrt(average_distance**2 - best_offset_fairway_hit**2)

    #Position Rory's dispersion on most fairway aimpoint
    landing_x, landing_y = position_dispersion(dispersion, best_aim_x, best_aim_y, best_aim_angle_fairway)

    # Find the green center
    green_center_coords = get_zone_center(zones_yards, "green")
    #Identify the x and y coordinates of the green center
    green_x, green_y = green_center_coords

    #Calculate the remaining distance
    remaining_distances = calculate_remaining_distance(landing_x, landing_y, green_x, green_y)
    print(f"Average remaining distance: {np.mean(remaining_distances)} yards")
    print(f"Shortest Remaining distance: {np.min(remaining_distances)} yards")
    print(f"Longest Remaining distance: {np.max(remaining_distances)} yards")

    # Classify each tee shot's landing zone
    landing_zones = classify_landing_zones(
        landing_x,
        landing_y,
        zones_yards
    )

    # Get all playable approach situations
    approach_situations = get_approach_situations(
        landing_x,
        landing_y,
        landing_zones,
        remaining_distances
    )

    print(f"\nTotal simulated tee shots: {len(landing_x)}")
    print(f"Playable approach situations: {len(approach_situations)}")

    # Display the first 50 approach situations
    """for situation in approach_situations[:50]:
        print(situation)"""

    #Plot Rory's dispersion on most fairway hits
    plot_course_map(
    zones_yards,
    (tee_x, tee_y),
    green_center_coords,
    landing_x,
    landing_y,
    best_aim_x,
    best_aim_y,
    "Rory McIlroy"
    )

    #Plot the highest eval-score map
    best_aim_x = tee_x + best_evalscore_offset
    best_aim_y = tee_y + np.sqrt(average_distance**2 - best_evalscore_offset**2)

    #Position Rory's dispersion on most fairway aimpoint
    landing_x, landing_y = position_dispersion(dispersion, best_aim_x, best_aim_y, best_aim_angle_eval)

    #Plot Rory's dispersion on best Eval-Score
    plot_course_map(   
    zones_yards,
    (tee_x, tee_y),
    green_center_coords,
    landing_x,
    landing_y,
    best_aim_x,
    best_aim_y,
    "Rory McIlroy")

    #Load Rory's historical approach dispersions
    approach_shots = filter_approach_shots(shots)
    targets = get_hole_target(shots)

    approach_dispersion = calculate_approach_dispersion(approach_shots, targets)

    #Identify potential recovery shots and exclude them from data
    approach_dispersion = classify_shot_intent(approach_dispersion)

    all_approach_x = []
    all_approach_y = []

    #simulate all approaches from every playable situation
    for situation in approach_situations:

        approach_x, approach_y = simulate_approach(
            situation["landing_x"],
            situation["landing_y"],
            situation["approach_group"],
            approach_dispersion,
            situation["remaining_distance"],
            green_x,
            green_y
        )

        # Store all simulated approach landing positions
        all_approach_x.extend(approach_x)
        all_approach_y.extend(approach_y)

    print("Total simulated approach outcomes:", len(all_approach_x))

    #35,500 shots are simulated for approach
    #Plotting all of these would be messy
    #Therefore I am randomly picking 2,000 of these to plot

   
    # Randomly select 2000 approach outcomes for visualization
    rng = np.random.default_rng(42)

    sample_size = min(2000, len(all_approach_x))

    indices = rng.choice(
        len(all_approach_x),
        size=sample_size,
        replace=False
    )

    plot_x = np.asarray(all_approach_x)[indices]
    plot_y = np.asarray(all_approach_y)[indices]

    #Plot all approach shots
    plot_course_map(
        zones_yards,
        (tee_x, tee_y),
        (green_x, green_y),
        plot_x,
        plot_y,
        green_x,
        green_y,
        "Rory McIlroy - Simulated Approach Shots",
        zoom=100
    )

    #Classify in what zones the approach shots end up:
    approach_landing_zones = classify_landing_zones(all_approach_x, all_approach_y, zones_yards)

    #count the number of shots ending up in each zone
    approach_zone_counts = Counter(approach_landing_zones)

    #calculate total number of approach shots
    total_approaches = len(approach_landing_zones)
    print(f"Total shots Again: {total_approaches}")

    #Print the result
    print("Rory Mcilroy - Approach shot statistics")
    
    for zone, count in approach_zone_counts.items():
        percentage = (count / total_approaches) * 100
        print(f"{zone}: {count} shots ({percentage:.2f}%)")

    #Calculate the distance remaining after approach has been hit
    approach_remaining_distances = calculate_remaining_distance(
        np.asarray(all_approach_x),
        np.asarray(all_approach_y),
        green_x,
        green_y
    )

    print("Remaining Distances After Approach:")
    print("Total shots:", len(approach_remaining_distances))
    print(f"Average distance: {np.mean(approach_remaining_distances):.2f} yards")
    print(f"Minimum distance: {np.min(approach_remaining_distances):.2f} yards")
    print(f"Maximum distance: {np.max(approach_remaining_distances):.2f} yards")

    #Store the simulated approach in a dataframe
    approach_results = pd.DataFrame({
        "landing_zone": approach_landing_zones,
        "remaining_distance": approach_remaining_distances

    })

    print(approach_results.head(20).to_string(index=False))



    """
    #THIS IS AN EXPERIMENT TO TEST THE APPROACH
    # Prepare Rory's existing approach dispersion
    approach_shots = filter_approach_shots(shots)
    targets = get_hole_target(shots)

    approach_dispersion = calculate_approach_dispersion(
        approach_shots,
        targets
    )

    approach_dispersion = classify_shot_intent(
        approach_dispersion
    )

    # Select all standard approaches from 175-200 yards
    approach_175_200 = approach_dispersion[
        (approach_dispersion["starting_distance"] >= 175) &
        (approach_dispersion["starting_distance"] < 200) &
        (~approach_dispersion["potential_recovery"])
    ].copy()

    print("Approach shots in 175-200 group:", len(approach_175_200))

    # Temporary test position: 185 yards directly below green center
    green_x, green_y = get_zone_center(zones_yards, "green")

    test_distance = 185
    test_x = green_x
    test_y = green_y - test_distance

    approach_x, approach_y = simulate_approach(
    test_x,
    test_y,
    (175, 200),
    approach_175_200,
    test_distance,
    green_x,
    green_y
    )

    plot_course_map(
    zones_yards,
    (tee_x, tee_y),
    (green_x, green_y),
    approach_x,
    approach_y,
    green_x,
    green_y,
    "Rory McIlroy - 175-200 Yard Approach Dispersion"
)
"""



if __name__ == "__main__":
    main()