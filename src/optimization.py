from collections import Counter
import numpy as np

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
    calculate_dispersion_typdir
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
    offsets = [-30, -20, -10, 0, 10, 20, 30]

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



if __name__ == "__main__":
    main()