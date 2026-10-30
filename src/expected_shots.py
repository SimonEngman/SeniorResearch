import numpy as np
import math

#Mathematical equations from Mark Broadie's research to calculate expected putts
#The input needs to be in feet, not yards
#The output will be expected remaining putts to hole out
def expected_putts(d_feet):
    if d_feet < 0:
        raise ValueError("The distance can't be negative") #We can't have a negative distance
    if d_feet == 0:
        return 0.0 #The ball is already in the hole
    
    #convert to yards for one putt probability
    d_yards = d_feet / 3

    #Parameter that will be used in Broadie's equation
    sigma_alpha = math.radians(1.46)
    sigma_d = 0.057
    t = 0.5
    h = 2 / 3

    # Standard normal cumulative distribution function
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Equation 5: One-putt probability
    hole_radius = 2.125 / 36  # Inches to yards
    alpha_c = math.atan(hole_radius / d_yards)

    direction_probability = 2 * phi(alpha_c / sigma_alpha) - 1

    distance_probability = (
        phi((h - t) / (sigma_d * (d_yards + t)))
        - phi(-t / (sigma_d * (d_yards + t)))
    )

    p1 = direction_probability * distance_probability

    # Equation 6: Three-putt probability
    # Distance is in feet for this fitted function
    a0 = 5.49
    a1 = -0.106
    a2 = 0.000563
    a3 = -0.00398

    p3 = 1 / (
        1 + math.exp(a0 + a1 * d_feet + a2 * d_feet**2)
    ) + a3

    # Equation 7: Expected putts remaining
    return 2 - p1 + p3

#The function uses linear interpolation and allows us tho calculate
#expected shots that are not 10, 20, 30 yards etc
def get_expected_strokes(zone, distance_yards):
    if zone not in EXPECTED_SHOTS:
        raise ValueError(f"Unknown landing zone: {zone}")

    if distance_yards < 0:
        raise ValueError("Distance cannot be negative")

    data = EXPECTED_SHOTS[zone]

    distances = np.array(sorted(data.keys()))
    strokes = np.array([data[d] for d in distances])

    if distance_yards < distances[0] or distance_yards > distances[-1]:
        raise ValueError("Distance outside available data range")

    return np.interp(distance_yards, distances, strokes)

#I have taken these value from Mark Broadie's research on Shots-Gained
#I am including distances <= 100 yards, since they won't have furter left in our simulation
#Bunker shots are <= 70 yards, since it is impossible to have a bunker shot longer than 70 yards on Augusta hole 13
EXPECTED_SHOTS = {
    "fairway": {
        10: 2.18,
        20: 2.40,
        30: 2.52,
        40: 2.60,
        50: 2.66,
        60: 2.70,
        70: 2.72,
        80: 2.75,
        90: 2.77,
        100: 2.80,
    },
    "rough": {
        10: 2.34,
        20: 2.59,
        30: 2.70,
        40: 2.78,
        50: 2.87,
        60: 2.91,
        70: 2.93,
        80: 2.96,
        90: 2.99,
        100: 3.02
    },
    "bunker": { 
        10: 2.43,
        20: 2.53,
        30: 2.66,
        40: 2.82,
        50: 2.92,
        60: 3.15,
        70: 3.21
    }
}

#Calculate the expected putts from 1 - 91 feet
PUTTING_DISTANCES = np.arange(1, 91)
EXPECTED_PUTTS = np.array([
    expected_putts(distance) for distance in PUTTING_DISTANCES
])



def main():
    print("Expected Putts from 1 - 90 feet:")
    for i in range(90):
        print(f"Expected Putts - {PUTTING_DISTANCES[i]} feet: {EXPECTED_PUTTS[i]:.2f}")


    #Testing the get_expected_shots funtion
    print(f"fairway, 20 yards: {get_expected_strokes("fairway", 20):.2f}") 
    print(f"fairway, 30 yards: {get_expected_strokes("fairway", 30):.2f}") 
    print(f"fairway, 25 yards: {get_expected_strokes("fairway", 25):.2f}") 
    print(f"rough, 17 yards: {get_expected_strokes("rough", 17):.2f}") 
    print(f"bunker, 21 yards: {get_expected_strokes("bunker", 21):.2f}") 


if __name__ == "__main__":
    main()