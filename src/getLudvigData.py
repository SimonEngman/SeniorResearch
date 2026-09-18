import pgatourpy as pga
import pandas as pd
import time

ludvigAberg_id = "52955" #Get result for Ludvig Aberg
season = 2025

# Get Ludvig's 2025 tournament results
results = pga.pga_player_results(
    ludvigAberg_id,
    season=season
)

all_shots = []
successful_tournaments = []
failed_tournaments = []

for index, row in results.iterrows():

    tournament_id = row["tournament_id"]
    tournament_name = row.get("tournament", "Unknown")

    print(f"\n[{index + 1}/{len(results)}] {tournament_name}")
    print("Tournament ID:", tournament_id)

    if pd.isna(tournament_id):
        print("Skipping - no tournament ID")
        continue

    tournament_shots = []

    # Try all four rounds
    for round_number in range(1, 5):

        print(f"  Round {round_number}...", end=" ")

        try:
            shots = pga.pga_shot_details(
                str(tournament_id),
                ludvigAberg_id,
                round=round_number
            )

            if len(shots) > 0:

                # Add information about the tournament and round
                shots["tournament_id"] = tournament_id
                shots["tournament"] = tournament_name
                shots["season"] = season
                shots["round_number"] = round_number

                tournament_shots.append(shots)

                print(f"{len(shots)} shots")

            else:
                print("0 shots")

        except Exception as e:
            print("ERROR:", e)

        # Small pause between API requests
        time.sleep(0.5)

    # Combine all rounds from this tournament
    if tournament_shots:

        tournament_data = pd.concat(
            tournament_shots,
            ignore_index=True
        )

        all_shots.append(tournament_data)

        successful_tournaments.append(tournament_name)

        print(
            "  Tournament total:",
            len(tournament_data),
            "shots"
        )

    else:

        failed_tournaments.append(tournament_name)

        print("No shot data found")

# Combine everything
if all_shots:
        
    ludvig_shots = pd.concat(
        all_shots,
        ignore_index=True
    )

    # Save the complete dataset
    output_file = "data/ludvig_2025_shots.csv"

    ludvig_shots.to_csv(
        output_file,
        index=False
    )

    print("\n==============================")
    print("DATA COLLECTION COMPLETE")
    print("==============================")

    print("Total shots:", len(ludvig_shots))
    print("Successful tournaments:", len(successful_tournaments))
    print("Tournaments without data:", len(failed_tournaments))

    print("\nSaved to:")
    print(output_file)

    print("\nColumns:")
    print(ludvig_shots.columns.tolist())

else:

    print("\nNo shot data was collected.")