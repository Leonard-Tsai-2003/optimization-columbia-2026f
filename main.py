import os
import argparse
from typing import Dict, List, Any
from data_preprocessing import build_problem_dataset
from optimization_model import solve_citi_bike_rebalancing
from plot_routes import plot_rebalancing_solution


def print_detailed_routing_plan(solution: Dict[str, Any], stations_dict: Dict[str, Any], depots: List[Dict[str, Any]]):
    """Generates a step-by-step routing and load management plan."""
    print("\n" + "=" * 70)
    print(" DETAILED OVERNIGHT REBALANCING ROUTING PLAN ")
    print("=" * 70)

    for truck_id, rinfo in solution['routes'].items():
        path = rinfo['path']
        print(f"\n[Vehicle: {truck_id}]")
        if not path or len(path) <= 1:
            print("  Status: INACTIVE / UNUSED (0.0 km traveled)")
            continue

        print(f"  Route Summary : {' -> '.join([p[:8] for p in path])}")
        print(f"  Distance      : {rinfo['distance_km']} km")
        print("  Leg-by-Leg Execution:")
        print("  " + "-" * 62)

        transfers = rinfo['transfers']
        loads = rinfo['loads']

        for step_idx, node in enumerate(path):
            if step_idx == 0:
                print(f"  Step {step_idx}: Depart {node} with 0 bikes on board.")
            elif step_idx == len(path) - 1:
                print(f"  Step {step_idx}: Return to {node}. Final Truck Load: 0 bikes.")
            else:
                st_info = stations_dict.get(node, {})
                st_name = st_info.get('name', f"Station {node[:8]}")
                net_z = transfers.get(node, 0)
                load_after = loads.get(node, 0)

                if net_z > 0:
                    action_desc = f"DELIVER +{net_z} bikes"
                elif net_z < 0:
                    action_desc = f"PICKUP {abs(net_z)} bikes"
                else:
                    action_desc = "VISIT / NO TRANSFER"

                print(f"  Step {step_idx}: Arrive at {node[:8]} ({st_name})")
                print(f"          Action     : {action_desc} (Target G_i = {st_info.get('G', 0)})")
                print(f"          Truck Load : {load_after} / 25 bikes")


def main():
    parser = argparse.ArgumentParser(description="Citi Bike Overnight Rebalancing Model - Lower Manhattan")
    parser.add_argument("--sample_pct", "-p", type=float, default=None,
                        help="Percentage of total Lower Manhattan stations to input into the model (e.g. 10 for 10%%, 25 for 25%%).")
    parser.add_argument("--num_stations", "-n", type=int, default=None,
                        help="Fixed count of stations to sample. Used if --sample_pct is not passed (default: 12).")
    parser.add_argument("--trip_data", "-t", type=str, default=None,
                        help="Folder directory or file path to October 2025 historical trip CSV/ZIP files.")
    parser.add_argument("--max_dist", "-w", type=float, default=25.0,
                        help="Max travel distance allowance per truck in km (default: 25.0).")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Operating travel cost per km in $/km (default: 1.5).")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Penalty cost per unit unmet target bike in $/bike (default: 5.0).")

    args = parser.parse_args()

    print("=" * 70)
    print(" Citi Bike Overnight Rebalancing - Lower Manhattan Area ")
    print("=" * 70)

    # Auto-detect '202510-citibike-tripdata' directory if user did not pass -t
    default_dir = "202510-citibike-tripdata"
    trip_path = args.trip_data
    if trip_path is None and os.path.exists(default_dir):
        trip_path = default_dir
        print(f"Auto-detected trip dataset folder: '{default_dir}'")

    num_st = args.num_stations if (args.sample_pct is None and args.num_stations is not None) else (12 if args.sample_pct is None else None)

    # 1. Fetch live data & process historical trip folder
    dataset = build_problem_dataset(
        num_stations=num_st,
        sample_pct=args.sample_pct,
        trip_data_path=trip_path
    )
    stations = dataset['stations']
    stations_dict = {s['id']: s for s in stations}

    print(f"\n--- Station Parameter Summary ({len(stations)} Lower Manhattan Stations) ---")
    print(f"{'Station Name':<38} | {'Cap (C_i)':<9} | {'Initial (I0)':<12} | {'Target (I*)':<11} | {'Net Delivery (G_i)'}")
    print("-" * 90)
    for s in stations:
        print(f"{s['name']:<38} | {s['C']:<9} | {s['I0']:<12} | {s['I_star']:<11} | {s['G']:<15}")

    # 2. Operational Depots & Parameters
    depots = [
        {'truck_id': 'Truck_1', 'lat': 40.708, 'lon': -74.011},  # Financial District Depot S_1
        {'truck_id': 'Truck_2', 'lat': 40.720, 'lon': -73.985}   # Lower East Side Depot S_2
    ]
    
    Q = 25              # Max truck capacity
    W_k = args.max_dist # Distance budget per truck
    alpha = args.alpha  # Operating travel cost ($/km)
    beta = args.beta    # Penalty per bike
    license_file = "gurobi.lic"

    # 3. Solve Model
    print("\nSolving optimization model via Gurobi...")
    solution = solve_citi_bike_rebalancing(
        data=dataset,
        depots=depots,
        truck_capacity=Q,
        max_distance_w=W_k,
        alpha=alpha,
        beta=beta,
        license_path=license_file
    )

    # 4. Display Results & Detailed Routing Plan
    print("\n" + "=" * 70)
    print(" LOWER MANHATTAN REBALANCING RESULTS ")
    print("=" * 70)
    
    if solution['status'] in [2, 9]:
        summary = solution['summary']
        print(f"Total Objective Cost  : ${summary['total_cost']}")
        print(f"  - Transportation Cost: ${summary['transport_cost']}")
        print(f"  - Inventory Penalty  : ${summary['penalty_cost']}")
        print(f"Total Unmet Shortfall : {summary['total_unmet_bikes']} bikes")

        # Detailed Routing Plan
        print_detailed_routing_plan(solution, stations_dict, depots)

        # Plot Solution Map with Sample Percentage
        plot_rebalancing_solution(
            dataset, 
            depots, 
            solution, 
            save_filename="rebalancing_routes_map.png",
            sample_pct=args.sample_pct
        )


if __name__ == "__main__":
    main()