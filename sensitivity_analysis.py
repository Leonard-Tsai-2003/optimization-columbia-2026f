"""
Citi Bike Rebalancing Model - Sensitivity Analysis Pipeline
===========================================================
Executes One-At-A-Time (OFAT) parameter sweeps for Q, w, alpha, beta, and m,
logs model performance and stats to Weights & Biases (W&B), and saves
local CSV, JSON, and 4-panel visual dashboard artifacts.
"""

import argparse
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gurobipy import GRB

# Import solver and validation functions directly from project modules
from main import validate_solution
from optimization_model import init_gurobi_env, solve_citi_bike_rebalancing
from plot_routes import plot_routes

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Baseline settings (aligned with main.py defaults)
DEFAULT_PARAMS = {
    'truck_capacity': 25,
    'max_distance': 25.0,
    'alpha': 1.5,
    'beta': 5.0,
    'num_trucks': 3,
}

# Parameter sweep grids
PARAM_SWEEPS = {
    'Q (Truck Capacity)': {
        'param_key': 'truck_capacity',
        'values': [21, 23, 25, 27, 29],
    },
    'w (Max Distance km)': {
        'param_key': 'max_distance',
        'values': [21.0, 23.0, 25.0, 27.0, 29.0],
    },
    'alpha (Transport Weight)': {
        'param_key': 'alpha',
        'values': [1.3, 1.4, 1.5, 1.6, 1.7],
    },
    'beta (Mismatch Penalty)': {
        'param_key': 'beta',
        'values': [4.5, 4.75, 5.0, 5.25, 5.5],
    },
    'm (Number of Trucks)': {
        'param_key': 'num_trucks',
        'values': [1, 2, 3, 4, 5],
    },
}


def load_inputs(root_dir: Path):
    """Loads and validates input station data matching main.py."""
    prepared_input = root_dir / 'outputs_stage6' / 'model_station_inputs.json'
    station_input = prepared_input if prepared_input.is_file() else root_dir / 'model_station_inputs.json'
    if not station_input.is_file():
        raise FileNotFoundError(f'Station input file not found in {root_dir}.')
    data = json.loads(station_input.read_text())

    station_table = pd.DataFrame(data['stations'])
    assert station_table['id'].is_unique
    assert (station_table['G'] == station_table['I_star'] - station_table['I0']).all()
    assert station_table['I_star'].sum() == station_table['I0'].sum()

    return data, prepared_input.is_file()


def get_depots_for_num_trucks(root_dir: Path, data: dict, num_trucks: int, use_prepared_input: bool):
    """Loads existing depots or selects p-median depots for num_trucks as in main.py."""
    depot_file = root_dir / 'selected_depots.json'
    if not depot_file.is_file():
        alt_depot_file = root_dir / 'outputs_stage7' / 'selected_depots.json'
        if alt_depot_file.is_file():
            depot_file = alt_depot_file

    existing_depots = json.loads(depot_file.read_text()) if depot_file.is_file() else []
    if not use_prepared_input and len(existing_depots) == num_trucks:
        return existing_depots

    stations = sorted(data['stations'], key=lambda station: station['id'])
    p = num_trucks
    if len(stations) < p:
        raise ValueError(f'At least {p} eligible stations are required to select {p} depots.')

    latitudes = np.array([station['lat'] for station in stations])
    longitudes = np.array([station['lon'] for station in stations])
    depot_distances = np.round(
        84 * np.abs(longitudes[:, None] - longitudes[None, :])
        + 111 * np.abs(latitudes[:, None] - latitudes[None, :]), 3
    )

    if math.comb(len(stations), p) <= 2_000_000:
        best_tuple = min(
            combinations(range(len(stations)), p),
            key=lambda idx_tuple: np.min(depot_distances[:, list(idx_tuple)], axis=1).sum()
        )
    else:
        n_st = len(stations)
        with init_gurobi_env(str(root_dir / "gurobi.lic")) as env:
            with gp.Model("Depot_PMedian_Selection", env=env) as m_depot:
                m_depot.Params.OutputFlag = 0
                z_dep = m_depot.addVars(n_st, vtype=GRB.BINARY, name="z")
                y_dep = m_depot.addVars(n_st, n_st, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")
                m_depot.addConstr(gp.quicksum(z_dep[j] for j in range(n_st)) == p)
                for i in range(n_st):
                    m_depot.addConstr(gp.quicksum(y_dep[i, j] for j in range(n_st)) == 1.0)
                    for j in range(n_st):
                        m_depot.addConstr(y_dep[i, j] <= z_dep[j])
                m_depot.setObjective(
                    gp.quicksum(
                        float(depot_distances[i, j]) * y_dep[i, j]
                        for i in range(n_st) for j in range(n_st)
                    ),
                    GRB.MINIMIZE,
                )
                m_depot.optimize()
                best_tuple = tuple(sorted(j for j in range(n_st) if z_dep[j].X > 0.5))

    nearest = np.argmin(depot_distances[:, list(best_tuple)], axis=1)
    depots = []
    for index, station_index in enumerate(best_tuple):
        station = stations[station_index]
        truck = f'Truck_{index + 1}'
        depots.append({
            'truck_id': truck,
            'depot_id': f'depot_{truck}',
            'name': f"Depot {index + 1} (candidate near {station['name']})",
            'lat': station['lat'],
            'lon': station['lon'],
            'candidate_station_id': station['id'],
            'nearest_station_count': int((nearest == index).sum()),
            'site_feasibility_verified': False,
        })
    return depots


def run_single_experiment(
    data, depots, root_dir, out_base_dir, time_limit,
    Q, w, a, b, m, sweep_name, val,
    max_neighbors=None, use_warm_start=True,
):
    """Runs solver for a single parameter configuration and returns recorded statistics."""
    run_id = (
        f"sweep_{sweep_name}_m{m}_Q{Q}_w{w}_a{a}_b{b}"
        .replace('.', '_').replace(' ', '_').replace('(', '').replace(')', '')
    )
    run_dir = out_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "selected_depots.json").write_text(json.dumps(depots, indent=2))

    license_path = str(root_dir / "gurobi.lic")

    start_time = time.time()
    solution = solve_citi_bike_rebalancing(
        data, depots, run_dir, license_path,
        truck_capacity=int(Q), max_distance_w=float(w),
        alpha=float(a), beta=float(b), time_limit=time_limit,
        max_neighbors=max_neighbors, use_warm_start=use_warm_start,
    )
    (run_dir / "solution.json").write_text(json.dumps(solution, indent=2))

    _, summary = validate_solution(
        data, depots, run_dir, int(Q), float(w), float(a), float(b),
    )
    elapsed_total = time.time() - start_time

    try:
        plot_routes(data, depots, solution, run_dir)
    except Exception as err:
        print(f"Warning: Route plotting failed for {run_id}: {err}")

    record = {
        'run_id': run_id,
        'swept_parameter': sweep_name,
        'swept_value': val,
        'num_trucks_m': int(m),
        'truck_capacity_Q': int(Q),
        'max_distance_w': float(w),
        'alpha': float(a),
        'beta': float(b),
        'time_limit_p': time_limit,
        'objective': float(summary['objective']),
        'objective_bound': float(solution.get('objective_bound', 0.0)),
        'total_distance_km': float(summary['approximate_total_distance_km']),
        'transport_cost': float(summary['transport_cost']),
        'mismatch_penalty_cost': float(summary['mismatch_penalty_cost']),
        'mismatch_before': int(summary['mismatch_before']),
        'mismatch_after': int(summary['mismatch_after']),
        'mismatch_reduction': int(summary['mismatch_reduction']),
        'bikes_moved': int(summary['bikes_moved']),
        'visited_stations': int(summary['visited_station_count']),
        'no_rebalancing_objective': float(summary['no_rebalancing_objective']),
        'objective_reduction': float(summary['objective_reduction']),
        'solver_status': solution['status'],
        'proven_optimal': bool(summary.get('proven_optimal', False)),
        'mip_gap': float(solution.get('gap', 0.0)),
        'solver_runtime_sec': float(solution.get('runtime_seconds', elapsed_total)),
    }
    return record, run_dir


def generate_local_dashboard(df: pd.DataFrame, output_path: Path):
    """Generates a 4-panel dashboard summarizing parameter sensitivity."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Citi Bike Rebalancing Sensitivity Analysis Dashboard", fontsize=18, fontweight='bold', y=0.98)

    # 1. Total Objective Cost vs Parameter Values
    ax1 = axes[0, 0]
    sns.lineplot(data=df, x='swept_value', y='objective', hue='swept_parameter', marker='o', linewidth=2.5, ax=ax1)
    ax1.set_title("Total Objective Cost ($Z = \\alpha D + \\beta U$)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Parameter Value", fontsize=11)
    ax1.set_ylabel("Objective Value ($)", fontsize=11)

    # 2. Total Travel Distance (km) vs Parameter Values
    ax2 = axes[0, 1]
    sns.lineplot(data=df, x='swept_value', y='total_distance_km', hue='swept_parameter', marker='s', linewidth=2.5, ax=ax2)
    ax2.set_title("Total Travel Distance (km)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Parameter Value", fontsize=11)
    ax2.set_ylabel("Distance (km)", fontsize=11)

    # 3. Post-Rebalancing Target Mismatch (Bikes)
    ax3 = axes[1, 0]
    sns.lineplot(data=df, x='swept_value', y='mismatch_after', hue='swept_parameter', marker='^', linewidth=2.5, ax=ax3)
    ax3.set_title("Remaining Inventory Mismatch (Bikes)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Parameter Value", fontsize=11)
    ax3.set_ylabel("Mismatch Count", fontsize=11)

    # 4. Transport Cost vs Penalty Cost Breakdown
    ax4 = axes[1, 1]
    df_melted = df.melt(
        id_vars=['swept_parameter', 'swept_value'], 
        value_vars=['transport_cost', 'mismatch_penalty_cost'],
        var_name='Cost Type', value_name='Cost ($)'
    )
    df_melted['Cost Type'] = df_melted['Cost Type'].map({
        'transport_cost': 'Transport Cost (alpha * dist)',
        'mismatch_penalty_cost': 'Penalty Cost (beta * mismatch)'
    })
    sns.barplot(data=df_melted, x='swept_parameter', y='Cost ($)', hue='Cost Type', ax=ax4, errorbar=None)
    ax4.set_title("Cost Component Breakdown by Parameter Sweep", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Swept Parameter Group", fontsize=11)
    ax4.set_ylabel("Cost ($)", fontsize=11)
    ax4.tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Local dashboard plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis for Citi Bike Rebalancing Model.")
    parser.add_argument("-p", "--time-limit", type=float, default=2000, help="Solver time limit per run in seconds")
    parser.add_argument("-k", "--max-neighbors", type=int, default=None, help="Optional k-nearest neighbor arc pruning limit (default: exact full graph)")
    parser.add_argument("--no-warm-start", action="store_true", help="Disable Clarke-Wright + local search MIP warm start")
    parser.add_argument("--wandb-project", type=str, default="citi-bike-sensitivity", help="W&B Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity/Username (optional)")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("Time limit must be positive and finite.")
    if args.max_neighbors is not None and args.max_neighbors < 1:
        parser.error("Max neighbors must be a positive integer.")

    ROOT = Path(__file__).resolve().parent
    out_base_dir = ROOT / "results_sensitivity"
    out_base_dir.mkdir(exist_ok=True)

    data, use_prepared_input = load_inputs(ROOT)
    depots_by_m = {}
    default_depots = get_depots_for_num_trucks(ROOT, data, DEFAULT_PARAMS['num_trucks'], use_prepared_input)
    depots_by_m[DEFAULT_PARAMS['num_trucks']] = default_depots

    print(f"Loaded {len(data['stations'])} stations and {len(default_depots)} baseline depots.")
    print(f"Time limit per run: {args.time_limit} seconds.")

    results_list = []
    runs_to_execute = []
    executed_configs = {}

    for param_name, config in PARAM_SWEEPS.items():
        key = config['param_key']
        for val in config['values']:
            params = DEFAULT_PARAMS.copy()
            params[key] = val
            config_tuple = (
                params['truck_capacity'],
                params['max_distance'],
                params['alpha'],
                params['beta'],
                params['num_trucks'],
            )

            runs_to_execute.append({
                'sweep_name': param_name,
                'swept_value': val,
                'Q': params['truck_capacity'],
                'w': params['max_distance'],
                'a': params['alpha'],
                'b': params['beta'],
                'm': params['num_trucks'],
                'config_tuple': config_tuple,
            })

    total_runs = len(runs_to_execute)
    print(f"Total sensitivity runs scheduled: {total_runs}")

    for idx, run_info in enumerate(runs_to_execute, 1):
        print(f"\n[{idx}/{total_runs}] Running Sweep: {run_info['sweep_name']} = {run_info['swept_value']}")
        print(
            f"Config -> m={run_info['m']}, Q={run_info['Q']}, w={run_info['w']}, "
            f"alpha={run_info['a']}, beta={run_info['b']}"
        )

        m_trucks = int(run_info['m'])
        if m_trucks not in depots_by_m:
            depots_by_m[m_trucks] = get_depots_for_num_trucks(ROOT, data, m_trucks, use_prepared_input)
        depots = depots_by_m[m_trucks]

        config_tuple = run_info['config_tuple']
        if config_tuple in executed_configs:
            print("Configuration already executed. Reusing solution stats...")
            base_rec, run_dir = executed_configs[config_tuple]
            rec = base_rec.copy()
            rec['swept_parameter'] = run_info['sweep_name']
            rec['swept_value'] = run_info['swept_value']
            rec['run_id'] = (
                f"sweep_{run_info['sweep_name']}_val_{run_info['swept_value']}"
                .replace('.', '_').replace(' ', '_').replace('(', '').replace(')', '')
            )
        else:
            rec, run_dir = run_single_experiment(
                data, depots, ROOT, out_base_dir, args.time_limit,
                run_info['Q'], run_info['w'], run_info['a'], run_info['b'], m_trucks,
                run_info['sweep_name'], run_info['swept_value'],
                max_neighbors=args.max_neighbors, use_warm_start=not args.no_warm_start,
            )
            executed_configs[config_tuple] = (rec, run_dir)

        results_list.append(rec)

        # Log individual run to W&B
        if WANDB_AVAILABLE and not args.disable_wandb:
            try:
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"{run_info['sweep_name']}={run_info['swept_value']}",
                    config={
                        'swept_parameter': run_info['sweep_name'],
                        'swept_value': run_info['swept_value'],
                        'num_trucks': run_info['m'],
                        'truck_capacity': run_info['Q'],
                        'max_distance': run_info['w'],
                        'alpha': run_info['a'],
                        'beta': run_info['b'],
                        'max_neighbors': args.max_neighbors,
                        'use_warm_start': not args.no_warm_start,
                        'time_limit': args.time_limit
                    },
                    reinit=True
                )
                wandb.log(rec)
                run.finish()
            except Exception as err:
                print(f"W&B logging skipped/failed: {err}")

    # Export aggregated tabular data
    df_results = pd.DataFrame(results_list)
    csv_path = out_base_dir / "sensitivity_results.csv"
    json_path = out_base_dir / "sensitivity_results.json"
    df_results.to_csv(csv_path, index=False)
    json_path.write_text(df_results.to_json(orient="records", indent=2))
    print(f"\nAll sensitivity statistics exported to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    # Render local dashboard plot
    dashboard_path = out_base_dir / "sensitivity_analysis_dashboard.png"
    generate_local_dashboard(df_results, dashboard_path)

    # Master W&B Dashboard Summary Upload
    if WANDB_AVAILABLE and not args.disable_wandb:
        try:
            master_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="Sensitivity_Summary_Dashboard",
                job_type="analysis",
                reinit=True
            )
            wandb.log({"Sensitivity_Dashboard": wandb.Image(str(dashboard_path))})
            wandb.log({"Sensitivity_Table": wandb.Table(dataframe=df_results)})
            master_run.finish()
            print("Uploaded summary table and visualization dashboard to W&B.")
        except Exception as err:
            print(f"Master W&B upload skipped: {err}")


if __name__ == "__main__":
    main()