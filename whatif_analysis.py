"""
Citi Bike Overnight Rebalancing — What-If Scenario Analysis
==========================================================
Conducts two what-if experiments using the Midtown Manhattan MILP solver:
  1. Question 1: Baseline vs. 1.1x positive target (G_i > 0) and 1.2x positive target.
  2. Question 2: Baseline (1x Targets, 3 Trucks) — replayed directly from Q1 baseline —
     vs. 2x targets (both positive and negative) with a 4th truck (+1 optimal depot).

Outputs (saved in `results_whatif/`):
  - Per-scenario folders (`q1_01_baseline`, `q1_02_pos_target_x1_1`,
    `q1_03_pos_target_x1_2`, `q2_01_baseline_3trucks`, `q2_03_2x_targets_4trucks_add1`)
    containing `solution.json`, `summary.csv`, `route_steps.csv`,
    `station_before_after.csv`, `gurobi.log`, and `routes.png`.
  - Aggregated tables (`whatif_all_runs.csv`, `whatif_q1_summary.csv`,
    `whatif_q2_summary.csv`, `whatif_summary.json`).
  - Two comparison charts styled after your sensitivity plot template:
      * results_whatif/whatif_q1_positive_targets.png
      * results_whatif/whatif_q2_2x_targets_4th_truck.png

Usage:
  python whatif_analysis.py -p 1200
  python whatif_analysis.py --plot-only   # Re-render comparison plots from saved CSVs
"""

import argparse
import copy
import csv
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimization_model import latlon_to_km, solve_citi_bike_rebalancing
from plot_routes import plot_routes
from sensitivity_analysis import DEFAULT_PARAMS, get_depots_for_num_trucks, load_inputs


# ---------------------------------------------------------------------------
# 1. Target Scaling & 4th Depot Selection Helpers
# ---------------------------------------------------------------------------

def build_scaled_target_data(
    base_data: dict,
    pos_multiplier: float = 1.0,
    neg_multiplier: float = 1.0,
    clip_to_capacity: bool = True,
) -> dict:
    """
    Returns a deep copy of `base_data` with station target gaps G_i scaled:
      - G_i > 0 (deficit stations): scaled by `pos_multiplier`
      - G_i < 0 (surplus stations): scaled by `neg_multiplier`
    Updates `I_star = I0 + G` and clips `I_star` to [0, C_i] so target inventories
    remain physically valid within midnight dock capacities.
    """
    scaled = copy.deepcopy(base_data)
    for st in scaled["stations"]:
        g_orig = int(st["G"])
        i0 = int(st["I0"])
        cap = int(st["C"])

        if g_orig > 0:
            g_new = int(round(g_orig * pos_multiplier))
        elif g_orig < 0:
            g_new = int(round(g_orig * neg_multiplier))
        else:
            g_new = 0

        i_star_new = i0 + g_new
        if clip_to_capacity:
            i_star_new = int(np.clip(i_star_new, 0, cap))
            g_new = i_star_new - i0

        st["I_star"] = i_star_new
        st["G"] = g_new
        st["target_method"] = f"whatif_pos{pos_multiplier}_neg{neg_multiplier}"

    return scaled


def get_augmented_fourth_depot(data: dict, base_depots: list[dict]) -> list[dict]:
    """
    Keeps the 3 baseline depots fixed and selects the 4th station site that minimizes
    total grid distance from all 165 stations to their nearest depot (greedy p-median).
    """
    depots = copy.deepcopy(base_depots)
    existing_ids = {d.get("candidate_station_id") for d in depots}
    stations = sorted(data["stations"], key=lambda s: s["id"])

    cur_min_dist = np.array([
        min(
            float(latlon_to_km(st["lat"], st["lon"], d["lat"], d["lon"]))
            for d in depots
        )
        for st in stations
    ])

    best_station = None
    best_total_dist = float("inf")
    for cand in stations:
        if cand["id"] in existing_ids:
            continue
        cand_dists = np.array([
            float(latlon_to_km(st["lat"], st["lon"], cand["lat"], cand["lon"]))
            for st in stations
        ])
        total_dist = float(np.minimum(cur_min_dist, cand_dists).sum())
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_station = cand

    if best_station is None:
        raise ValueError("Unable to find an eligible 4th depot candidate.")

    depots.append({
        "truck_id": "Truck_4",
        "depot_id": "depot_Truck_4",
        "name": f"Depot 4 (candidate near {best_station['name']})",
        "lat": best_station["lat"],
        "lon": best_station["lon"],
        "candidate_station_id": best_station["id"],
        "nearest_station_count": int(
            sum(
                float(latlon_to_km(st["lat"], st["lon"], best_station["lat"], best_station["lon"]))
                <= cur_min_dist[idx]
                for idx, st in enumerate(stations)
            )
        ),
        "site_feasibility_verified": False,
    })
    return depots


# ---------------------------------------------------------------------------
# 2. Solution Validation
# ---------------------------------------------------------------------------

def validate_whatif_solution(
    data: dict,
    depots: list[dict],
    results_dir: Path,
    capacity: int,
    max_distance: float,
    alpha: float,
    beta: float,
) -> dict:
    """
    Validates route feasibility, load propagation, capacity limits, bike conservation,
    and objective consistency for both fleet-conserving and asymmetric target scenarios.
    """
    sol = json.loads((results_dir / "solution.json").read_text())
    st_list = data["stations"]
    stations = {s["id"]: s for s in st_list}
    nodes = {**stations, **{d["depot_id"]: d for d in depots}}
    st_results = {s["id"]: s for s in sol["station_results"]}

    assert sol["has_feasible_solution"]
    assert len(stations) == len(st_list) == len(st_results)
    assert set(stations) == set(st_results)
    assert set(sol["routes"]) == {d["truck_id"] for d in depots}

    seen = set()
    route_rows = []
    delivery = dict.fromkeys(stations, 0)
    total_distance = 0.0
    moved_bikes = 0

    for d in depots:
        truck = d["truck_id"]
        route = sol["routes"][truck]
        path = route["path"]
        assert path[0] == path[-1] == d["depot_id"]
        assert set(route["transfers"]) == set(path[1:-1])

        load = 0
        dist = 0.0
        route_rows.append({
            "truck": truck,
            "order": 0,
            "station_id": path[0],
            "name": d["name"],
            "net_delivery": 0,
            "load_after": 0,
            "distance_so_far_km": 0.0,
        })

        for order, (a, b) in enumerate(zip(path[:-1], path[1:]), 1):
            dist += float(latlon_to_km(nodes[a]["lat"], nodes[a]["lon"], nodes[b]["lat"], nodes[b]["lon"]))
            z_val = 0
            if b in stations:
                assert b not in seen, f"Duplicate visit to station {b}"
                seen.add(b)
                z_val = int(route["transfers"][b])
                delivery[b] += z_val
                load -= z_val
                assert math.isclose(load, route["loads"][b], abs_tol=1e-6)
                moved_bikes += max(-z_val, 0)
            else:
                assert b == d["depot_id"] and order == len(path) - 1
            assert 0 <= load <= capacity
            route_rows.append({
                "truck": truck,
                "order": order,
                "station_id": b,
                "name": nodes[b]["name"],
                "net_delivery": z_val,
                "load_after": load,
                "distance_so_far_km": round(dist, 3),
            })

        assert load == 0 and sum(route["transfers"].values()) == 0
        assert dist <= max_distance + 1e-6
        assert math.isclose(dist, route["distance_km"], abs_tol=1e-6)
        total_distance += dist

    station_rows = []
    for sid, st in stations.items():
        r = st_results[sid]
        final_inv = st["I0"] + delivery[sid]
        assert 0 <= final_inv <= st["C"]
        assert r["I0"] == st["I0"] and r["target"] == st["I_star"]
        assert r["net_delivery"] == delivery[sid] and r["final_inventory"] == final_inv
        assert r["mismatch"] == abs(final_inv - st["I_star"])
        station_rows.append({
            "station_id": sid,
            "name": st["name"],
            "effective_capacity": st["C"],
            "midnight_inventory": st["I0"],
            "target_inventory": st["I_star"],
            "net_delivery_total": delivery[sid],
            "final_inventory": final_inv,
            "absolute_mismatch_before": abs(st["I0"] - st["I_star"]),
            "absolute_mismatch_after": abs(final_inv - st["I_star"]),
            "visited": sid in seen,
        })

    initial_total = sum(s["I0"] for s in st_list)
    target_total = sum(s["I_star"] for s in st_list)
    before_mismatch = sum(r["absolute_mismatch_before"] for r in station_rows)
    after_mismatch = sum(r["absolute_mismatch_after"] for r in station_rows)
    lower_bound = abs(target_total - initial_total)
    total_cost = alpha * total_distance + beta * after_mismatch

    assert sum(delivery.values()) == 0
    assert sum(r["final_inventory"] for r in station_rows) == initial_total
    assert after_mismatch >= lower_bound
    assert before_mismatch == sol["baseline_mismatch"]
    assert after_mismatch == sol["total_mismatch"]
    assert math.isclose(total_distance, sol["total_distance_km"], abs_tol=1e-6)
    assert math.isclose(total_cost, sol["objective"], abs_tol=1e-6)

    def write_csv(filename: str, rows: list[dict]):
        with (results_dir / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "station_count": len(st_list),
        "visited_station_count": len(seen),
        "bikes_moved": moved_bikes,
        "initial_inventory_total": initial_total,
        "target_inventory_total": target_total,
        "conservation_mismatch_lower_bound": lower_bound,
        "mismatch_before": before_mismatch,
        "mismatch_after": after_mismatch,
        "mismatch_reduction": before_mismatch - after_mismatch,
        "approximate_total_distance_km": round(total_distance, 3),
        "transport_cost": round(alpha * total_distance, 4),
        "mismatch_penalty_cost": round(beta * after_mismatch, 4),
        "objective": round(total_cost, 4),
        "no_rebalancing_objective": round(beta * before_mismatch, 4),
        "objective_reduction": round(beta * before_mismatch - total_cost, 4),
        "proven_optimal": sol["status"] == "Optimal",
        "checks_passed": True,
    }

    write_csv("route_steps.csv", route_rows)
    write_csv("station_before_after.csv", station_rows)
    write_csv("summary.csv", [summary])
    (results_dir / "stage9_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# 3. Comparison Visualization (Matching User's Matplotlib Template)
# ---------------------------------------------------------------------------

def plot_whatif_comparison(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    output_path: Path,
):
    """Generates the two-panel Objective & MIP Gap comparison plot."""
    df = df.reset_index(drop=True)
    x = np.arange(len(df))
    labels = df["short_label"].tolist()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 6.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1]}
    )

    # Top panel: Objective & Lower Bound with shaded optimality gap
    ax1.fill_between(
        x, df["objective_bound"], df["objective"],
        color="#3B82F6", alpha=0.14, label="Optimality Gap Band (True Optimum Lies Within)"
    )
    ax1.plot(
        x, df["objective"], "o-", lw=2.3, ms=7, color="#1D4ED8",
        label="Incumbent Objective (Upper Bound)"
    )
    ax1.plot(
        x, df["objective_bound"], "s--", lw=2.0, ms=6.5, color="#0F766E",
        label="Best Bound (Lower Bound)"
    )

    for idx, r in df.iterrows():
        ax1.annotate(
            f"{r['objective']:.1f}", (idx, r["objective"]), xytext=(0, 9),
            textcoords="offset points", ha="center", fontweight="bold", color="#1E3A8A"
        )
        ax1.annotate(
            f"{r['objective_bound']:.1f}", (idx, r["objective_bound"]), xytext=(0, -16),
            textcoords="offset points", ha="center", color="#115E59"
        )

    y_min = float(df["objective_bound"].min())
    y_max = float(df["objective"].max())
    y_span = max(y_max - y_min, 25.0)
    ax1.set_ylim(y_min - 0.18 * y_span, y_max + 0.24 * y_span)
    ax1.set_xlim(-0.45, len(df) - 0.55)

    ax1.set_ylabel("Objective Value", fontweight="bold")
    ax1.set_title(title, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="upper left", frameon=True)

    # Bottom panel: MIP Gap (%)
    bar_width = 0.28 if len(df) >= 3 else 0.24
    bars = ax2.bar(
        x, df["mip_gap"] * 100, width=bar_width,
        color="#F59E0B", alpha=0.85, edgecolor="#D97706"
    )
    for bar, g in zip(bars, df["mip_gap"] * 100):
        ax2.annotate(
            f"{g:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", color="#92400E"
        )

    max_gap = max(float((df["mip_gap"] * 100).max()), 5.0)
    ax2.set_ylabel("MIP Gap (%)", fontweight="bold")
    ax2.set_xlabel(xlabel, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max_gap * 1.32)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# 4. Scenario Execution & Baseline Replay Pipeline
# ---------------------------------------------------------------------------

def run_whatif_scenario(
    scenario: dict,
    root_dir: Path,
    out_base_dir: Path,
    time_limit: float,
    truck_capacity: int,
    max_distance: float,
    alpha: float,
    beta: float,
    max_neighbors: int | None,
    use_warm_start: bool,
) -> dict:
    """Executes one scenario, saves all artifacts into its subfolder, and returns metrics."""
    run_id = scenario["run_id"]
    run_dir = out_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data = scenario["data"]
    depots = scenario["depots"]
    (run_dir / "model_station_inputs.json").write_text(json.dumps(data, indent=2))
    (run_dir / "selected_depots.json").write_text(json.dumps(depots, indent=2))

    print(f"\n{'=' * 76}")
    print(f"Running Scenario: {scenario['question']} | {scenario['name']} ({run_id})")
    print(
        f"  Trucks={len(depots)}, PosTargetMult={scenario['pos_mult']}x, "
        f"NegTargetMult={scenario['neg_mult']}x, Q={truck_capacity}, W={max_distance} km"
    )
    print(f"{'=' * 76}")

    t0 = time.time()
    solution = solve_citi_bike_rebalancing(
        data,
        depots,
        run_dir,
        str(root_dir / "gurobi.lic"),
        truck_capacity=truck_capacity,
        max_distance_w=max_distance,
        alpha=alpha,
        beta=beta,
        time_limit=time_limit,
        max_neighbors=max_neighbors,
        use_warm_start=use_warm_start,
    )
    (run_dir / "solution.json").write_text(json.dumps(solution, indent=2))

    summary = validate_whatif_solution(
        data, depots, run_dir, truck_capacity, max_distance, alpha, beta
    )
    elapsed = time.time() - t0

    try:
        plot_routes(data, depots, solution, run_dir)
    except Exception as err:
        print(f"Warning: Route map rendering failed for {run_id}: {err}")

    single_visit_floor = int(sum(max(abs(s["G"]) - truck_capacity, 0) for s in data["stations"]))

    return {
        "question": scenario["question"],
        "run_id": run_id,
        "scenario_name": scenario["name"],
        "short_label": scenario["short_label"],
        "pos_target_multiplier": scenario["pos_mult"],
        "neg_target_multiplier": scenario["neg_mult"],
        "num_trucks": len(depots),
        "truck_capacity_Q": truck_capacity,
        "max_distance_w": max_distance,
        "alpha": alpha,
        "beta": beta,
        "objective": float(summary["objective"]),
        "objective_bound": float(solution["objective_bound"]),
        "mip_gap": float(solution["gap"]),
        "total_distance_km": float(summary["approximate_total_distance_km"]),
        "transport_cost": float(summary["transport_cost"]),
        "mismatch_penalty_cost": float(summary["mismatch_penalty_cost"]),
        "mismatch_before": int(summary["mismatch_before"]),
        "mismatch_after": int(summary["mismatch_after"]),
        "mismatch_reduction": int(summary["mismatch_reduction"]),
        "single_visit_floor": single_visit_floor,
        "bikes_moved": int(summary["bikes_moved"]),
        "visited_stations": int(summary["visited_station_count"]),
        "solver_status": solution["status"],
        "runtime_sec": round(float(solution.get("runtime_seconds", elapsed)), 2),
    }


def replay_q1_baseline_for_q2(
    q1_baseline_record: dict,
    q2_scenario: dict,
    out_base_dir: Path,
) -> dict:
    """
    Replays the completed Q1 baseline run (`q1_01_baseline`) for Q2 (`q2_01_baseline_3trucks`),
    copying its saved artifacts into the Q2 folder and reusing its exact metrics without
    re-running the solver.
    """
    src_dir = out_base_dir / q1_baseline_record["run_id"]
    dst_dir = out_base_dir / q2_scenario["run_id"]
    if src_dir.is_dir():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    replayed = copy.deepcopy(q1_baseline_record)
    replayed["question"] = q2_scenario["question"]
    replayed["run_id"] = q2_scenario["run_id"]
    replayed["scenario_name"] = q2_scenario["name"]
    replayed["short_label"] = q2_scenario["short_label"]
    print(f"\n[Replay] Reused Q1 baseline ({q1_baseline_record['run_id']}) -> {q2_scenario['run_id']}")
    return replayed


def main():
    parser = argparse.ArgumentParser(
        description="Run What-If Questions 1 & 2 for Midtown Citi Bike Rebalancing."
    )
    parser.add_argument(
        "-p", "--time-limit", type=float, default=1200.0,
        help="Gurobi solver time limit per run in seconds (default: 1200)"
    )
    parser.add_argument(
        "-Q", "--truck-capacity", type=int, default=DEFAULT_PARAMS["truck_capacity"],
        help="Truck capacity in bikes (default: 25)"
    )
    parser.add_argument(
        "-w", "--max-distance", type=float, default=DEFAULT_PARAMS["max_distance"],
        help="Max route distance per truck in km (default: 25.0)"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=DEFAULT_PARAMS["alpha"],
        help="Distance travel cost weight alpha (default: 1.5)"
    )
    parser.add_argument(
        "-b", "--beta", type=float, default=DEFAULT_PARAMS["beta"],
        help="Mismatch penalty weight beta (default: 5.0)"
    )
    parser.add_argument(
        "-k", "--max-neighbors", type=int, default=None,
        help="Optional k-nearest neighbor arc pruning limit (default: full graph)"
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Disable Clarke-Wright + DP warm start"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_whatif",
        help="Output folder name relative to repo root (default: results_whatif)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip solver execution and regenerate comparison plots from saved CSVs"
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    out_base_dir = root_dir / args.output_dir
    out_base_dir.mkdir(parents=True, exist_ok=True)

    q1_csv = out_base_dir / "whatif_q1_summary.csv"
    q2_csv = out_base_dir / "whatif_q2_summary.csv"

    if args.plot_only:
        df_q1 = pd.read_csv(q1_csv)
        df_q2 = pd.read_csv(q2_csv)
    else:
        base_data, use_prepared_input = load_inputs(root_dir)
        depots_3 = get_depots_for_num_trucks(root_dir, base_data, 3, use_prepared_input)
        depots_4_add1 = get_augmented_fourth_depot(base_data, depots_3)

        data_pos_1_1 = build_scaled_target_data(base_data, pos_multiplier=1.1, neg_multiplier=1.0)
        data_pos_1_2 = build_scaled_target_data(base_data, pos_multiplier=1.2, neg_multiplier=1.0)
        data_all_2_0 = build_scaled_target_data(base_data, pos_multiplier=2.0, neg_multiplier=2.0)

        # Question 1: Baseline vs. 1.1x and 1.2x positive targets
        q1_scenarios = [
            {
                "question": "Q1",
                "run_id": "q1_01_baseline",
                "name": "Baseline (1.0x Positive Target, 3 Trucks)",
                "short_label": "Baseline\n(1.0x Pos)",
                "data": base_data,
                "depots": depots_3,
                "pos_mult": 1.0,
                "neg_mult": 1.0,
            },
            {
                "question": "Q1",
                "run_id": "q1_02_pos_target_x1_1",
                "name": "1.1x Positive Target (3 Trucks)",
                "short_label": "1.1x Positive\nTarget",
                "data": data_pos_1_1,
                "depots": depots_3,
                "pos_mult": 1.1,
                "neg_mult": 1.0,
            },
            {
                "question": "Q1",
                "run_id": "q1_03_pos_target_x1_2",
                "name": "1.2x Positive Target (3 Trucks)",
                "short_label": "1.2x Positive\nTarget",
                "data": data_pos_1_2,
                "depots": depots_3,
                "pos_mult": 1.2,
                "neg_mult": 1.0,
            },
        ]

        # Question 2: Baseline (replayed from Q1) vs. 2x targets with 4 trucks (+1 depot)
        q2_scenarios = [
            {
                "question": "Q2",
                "run_id": "q2_01_baseline_3trucks",
                "name": "Baseline (1x Targets, 3 Trucks)",
                "short_label": "Baseline\n(1x, 3 Trucks)",
                "data": base_data,
                "depots": depots_3,
                "pos_mult": 1.0,
                "neg_mult": 1.0,
            },
            {
                "question": "Q2",
                "run_id": "q2_03_2x_targets_4trucks_add1",
                "name": "2x Targets (Pos & Neg, 4 Trucks +1 Depot)",
                "short_label": "2x Targets\n(4 Trucks, +1 Depot)",
                "data": data_all_2_0,
                "depots": depots_4_add1,
                "pos_mult": 2.0,
                "neg_mult": 2.0,
            },
        ]

        records = []
        q1_baseline_record = None

        for sc in q1_scenarios:
            rec = run_whatif_scenario(
                sc,
                root_dir=root_dir,
                out_base_dir=out_base_dir,
                time_limit=args.time_limit,
                truck_capacity=args.truck_capacity,
                max_distance=args.max_distance,
                alpha=args.alpha,
                beta=args.beta,
                max_neighbors=args.max_neighbors,
                use_warm_start=not args.no_warm_start,
            )
            if sc["run_id"] == "q1_01_baseline":
                q1_baseline_record = copy.deepcopy(rec)
            records.append(rec)

        for sc in q2_scenarios:
            if sc["run_id"] == "q2_01_baseline_3trucks":
                rec = replay_q1_baseline_for_q2(q1_baseline_record, sc, out_base_dir)
            else:
                rec = run_whatif_scenario(
                    sc,
                    root_dir=root_dir,
                    out_base_dir=out_base_dir,
                    time_limit=args.time_limit,
                    truck_capacity=args.truck_capacity,
                    max_distance=args.max_distance,
                    alpha=args.alpha,
                    beta=args.beta,
                    max_neighbors=args.max_neighbors,
                    use_warm_start=not args.no_warm_start,
                )
            records.append(rec)

        df_all = pd.DataFrame(records)
        df_q1 = df_all[df_all["question"] == "Q1"].reset_index(drop=True)
        df_q2 = df_all[df_all["question"] == "Q2"].reset_index(drop=True)

        df_all.to_csv(out_base_dir / "whatif_all_runs.csv", index=False)
        df_q1.to_csv(q1_csv, index=False)
        df_q2.to_csv(q2_csv, index=False)
        (out_base_dir / "whatif_summary.json").write_text(
            df_all.to_json(orient="records", indent=2)
        )

    # Generate comparison figures matching your sensitivity plot layout
    plot_whatif_comparison(
        df_q1,
        title="What-If Analysis (Q1): Impact of Positive Target Scaling on Objective & MIP Gap",
        xlabel="Positive Target Scenario (Deficit Stations G_i > 0)",
        output_path=out_base_dir / "whatif_q1_positive_targets.png",
    )
    plot_whatif_comparison(
        df_q2,
        title="What-If Analysis (Q2): Baseline vs. 2x Targets with 4th Truck (+1 Depot)",
        xlabel="Scenario Configuration",
        output_path=out_base_dir / "whatif_q2_2x_targets_4th_truck.png",
    )

    print(f"\nAll what-if results and plots saved in: {out_base_dir.resolve()}")


if __name__ == "__main__":
    main()