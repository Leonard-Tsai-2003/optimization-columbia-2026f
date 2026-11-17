import os
import heapq

import gurobipy as gp
import numpy as np
from gurobipy import GRB
import time
import math

def latlon_to_km(lat1, lon1, lat2, lon2):
    return np.round(84 * np.abs(lon1 - lon2) + 111 * np.abs(lat1 - lat2), 3)

def init_gurobi_env(license_path: str = "gurobi.lic") -> gp.Env:
    """Initializes Gurobi Environment using local gurobi.lic license file."""
    abs_lic_path = os.path.abspath(license_path)
    if os.path.exists(abs_lic_path) and not os.environ.get("GRB_LICENSE_FILE"):
        os.environ["GRB_LICENSE_FILE"] = abs_lic_path
        print(f"Loading Gurobi license from: {abs_lic_path}")
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    return env


def extract_solution(model, data, depots, x, z, l, D, alpha, beta, active_ids=None):
    all_stations = data['stations']
    I = active_ids if active_ids is not None else [station['id'] for station in all_stations]
    active_set = set(I)
    K = [depot['truck_id'] for depot in depots]
    G = {station['id']: station['G'] for station in all_stations}
    
    status_labels = {
        GRB.OPTIMAL: 'Optimal',
        GRB.TIME_LIMIT: 'Time limit reached',
        GRB.INFEASIBLE: 'Infeasible',
        GRB.INTERRUPTED: 'Stagnation callback terminated',  # Added status label
    }
    status = status_labels.get(model.Status, f"Status code {model.Status}")
    runtime = model.Runtime
    
    if model.SolCount == 0:
        raise RuntimeError('Gurobi returned no feasible incumbent. Check the log; do not reuse old results.')
    
    gap = model.MIPGap if I else 0.0
    bound = model.ObjBound
    
    result = {
        'solver': 'Gurobi',
        'status': status,
        'runtime_seconds': runtime,
        'has_feasible_solution': True,
        'gap': gap,
        'objective_bound': bound,
        'stations': len(all_stations),
        'routing_stations': len(I),
        'trucks': len(K),
        'routes': {},
        'station_results': []
    }
    
    distance = 0.0
    for k in K:
        depot = f'depot_{k}'
        nodes = I + [depot]
        arcs = []
        for i in nodes:
            for j in nodes:
                if i != j and (i, j, k) in x and x[i, j, k].X > 0.5:
                    arcs.append((i, j))
        
        next_stop = dict(arcs)
        path = [depot]
        current = depot
        for _ in range(len(I) + 2):
            if current not in next_stop:
                break
            current = next_stop[current]
            path.append(current)
            if current == depot:
                break
                
        assert not arcs or (path[-1] == depot and len(path) - 1 == len(arcs))
        dist = sum(D[i, j, k] for i, j in arcs)
        distance += dist
        
        # Round l[i, k].X to integer to eliminate floating-point residual noise
        result['routes'][k] = {
            'path': path,
            'distance_km': dist,
            'transfers': {i: round(z[i, k].X) for i in I if i in path},
            'loads': {i: round(l[i, k].X) for i in I if i in path}
        }

    for station in all_stations:
        i = station['id']
        target_gap = station['G']
        net = round(sum(z[i, k].X for k in K)) if i in active_set else 0
        initial = station['I0']
        final = initial + net
        assert 0 <= final <= station['C']
        result['station_results'].append({
            'id': i, 'I0': initial, 'net_delivery': net, 'final_inventory': final,
            'target': station['I_star'], 'mismatch': abs(target_gap - net),
        })
        
    assert sum(r['net_delivery'] for r in result['station_results']) == 0
    mismatch = sum(r['mismatch'] for r in result['station_results'])
    result.update(
        total_distance_km=distance,
        total_mismatch=mismatch,
        objective=alpha * distance + beta * mismatch,
        baseline_mismatch=sum(abs(v) for v in G.values()),
        baseline_objective=beta * sum(abs(v) for v in G.values())
    )
    return result

def create_stagnation_callback(norel_stall_limit=100.0, mip_stall_limit=200.0):
    """
    Custom Gurobi callback to:
    1. Force NoRel to proceed to MIP search after 100s without improvement.
    2. Terminate standard MIP optimization after 200s without improvement.
    """
    def callback(model, where):
        now = time.time()

        if not hasattr(model, "_cb_state"):
            model._cb_state = {
                "norel_best": float("inf"),
                "norel_last_impr": now,
                "norel_proceeded": False,
                "mip_best": float("inf"),
                "mip_last_impr": now,
            }

        state = model._cb_state

        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
            # Query the current MIP phase (0 = NoRel, 1 = MIP Search, 2 = Solution Improvement)
            try:
                phase = model.cbGet(GRB.Callback.MIP_PHASE)
            except gp.GurobiError:
                phase = 1

            # ------------------------------------------------------------------
            # Phase 0: No Relaxation Heuristic (NoRel)
            # ------------------------------------------------------------------
            if phase == 0:
                try:
                    obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                except gp.GurobiError:
                    obj = float("inf")

                if obj < state["norel_best"] - 1e-6:
                    state["norel_best"] = obj
                    state["norel_last_impr"] = now
                elif (now - state["norel_last_impr"] > norel_stall_limit) and not state["norel_proceeded"]:
                    state["norel_proceeded"] = True
                    print(f"\n[Callback] NoRel heuristic stalled for {norel_stall_limit}s without improvement. Advancing to MIP search...")
                    model.cbProceed()

            # ------------------------------------------------------------------
            # Phase 1 & 2: Standard MIP Search & Branch-and-Cut
            # ------------------------------------------------------------------
            else:
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                else:
                    obj = model.cbGet(GRB.Callback.MIP_OBJBST)

                if obj < state["mip_best"] - 1e-6:
                    state["mip_best"] = obj
                    state["mip_last_impr"] = now
                elif (now - state["mip_last_impr"] > mip_stall_limit) and (state["mip_best"] < float("inf")):
                    print(f"\n[Callback] MIP search stalled for {mip_stall_limit}s without improvement. Terminating solver...")
                    model.terminate()

    return callback


def _optimize_route_transfers(seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set):
    """
    Exact O(m * Q) Dynamic Programming solver that computes optimal integer transfers
    and cumulative loads along a fixed station sequence `seq` for truck `k`.
    Returns (is_feasible, clean_seq, dist, reduction, transfers_dict, loads_dict).
    """
    depot_k = f"depot_{k}"
    if not seq:
        return True, [], 0.0, 0, {}, {}

    # Check arc availability & route distance
    full_path = [depot_k] + list(seq) + [depot_k]
    dist = 0.0
    for a, b in zip(full_path[:-1], full_path[1:]):
        if (a, b, k) not in arc_set:
            return False, [], float("inf"), 0, {}, {}
        dist += float(D[(a, b, k)])
    if dist > max_distance_w + 1e-9:
        return False, [], dist, 0, {}, {}

    m = len(seq)
    Q = truck_capacity
    NEG_INF = -10**9
    dp = np.full((m + 1, Q + 1), NEG_INF, dtype=int)
    parent_l = np.full((m + 1, Q + 1), -1, dtype=int)
    parent_z = np.zeros((m + 1, Q + 1), dtype=int)
    dp[0, 0] = 0

    for r, sid in enumerate(seq):
        if G[sid] > 0:
            u_lim = max_drop[sid]
            for load in range(Q + 1):
                cur = dp[r, load]
                if cur <= NEG_INF // 2:
                    continue
                lim = min(u_lim, load)
                for z_val in range(lim + 1):
                    n_load = load - z_val
                    cand = cur + z_val
                    if cand > dp[r + 1, n_load]:
                        dp[r + 1, n_load] = cand
                        parent_l[r + 1, n_load] = load
                        parent_z[r + 1, n_load] = z_val
        else:
            l_lim = max_pick[sid]
            for load in range(Q + 1):
                cur = dp[r, load]
                if cur <= NEG_INF // 2:
                    continue
                lim = min(l_lim, Q - load)
                for p_val in range(lim + 1):
                    n_load = load + p_val
                    cand = cur + p_val
                    if cand > dp[r + 1, n_load]:
                        dp[r + 1, n_load] = cand
                        parent_l[r + 1, n_load] = load
                        parent_z[r + 1, n_load] = -p_val

    best_red = int(dp[m, 0])
    if best_red <= 0:
        return True, [], 0.0, 0, {}, {}

    # Backtrack optimal transfers
    raw_z = [0] * m
    cur_l = 0
    for r in range(m, 0, -1):
        raw_z[r - 1] = int(parent_z[r, cur_l])
        cur_l = int(parent_l[r, cur_l])

    # Strip zero-transfer stops (triangle inequality preserves or improves distance)
    if any(val == 0 for val in raw_z):
        pruned_seq = [sid for sid, val in zip(seq, raw_z) if val != 0]
        return _optimize_route_transfers(
            pruned_seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
        )

    transfers = {}
    loads = {}
    running_load = 0
    for sid, z_val in zip(seq, raw_z):
        running_load -= z_val
        assert 0 <= running_load <= Q
        transfers[sid] = z_val
        loads[sid] = running_load
    assert running_load == 0

    return True, list(seq), dist, best_red, transfers, loads


def _build_clarke_wright_warm_start(
    I, K, depots, D, G, max_pick, max_drop, truck_capacity, max_distance_w, alpha, beta, arc_set
):
    """
    Constructs a guaranteed-feasible, high-quality initial solution using:
    1. Multi-start spatial partitioning across depots.
    2. Surplus-deficit pair seeding + Clarke-Wright Savings route merging.
    3. Exact DP load-balanced transfer allocation + 2-opt/relocate/insertion local search.
    """
    depot_map = {d['truck_id']: d for d in depots}
    baseline_mismatch = sum(abs(G[i]) for i in I)

    # Generate candidate station partitions across the fleet
    partitions = []
    if len(K) == 1:
        partitions.append({K[0]: list(I)})
    else:
        # Partition 0: Nearest depot assignment
        p0 = {k: [] for k in K}
        for i in I:
            best_k = min(K, key=lambda k: D[(f"depot_{k}", i, k)])
            p0[best_k].append(i)
        partitions.append(p0)

        # Partition 1: Regret-balanced assignment (prevents one truck from overloading)
        if len(K) >= 2:
            p1 = {k: [] for k in K}
            cap_per_truck = math.ceil(len(I) / len(K))
            ranked_stations = sorted(
                I,
                key=lambda i: sorted(D[(f"depot_{k}", i, k)] for k in K)[1]
                            - sorted(D[(f"depot_{k}", i, k)] for k in K)[0],
                reverse=True,
            )
            for i in ranked_stations:
                candidates = sorted(K, key=lambda k: D[(f"depot_{k}", i, k)])
                chosen = next((k for k in candidates if len(p1[k]) < cap_per_truck), candidates[0])
                p1[chosen].append(i)
            partitions.append(p1)

            # Partition 2: Latitude-band assignment ordered from south to north
            k_sorted_lat = sorted(K, key=lambda k: depot_map[k]['lat'])
            p2 = {k: [] for k in K}
            for i in I:
                best_idx = int(np.argmin([D[(f"depot_{k}", i, k)] for k in k_sorted_lat]))
                p2[k_sorted_lat[best_idx]].append(i)
            partitions.append(p2)

    best_overall = None
    best_obj = beta * baseline_mismatch

    for part in partitions:
        routes_seq = {}
        for k in K:
            depot_k = f"depot_{k}"
            sub_I = part[k]
            surplus = [i for i in sub_I if G[i] < 0]
            deficit = [i for i in sub_I if G[i] > 0]

            # Step 1: Seed balanced surplus->deficit mini-routes sorted by mismatch/km ratio
            pair_candidates = []
            for s_id in surplus:
                for d_id in deficit:
                    if (depot_k, s_id, k) in arc_set and (s_id, d_id, k) in arc_set and (d_id, depot_k, k) in arc_set:
                        dist_sd = float(D[(depot_k, s_id, k)] + D[(s_id, d_id, k)] + D[(d_id, depot_k, k)])
                        if dist_sd <= max_distance_w:
                            gain = min(max_pick[s_id], max_drop[d_id])
                            score = gain / max(dist_sd, 0.05)
                            pair_candidates.append((score, s_id, d_id))
            pair_candidates.sort(reverse=True)

            used = set()
            mini_routes = []
            for _, s_id, d_id in pair_candidates:
                if s_id not in used and d_id not in used:
                    mini_routes.append([s_id, d_id])
                    used.add(s_id)
                    used.add(d_id)

            # Step 2: Clarke-Wright Savings merging of mini-routes
            merged = True
            while merged and len(mini_routes) > 1:
                merged = False
                savings_list = []
                for idx_a in range(len(mini_routes)):
                    for idx_b in range(len(mini_routes)):
                        if idx_a == idx_b:
                            continue
                        u_end = mini_routes[idx_a][-1]
                        v_start = mini_routes[idx_b][0]
                        if (u_end, v_start, k) in arc_set and (u_end, depot_k, k) in arc_set and (depot_k, v_start, k) in arc_set:
                            sav = float(D[(u_end, depot_k, k)] + D[(depot_k, v_start, k)] - D[(u_end, v_start, k)])
                            if sav > 0:
                                savings_list.append((sav, idx_a, idx_b))
                savings_list.sort(reverse=True)

                for _, idx_a, idx_b in savings_list:
                    cand_seq = mini_routes[idx_a] + mini_routes[idx_b]
                    ok_a, _, dist_a, red_a, _, _ = _optimize_route_transfers(
                        mini_routes[idx_a], k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                    )
                    ok_b, _, dist_b, red_b, _, _ = _optimize_route_transfers(
                        mini_routes[idx_b], k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                    )
                    ok_c, clean_c, dist_c, red_c, _, _ = _optimize_route_transfers(
                        cand_seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                    )
                    if ok_a and ok_b and ok_c and clean_c:
                        cost_before = alpha * (dist_a + dist_b) - beta * (red_a + red_b)
                        cost_after = alpha * dist_c - beta * red_c
                        if cost_after <= cost_before + 1e-6:
                            new_routes = [r for idx, r in enumerate(mini_routes) if idx not in (idx_a, idx_b)]
                            new_routes.append(clean_c)
                            mini_routes = new_routes
                            merged = True
                            break

            # Pick best single tour for truck k
            best_k_seq = []
            best_k_delta = 0.0
            for r_seq in mini_routes:
                ok, clean_r, dist_r, red_r, _, _ = _optimize_route_transfers(
                    r_seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                )
                if ok and clean_r:
                    delta = alpha * dist_r - beta * red_r
                    if delta < best_k_delta:
                        best_k_delta = delta
                        best_k_seq = clean_r
            routes_seq[k] = best_k_seq

        # Step 3: Local search polish (2-Opt + Greedy Insertion + Relocate)
        improved = True
        while improved:
            improved = False
            # 3a. 2-opt intra-route reversal
            for k in K:
                seq = routes_seq[k]
                if len(seq) < 4:
                    continue
                _, _, cur_dist, cur_red, _, _ = _optimize_route_transfers(
                    seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                )
                cur_val = alpha * cur_dist - beta * cur_red
                for a in range(len(seq) - 1):
                    for b in range(a + 2, len(seq) + 1):
                        cand = seq[:a] + list(reversed(seq[a:b])) + seq[b:]
                        ok, clean_c, dist_c, red_c, _, _ = _optimize_route_transfers(
                            cand, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                        )
                        if ok and clean_c and (alpha * dist_c - beta * red_c < cur_val - 1e-6):
                            routes_seq[k] = clean_c
                            improved = True
                            break
                    if improved:
                        break

            # 3b. Greedy insertion of unvisited active stations
            visited_all = {sid for k in K for sid in routes_seq[k]}
            unvisited = [sid for sid in I if sid not in visited_all]
            best_ins = None
            best_ins_gain = -1e-6
            for k in K:
                seq = routes_seq[k]
                _, _, cur_dist, cur_red, _, _ = _optimize_route_transfers(
                    seq, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                )
                cur_val = alpha * cur_dist - beta * cur_red
                for sid in unvisited:
                    for pos in range(len(seq) + 1):
                        cand = seq[:pos] + [sid] + seq[pos:]
                        ok, clean_c, dist_c, red_c, _, _ = _optimize_route_transfers(
                            cand, k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
                        )
                        if ok and clean_c:
                            gain = (alpha * dist_c - beta * red_c) - cur_val
                            if gain < best_ins_gain:
                                best_ins_gain = gain
                                best_ins = (k, clean_c)
            if best_ins is not None:
                routes_seq[best_ins[0]] = best_ins[1]
                improved = True

        # Evaluate complete fleet solution
        fleet_plan = {}
        total_dist = 0.0
        total_red = 0
        valid_fleet = True
        seen_fleet = set()
        for k in K:
            ok, clean_r, dist_r, red_r, trans_r, loads_r = _optimize_route_transfers(
                routes_seq[k], k, D, G, max_pick, max_drop, truck_capacity, max_distance_w, arc_set
            )
            if not ok or any(sid in seen_fleet for sid in clean_r):
                valid_fleet = False
                break
            seen_fleet.update(clean_r)
            fleet_plan[k] = {
                'seq': clean_r,
                'dist': dist_r,
                'transfers': trans_r,
                'loads': loads_r,
            }
            total_dist += dist_r
            total_red += red_r

        if valid_fleet:
            obj = alpha * total_dist + beta * (baseline_mismatch - total_red)
            if obj < best_obj:
                best_obj = obj
                best_overall = fleet_plan

    return best_overall, best_obj


def solve_citi_bike_rebalancing(
    data, depots, results_dir, license_path, truck_capacity=25,
    max_distance_w=25.0, alpha=1.5, beta=5.0, time_limit=1200,
    max_neighbors=None, use_warm_start=True,
):
    """Build and solve the Midtown routing model with tightened formulation and solver parameters."""
    all_stations = data['stations']
    # 1. Exact reduction: stations with G_i == 0 require zero transfer and are skipped under metric distance
    stations = [s for s in all_stations if s['G'] != 0]
    I = [s['id'] for s in stations]
    K = [d['truck_id'] for d in depots]

    C = {s['id']: s['C'] for s in stations}
    I0 = {s['id']: s['I0'] for s in stations}
    G = {s['id']: s['G'] for s in stations}

    max_pick = {i: min(-G[i], I0[i], truck_capacity) if G[i] < 0 else 0 for i in I}
    max_drop = {i: min(G[i], C[i] - I0[i], truck_capacity) if G[i] > 0 else 0 for i in I}

    depot_map = {d['truck_id']: d for d in depots}

    # Distance Matrix D_ij
    D = {}
    for k in K:
        nodes = I + [f"depot_{k}"]
        coords = {s['id']: (s['lat'], s['lon']) for s in stations}
        coords[f"depot_{k}"] = (depot_map[k]['lat'], depot_map[k]['lon'])

        for i in nodes:
            for j in nodes:
                if i != j:
                    lat1, lon1 = coords[i]
                    lat2, lon2 = coords[j]
                    D[(i, j, k)] = latlon_to_km(lat1, lon1, lat2, lon2)

    # Tightest upper bound on maximum stops per truck within distance budget W_k
    nonzero_dists = [val for val in D.values() if val > 0]
    min_step_km = min(nonzero_dists) if nonzero_dists else 0.05
    n_max_stops = {
        k: min(len(I), max(1, int(np.floor(max_distance_w / min_step_km))))
        for k in K
    }

    with init_gurobi_env(license_path) as env:
        with gp.Model("CitiBike_Midtown_Rebalancing", env=env) as model:
            # ==============================================================================
            # OPTIMIZED GUROBI PARAMETERS (Warm-start & dual-bound accelerated)
            # ==============================================================================
            model.Params.TimeLimit = time_limit
            model.Params.MIPGap = 0.10          # 10% target gap
            model.Params.Method = 1             # Dual Simplex for fast root LP solves
            model.Params.Presolve = 2           # Aggressive presolve on tightened active graph
            model.Params.NoRelHeurTime = (
                min(300, max(15, time_limit // 15)) if use_warm_start
                else min(300, max(60, time_limit // 5))
            )
            model.Params.MIPFocus = 3           # Focus on optimality & dual bound improvement
            model.Params.Heuristics = 0.15      # Balanced heuristic search effort
            model.Params.CutPasses = 10         # Stronger root/tree cut passes to lift lower bound
            model.Params.RINS = 10              # RINS neighborhood search around warm start
            model.Params.Symmetry = 2           # Aggressive symmetry breaking
            model.Params.Threads = 0            # Utilize all CPU cores

            # Decision Variables
            x = {}  # Routing variable x_{ijk} in {0,1}
            z = {}  # Sign-locked integer transfer z_{ik}
            l = {}  # Load l_{ik} >= 0
            y = {}  # Subtour elimination sequence variable y_{ik} >= 0

            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]

                l[(depot_k, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"l_{depot_k}_{k}")

                for i in I:
                    # 2. Sign-locked & magnitude-bounded transfer variables
                    z[(i, k)] = model.addVar(
                        vtype=GRB.INTEGER, lb=-max_pick[i], ub=max_drop[i], name=f"z_{i}_{k}"
                    )
                    l[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=truck_capacity, name=f"l_{i}_{k}")
                    y[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n_max_stops[k], name=f"y_{i}_{k}")

                # Optional k-nearest neighbor shortlist for sparse acceleration
                allowed_neighbors = None
                if max_neighbors is not None and max_neighbors < len(I):
                    allowed_neighbors = {}
                    for i in I:
                        nearest = heapq.nsmallest(
                            max_neighbors,
                            (j for j in I if j != i),
                            key=lambda j: D[(i, j, k)]
                        )
                        allowed_neighbors[i] = set(nearest)

                # 3. Distance-based & depot-direction arc pruning
                for i in nodes_k:
                    for j in nodes_k:
                        if i == j:
                            continue
                        # First stop from empty depot must be surplus (G_j < 0); last stop before empty return must be deficit (G_i > 0)
                        if i == depot_k and G[j] > 0:
                            continue
                        if j == depot_k and G[i] < 0:
                            continue
                        if allowed_neighbors is not None and i != depot_k and j != depot_k:
                            if j not in allowed_neighbors[i] and i not in allowed_neighbors[j]:
                                continue
                        d_start = D[(depot_k, i, k)] if i != depot_k else 0
                        d_end = D[(j, depot_k, k)] if j != depot_k else 0
                        if D[(i, j, k)] + d_start + d_end <= max_distance_w:
                            x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

            model.update()

            # Precompute visit indicator expressions v_{ik} = sum_j x_{jik}
            visit_expr = {}
            for k in K:
                nodes_k = I + [f"depot_{k}"]
                for i in I:
                    visit_expr[(i, k)] = gp.quicksum(
                        x[(j, i, k)] for j in nodes_k if j != i and (j, i, k) in x
                    )

            # Objective Function (Exact reformulation eliminating u_i variables & 2|I| constraints)
            baseline_mismatch = sum(abs(s['G']) for s in all_stations)
            total_distance = gp.quicksum(D[arc] * x[arc] for arc in x)
            transport_cost = alpha * total_distance
            mismatch_reduction = gp.quicksum(
                z[(i, k)] if G[i] > 0 else -z[(i, k)]
                for k in K for i in I
            )
            penalty_cost = beta * (baseline_mismatch - mismatch_reduction)
            model.setObjective(transport_cost + penalty_cost, GRB.MINIMIZE)

            # (4) Depot Flow Balance
            for k in K:
                depot_k = f"depot_{k}"
                model.addConstr(
                    gp.quicksum(x[(depot_k, j, k)] for j in I if (depot_k, j, k) in x) == 
                    gp.quicksum(x[(i, depot_k, k)] for i in I if (i, depot_k, k) in x),
                    name=f"depot_flow_balance_{k}"
                )
                model.addConstr(
                    gp.quicksum(x[(depot_k, j, k)] for j in I if (depot_k, j, k) in x) <= 1,
                    name=f"depot_use_{k}"
                )

            # (5) Route Continuity
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for i in nodes_k:
                    model.addConstr(
                        gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i and (j, i, k) in x) == 
                        gp.quicksum(x[(i, j, k)] for j in nodes_k if j != i and (i, j, k) in x),
                        name=f"continuity_{i}_{k}"
                    )

            # (6) Single Visit per Station
            for i in I:
                model.addConstr(
                    gp.quicksum(visit_expr[(i, k)] for k in K) <= 1,
                    name=f"single_visit_{i}"
                )

            # (7) Tightened Sign-Locked Transfer Bounds & Minimum-Transfer Valid Cuts
            for k in K:
                for i in I:
                    v_ik = visit_expr[(i, k)]
                    if G[i] > 0:
                        model.addConstr(z[(i, k)] <= max_drop[i] * v_ik, name=f"transfer_ub_{i}_{k}")
                        model.addConstr(z[(i, k)] >= v_ik, name=f"min_transfer_lb_{i}_{k}")
                    else:
                        model.addConstr(z[(i, k)] >= -max_pick[i] * v_ik, name=f"transfer_lb_{i}_{k}")
                        model.addConstr(-z[(i, k)] >= v_ik, name=f"min_transfer_ub_{i}_{k}")

            # (8) Station Dock Capacity Constraints (tightened by G_i bounds)
            for i in I:
                total_transfer = gp.quicksum(z[(i, k)] for k in K)
                if G[i] > 0:
                    model.addConstr(total_transfer <= max_drop[i], name=f"station_max_inv_{i}")
                else:
                    model.addConstr(-total_transfer <= max_pick[i], name=f"station_min_inv_{i}")

            # (10) Vehicle Load Bounds
            for k in K:
                for i in I:
                    model.addConstr(l[(i, k)] <= truck_capacity * visit_expr[(i, k)], name=f"load_ub_{i}_{k}")

            # (11) Arc-Specific Asymmetric Tightened Load Flow Tracking Constraints
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for j in I:
                    m_load_pos = truck_capacity + max_drop[j]
                    m_load_neg = truck_capacity + max_pick[j]
                    for i in nodes_k:
                        if i != j and (i, j, k) in x:
                            model.addConstr(
                                l[(j, k)] - l[(i, k)] + z[(j, k)] >= -m_load_neg * (1 - x[(i, j, k)]),
                                name=f"load_track_lb_{i}_{j}_{k}"
                            )
                            model.addConstr(
                                l[(j, k)] - l[(i, k)] + z[(j, k)] <= m_load_pos * (1 - x[(i, j, k)]),
                                name=f"load_track_ub_{i}_{j}_{k}"
                            )

            # (12) Empty Return to Depot
            for k in K:
                depot_k = f"depot_{k}"
                for i in I:
                    if (i, depot_k, k) in x:
                        model.addConstr(
                            l[(i, k)] <= truck_capacity * (1 - x[(i, depot_k, k)]),
                            name=f"empty_return_{i}_{k}"
                        )

            # (13) Max Travel Distance Limit
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                model.addConstr(
                    gp.quicksum(D[(i, j, k)] * x[(i, j, k)] for i in nodes_k for j in nodes_k if i != j and (i, j, k) in x) <= max_distance_w,
                    name=f"dist_limit_{k}"
                )

            # (14) Lifted Desrochers-Laporte MTZ Subtour Elimination Constraints
            for k in K:
                depot_k = f"depot_{k}"
                n_max = n_max_stops[k]
                for i in I:
                    v_ik = visit_expr[(i, k)]
                    x_start = x[(depot_k, i, k)] if (depot_k, i, k) in x else 0
                    model.addConstr(
                        y[(i, k)] <= n_max * v_ik - (n_max - 1) * x_start,
                        name=f"seq_ub_{i}_{k}"
                    )
                    model.addConstr(
                        y[(i, k)] >= 2 * v_ik - x_start,
                        name=f"seq_lb_{i}_{k}"
                    )
                    for j in I:
                        if i != j and (i, j, k) in x:
                            rev_term = (n_max - 2) * x[(j, i, k)] if (j, i, k) in x else 0
                            model.addConstr(
                                y[(j, k)] >= y[(i, k)] + 1 - n_max * (1 - x[(i, j, k)]) + rev_term,
                                name=f"seq_step_{i}_{j}_{k}"
                            )

            # Valid Metric Tour-Length Lower Bound Cuts (tightens LP relaxation distance bound)
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                half_perim_terms = []
                for i in I:
                    in_dists = [D[(j, i, k)] for j in nodes_k if j != i and (j, i, k) in x]
                    out_dists = [D[(i, j, k)] for j in nodes_k if j != i and (i, j, k) in x]
                    if in_dists and out_dists:
                        d_min_i = 0.5 * (min(in_dists) + min(out_dists))
                        half_perim_terms.append(d_min_i * visit_expr[(i, k)])
                if half_perim_terms:
                    model.addConstr(
                        gp.quicksum(D[(i, j, k)] * x[(i, j, k)] for i in nodes_k for j in nodes_k if i != j and (i, j, k) in x)
                        >= gp.quicksum(half_perim_terms),
                        name=f"valid_metric_tour_lb_{k}"
                    )

            for k in K:
                model.addConstr(gp.quicksum(z[i, k] for i in I) == 0, name=f'explicit_conservation_{k}')

            # Inject Clarke-Wright + Local Search MIP Warm Start
            if use_warm_start and I:
                t_ws = time.time()
                fleet_plan, ws_obj = _build_clarke_wright_warm_start(
                    I, K, depots, D, G, max_pick, max_drop, truck_capacity,
                    max_distance_w, alpha, beta, set(x.keys())
                )
                if fleet_plan is not None:
                    for var in x.values():
                        var.Start = 0.0
                    for var in z.values():
                        var.Start = 0.0
                    for var in l.values():
                        var.Start = 0.0
                    for var in y.values():
                        var.Start = 0.0

                    for k in K:
                        depot_k = f"depot_{k}"
                        plan = fleet_plan[k]
                        seq = plan['seq']
                        if seq:
                            full_path = [depot_k] + seq + [depot_k]
                            for a, b in zip(full_path[:-1], full_path[1:]):
                                x[(a, b, k)].Start = 1.0
                            for order, sid in enumerate(seq, 1):
                                z[(sid, k)].Start = float(plan['transfers'][sid])
                                l[(sid, k)].Start = float(plan['loads'][sid])
                                y[(sid, k)].Start = float(order)
                    print(f"[WarmStart] Clarke-Wright + DP/LocalSearch found feasible start: obj={ws_obj:.4f} in {time.time() - t_ws:.2f}s")

            # Create the stagnation callback (50s for NoRel, 100s for MIP)
            stagnation_cb = create_stagnation_callback(
                norel_stall_limit=50.0,
                mip_stall_limit=100.0
            )

            model.Params.LogFile = str(results_dir / 'gurobi.log')
            model.optimize(stagnation_cb)

            if model.SolCount == 0:
                raise RuntimeError('No feasible solution was found. Inspect the solver log.')
            solution = extract_solution(model, data, depots, x, z, l, D, alpha, beta, active_ids=I)

    solution["parameters"] = {
        "truck_capacity": truck_capacity, "max_distance": max_distance_w,
        "time_limit": time_limit, "alpha": alpha, "beta": beta,
    }
    return solution