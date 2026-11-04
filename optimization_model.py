import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB


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


def extract_solution(model, data, depots, x, z, l, D, alpha, beta):
    all_stations = data['stations']
    I = [station['id'] for station in all_stations]
    K = [depot['truck_id'] for depot in depots]
    G = {station['id']: station['G'] for station in all_stations}
    status_labels = {
        GRB.OPTIMAL: 'Optimal',
        GRB.TIME_LIMIT: 'Time limit reached',
        GRB.INFEASIBLE: 'Infeasible',
    }
    status = status_labels.get(model.Status, str(model.Status))
    runtime=model.Runtime
    if model.SolCount == 0:
        raise RuntimeError('Gurobi returned no feasible incumbent. Check the log; do not reuse old results.')
    gap=model.MIPGap if I else 0.0
    bound=model.ObjBound
    result={'solver':'Gurobi','status':status,'runtime_seconds':runtime,'has_feasible_solution':True,
            'gap':gap,'objective_bound':bound,'stations':len(all_stations),'routing_stations':len(I),'trucks':len(K),'routes':{},'station_results':[]}
    distance=0.0
    for k in K:
        depot=f'depot_{k}'
        nodes=I+[depot]
        arcs = []
        for i in nodes:
            for j in nodes:
                if i != j and x[i, j, k].X > 0.5:
                    arcs.append((i, j))
        next_stop = dict(arcs)
        path = [depot]
        current = depot
        for _ in range(len(I)+2):
            if current not in next_stop:
                break
            current = next_stop[current]
            path.append(current)
            if current == depot:
                break
        assert not arcs or (path[-1]==depot and len(path)-1==len(arcs))
        dist=sum(D[i,j,k] for i,j in arcs)
        distance+=dist
        result['routes'][k]={'path':path,'distance_km':dist,
            'transfers':{i:round(z[i, k].X) for i in I if i in path},
            'loads':{i:l[i, k].X for i in I if i in path}}
    for station in all_stations:
        i = station['id']
        target_gap = station['G']
        net = round(sum(z[i, k].X for k in K))
        initial = station['I0']
        final = initial + net
        assert 0 <= final <= station['C']
        result['station_results'].append({
            'id':i, 'I0':initial, 'net_delivery':net, 'final_inventory':final,
            'target':station['I_star'], 'mismatch':abs(target_gap-net),
        })
    assert sum(r['net_delivery'] for r in result['station_results'])==0
    mismatch=sum(r['mismatch'] for r in result['station_results'])
    result.update(total_distance_km=distance,total_mismatch=mismatch,objective=alpha*distance+beta*mismatch,
        baseline_mismatch=sum(abs(v) for v in G.values()),baseline_objective=beta*sum(abs(v) for v in G.values()))
    return result


def solve_citi_bike_rebalancing(
    data, depots, results_dir, license_path, truck_capacity=25,
    max_distance_w=25.0, alpha=1.5, beta=5.0, time_limit=600,
):
    """Build and solve the Midtown routing model, then return its solution."""
    all_stations = data['stations']
    stations = all_stations
    I = [s['id'] for s in stations]
    K = [d['truck_id'] for d in depots]

    C = {s['id']: s['C'] for s in stations}
    I0 = {s['id']: s['I0'] for s in stations}
    G = {s['id']: s['G'] for s in stations}

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

    max_capacity = max(C.values()) if len(C) > 0 else 50
    M = 2 * truck_capacity + max_capacity

    with init_gurobi_env(license_path) as env:
        with gp.Model("CitiBike_Midtown_Rebalancing", env=env) as model:
            model.Params.TimeLimit = time_limit
            model.Params.Threads = 2

            # Decision Variables
            x = {}  # Routing variable x_{ijk} in {0,1}
            z = {}  # Net delivery z_{ik} in Integer
            l = {}  # Load l_{ik} >= 0
            u = {}  # Absolute target mismatch u_i >= 0
            y = {}  # Subtour elimination sequence variable y_{ik} >= 0

            for i in I:
                u[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{i}")

            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]

                l[(depot_k, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"l_{depot_k}_{k}")

                for i in I:
                    z[(i, k)] = model.addVar(vtype=GRB.INTEGER, lb=-truck_capacity, ub=truck_capacity, name=f"z_{i}_{k}")
                    l[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=truck_capacity, name=f"l_{i}_{k}")
                    y[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=len(I), name=f"y_{i}_{k}")

                for i in nodes_k:
                    for j in nodes_k:
                        if i != j:
                            x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

            model.update()

            # Objective Function (Equation 3)
            total_distance = gp.quicksum(D[arc] * x[arc] for arc in x)
            transport_cost = alpha * total_distance
            penalty_cost = gp.quicksum(beta * u[i] for i in I)
            model.setObjective(transport_cost + penalty_cost, GRB.MINIMIZE)

            # (4) Depot Flow Balance
            for k in K:
                depot_k = f"depot_{k}"
                model.addConstr(
                    gp.quicksum(x[(depot_k, j, k)] for j in I) == gp.quicksum(x[(i, depot_k, k)] for i in I),
                    name=f"depot_flow_balance_{k}"
                )
                model.addConstr(
                    gp.quicksum(x[(depot_k, j, k)] for j in I) <= 1,
                    name=f"depot_use_{k}"
                )

            # (5) Route Continuity
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for i in nodes_k:
                    model.addConstr(
                        gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i) == 
                        gp.quicksum(x[(i, j, k)] for j in nodes_k if j != i),
                        name=f"continuity_{i}_{k}"
                    )

            # (6) At most one visit per station
            for i in I:
                model.addConstr(
                    gp.quicksum(x[(j, i, k)] for k in K for j in I + [f"depot_{k}"] if j != i) <= 1,
                    name=f"single_visit_{i}"
                )

            # (7) Transfers bounded by visit
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for i in I:
                    visit_var = gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i)
                    model.addConstr(z[(i, k)] >= -I0[i] * visit_var, name=f"transfer_lb_{i}_{k}")
                    model.addConstr(z[(i, k)] <= (C[i] - I0[i]) * visit_var, name=f"transfer_ub_{i}_{k}")

            # (8) Station Dock Capacity Constraints
            for i in I:
                total_transfer = gp.quicksum(z[(i, k)] for k in K)
                model.addConstr(I0[i] + total_transfer >= 0, name=f"station_min_inv_{i}")
                model.addConstr(I0[i] + total_transfer <= C[i], name=f"station_max_inv_{i}")

            # (9) Absolute Target Mismatch Penalty
            for i in I:
                total_transfer = gp.quicksum(z[(i, k)] for k in K)
                model.addConstr(G[i] - total_transfer <= u[i], name=f"penalty_pos_{i}")
                model.addConstr(G[i] - total_transfer >= -u[i], name=f"penalty_neg_{i}")

            # (10) Vehicle Load Bound
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for i in I:
                    visit_var = gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i)
                    model.addConstr(l[(i, k)] <= truck_capacity * visit_var, name=f"load_ub_{i}_{k}")

            # (11) Big-M Load Flow Balance
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                for j in I:
                    for i in nodes_k:
                        if i != j:
                            model.addConstr(
                                l[(j, k)] - l[(i, k)] + z[(j, k)] >= -M * (1 - x[(i, j, k)]),
                                name=f"load_track_lb_{i}_{j}_{k}"
                            )
                            model.addConstr(
                                l[(j, k)] - l[(i, k)] + z[(j, k)] <= M * (1 - x[(i, j, k)]),
                                name=f"load_track_ub_{i}_{j}_{k}"
                            )

            # (12) Empty Return to Depot
            for k in K:
                depot_k = f"depot_{k}"
                for i in I:
                    model.addConstr(
                        l[(i, k)] <= truck_capacity * (1 - x[(i, depot_k, k)]),
                        name=f"empty_return_{i}_{k}"
                    )

            # (13) Max Travel Budget Limit
            for k in K:
                depot_k = f"depot_{k}"
                nodes_k = I + [depot_k]
                model.addConstr(
                    gp.quicksum(D[(i, j, k)] * x[(i, j, k)] for i in nodes_k for j in nodes_k if i != j) <= max_distance_w,
                    name=f"dist_limit_{k}"
                )

            # (14) SUBTOUR ELIMINATION (MTZ Sequence Ordering Constraints)
            for k in K:
                depot_k = f"depot_{k}"
                for i in I:
                    visit_var = gp.quicksum(x[(j, i, k)] for j in I + [depot_k] if j != i)
                    model.addConstr(y[(i, k)] <= len(I) * visit_var, name=f"seq_ub_{i}_{k}")
                    model.addConstr(y[(i, k)] >= 1 - len(I) * (1 - x[(depot_k, i, k)]), name=f"seq_depot_{i}_{k}")
                    for j in I:
                        if i != j:
                            model.addConstr(
                                y[(j, k)] >= y[(i, k)] + 1 - (len(I)+1) * (1 - x[(i, j, k)]),
                                name=f"seq_step_{i}_{j}_{k}"
                            )


            for k in K:
                model.addConstr(gp.quicksum(z[i,k] for i in I)==0,name=f'explicit_conservation_{k}')

            model.Params.LogFile = str(results_dir / 'gurobi.log')
            model.optimize()
            if model.SolCount == 0:
                raise RuntimeError('No feasible solution was found. Inspect the solver log.')
            solution = extract_solution(model, data, depots, x, z, l, D, alpha, beta)

    solution["parameters"] = {
        "truck_capacity": truck_capacity, "max_distance": max_distance_w,
        "time_limit": time_limit, "alpha": alpha, "beta": beta,
    }
    return solution
