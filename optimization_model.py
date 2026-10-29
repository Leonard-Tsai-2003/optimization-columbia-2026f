import os
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Any
from data_preprocessing import latlon_to_km


def init_gurobi_env(license_path: str = "gurobi.lic") -> gp.Env:
    """Initializes Gurobi Environment using local gurobi.lic license file."""
    abs_lic_path = os.path.abspath(license_path)
    if os.path.exists(abs_lic_path):
        os.environ["GRB_LICENSE_FILE"] = abs_lic_path
        print(f"Loading Gurobi license from: {abs_lic_path}")
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    return env


def solve_citi_bike_rebalancing(
    data: Dict[str, Any],
    depots: List[Dict[str, Any]],
    truck_capacity: int = 25,
    max_distance_w: float = 25.0,
    alpha: float = 1.5,
    beta: float = 5.0,
    license_path: str = "gurobi.lic"
) -> Dict[str, Any]:
    """Builds and solves the overnight rebalancing MILP (Milestone 3) with Subtour Elimination."""
    stations = data['stations']
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

    env = init_gurobi_env(license_path=license_path)
    model = gp.Model("CitiBike_Lower_Manhattan_Rebalancing", env=env)
    model.Params.TimeLimit = 120

    # Decision Variables
    x = {}  # Routing variable x_{ijk} in {0,1}
    z = {}  # Net delivery z_{ik} in Integer
    l = {}  # Load l_{ik} >= 0
    u = {}  # Target shortfall u_i >= 0
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
    transport_cost = gp.quicksum(alpha * D[(i, j, k)] * x[(i, j, k)] 
                                for k in K 
                                for i in I + [f"depot_{k}"] 
                                for j in I + [f"depot_{k}"] if i != j)
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

    # (9) Unmet Target Shortfall Penalty
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
                        y[(j, k)] >= y[(i, k)] + 1 - len(I) * (1 - x[(i, j, k)]),
                        name=f"seq_step_{i}_{j}_{k}"
                    )

    model.optimize()

    solution = {
        'status': model.Status,
        'objective_value': None,
        'routes': {},
        'unmet_inventory': {},
        'summary': {}
    }

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        solution['objective_value'] = model.ObjVal
        
        for k in K:
            depot_k = f"depot_{k}"
            nodes_k = I + [depot_k]
            active_arcs = [(i, j) for i in nodes_k for j in nodes_k if i != j and x[(i, j, k)].X > 0.5]
            
            # Reconstruct ordered path starting from depot
            route_path = []
            if active_arcs:
                curr = depot_k
                route_path.append(curr)
                total_dist = 0.0
                while True:
                    next_node = None
                    for i, j in active_arcs:
                        if i == curr:
                            next_node = j
                            total_dist += D[(i, j, k)]
                            break
                    if next_node is None:
                        break
                    route_path.append(next_node)
                    curr = next_node
                    if curr == depot_k:
                        break
            else:
                total_dist = 0.0

            solution['routes'][k] = {
                'path': route_path,
                'distance_km': round(total_dist, 2),
                'transfers': {i: int(round(z[(i, k)].X)) for i in I if (i, k) in z and abs(z[(i, k)].X) > 0.01},
                'loads': {i: int(round(l[(i, k)].X)) for i in I if (i, k) in l and (i in route_path)}
            }

        total_unmet = 0
        for i in I:
            val = int(round(u[i].X))
            solution['unmet_inventory'][i] = val
            total_unmet += val

        solution['summary'] = {
            'total_cost': round(model.ObjVal, 2),
            'transport_cost': round(transport_cost.getValue(), 2),
            'penalty_cost': round(penalty_cost.getValue(), 2),
            'total_unmet_bikes': total_unmet
        }

    return solution