# Citi Bike Overnight Rebalancing Optimization System (Midtown Manhattan)

An operational optimization framework developed in Python using **Gurobi** (`gurobipy`) to solve the overnight bike rebalancing problem across 165 stations in Midtown Manhattan. The system integrates historical GBFS feed snapshots, regularized Poisson regression demand forecasting, a fleet-conserving dynamic programming target allocation, tightened Mixed-Integer Linear Programming (MILP) with lifted Desrochers-Laporte Miller-Tucker-Zemlin (MTZ) subtour elimination, a multi-start Clarke-Wright Savings + exact Dynamic Programming warm-start heuristic, custom dynamic callback controls for heuristic and search phase stagnation, and geographic route visualization using `matplotlib`, `contextily`, and `pyproj`.

---

## 1. Project Overview & Key Upgrades

This project addresses the overnight Citi Bike rebalancing problem between 14:00 PM and 06:00 AM prior to the morning peak (06:00 AM – 10:00 AM). 

### Key Features & Model Refinements
* **Midtown Scope (14th St to 59th St) & Active Station Filtering**: Evaluates all 165 active stations in Midtown Manhattan without downsampling, while analytically pruning the 38 zero-imbalance stations ($G_i = 0$) from the routing graph ($165 \to 127$ active stations) with zero optimality loss under Manhattan metric distances.
* **Effective Dock Capacity ($C_i$)**: Calculated as available bikes + available docks at midnight. Unavailable or damaged bikes/docks are excluded as they cannot recover overnight.
* **Demand Forecasting & Fleet-Conserving Allocation**: Combines a regularized Poisson regression model for morning pickups ($P_i$) and station-weekday averages for returns ($R_i$). Target inventories ($I_i^*$) are optimized via dynamic programming to minimize replayed blocked trips while strictly preserving total fleet inventory ($\sum_{i} I_i^* = \sum_{i} I_i^0$).
* **Optimized Depot Locations**: Discrete $p$-median facility placement over station coordinates (default $m = 3$ trucks) yields three depot hubs at **Broadway & W 25th St** (Truck 1), **W 47th St & 6th Ave** (Truck 2), and **Port Authority Bus Terminal – W 41st St & 8th Ave** (Truck 3).
* **Tightened MILP Formulation & Valid Cuts**: Incorporates sign-locked transfer bounds, exact objective elimination of mismatch penalty variables $u_i$, depot-direction arc pruning, arc-specific asymmetric Big-M load bounds, lifted 2-cycle MTZ cuts, minimum-transfer cuts ($|z_{ik}| \ge v_{ik}$), and metric half-perimeter tour-length lower bounds.
* **Clarke-Wright + Exact DP Warm-Start Pipeline**: Generates a guaranteed-feasible initial incumbent at $t = 0$ via depot-partitioned Clarke-Wright Savings route merging, $O(mQ)$ Dynamic Programming load-feasible transfer allocation, and 2-Opt / greedy insertion local search.
* **Dynamic Stagnation Callbacks**: Automated phase transition control to advance out of the No Relaxation (NoRel) heuristic after 50 seconds without objective improvement, and terminate MIP search after 100 seconds without improvement.

---

## 2. Mathematical Model Formulation

### Objective Function

The MILP model minimizes total operational cost, trading off truck travel distance against penalties for station target inventory mismatches (reformulated directly over sign-locked transfers):

$$\min \alpha \sum_{k \in \mathcal{K}} \sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} + \beta \left( \sum_{i \in \mathcal{I}} |G_i| - \sum_{k \in \mathcal{K}} \left( \sum_{i \in \mathcal{I}^+} z_{ik} - \sum_{i \in \mathcal{I}^-} z_{ik} \right) \right)$$

### Sets and Parameters

| Notation | Description | Value / Definition |
| :--- | :--- | :--- |
| $\mathcal{I}$ | Full set of Midtown Manhattan stations | $165$ stations ($127$ active with $G_i \neq 0$, $38$ balanced with $G_i = 0$) |
| $\mathcal{I}_{\text{active}}$ | Active imbalanced stations ($\mathcal{I}^+ \cup \mathcal{I}^-$) | $n = 127$ stations ($\mathcal{I}^+ = \{i \mid G_i > 0\}$, $\mathcal{I}^- = \{i \mid G_i < 0\}$) |
| $\mathcal{K}$ | Fleet of rebalancing trucks | $|\mathcal{K}| = 3$ (`Truck_1`, `Truck_2`, `Truck_3`) |
| $S_k$ | Starting and ending depot for truck $k$ | Depot 1: $(40.7429^\circ \text{N}, -73.9892^\circ \text{W})$, Depot 2: $(40.7582^\circ \text{N}, -73.9820^\circ \text{W})$, Depot 3: $(40.7564^\circ \text{N}, -73.9900^\circ \text{W})$ |
| $\mathcal{V}_k$ | Node set for truck $k$ | $\mathcal{V}_k = \mathcal{I}_{\text{active}} \cup \{S_k\}$ |
| $\mathcal{A}_k$ | Feasible directed arcs for truck $k$ | Pruned by distance budget $W_k$ and depot first/last stop direction |
| $I_i^0$ | Initial available inventory at midnight | Extracted from GBFS status snapshot |
| $C_i$ | Effective station dock capacity | Available bikes + available docks at midnight |
| $I_i^*$ | Target inventory at station $i$ | Prepared via fleet-conserving dynamic programming |
| $G_i$ | Net required inventory change | $G_i = I_i^* - I_i^0$ ($G_i > 0$ deficit; $G_i < 0$ surplus) |
| $U_i, L_i$ | Station max delivery / pickup bounds | $U_i = \min(G_i, C_i - I_i^0, Q)$ for $G_i > 0$; $L_i = \min(-G_i, I_i^0, Q)$ for $G_i < 0$ |
| $D_{ij}$ | Grid distance approximation (km) | $D_{ij} = \text{round}\left(84 \cdot |\text{lon}_i - \text{lon}_j| + 111 \cdot |\text{lat}_i - \text{lat}_j|, 3\right)$ |
| $Q$ | Maximum truck payload capacity | $Q = 25$ bikes |
| $W_k$ | Maximum distance allowance per truck | $W_k = 25.0$ km |
| $\alpha$ | Distance travel cost coefficient | $\$1.50 / \text{km}$ |
| $\beta$ | Target mismatch penalty coefficient | $\$5.00 / \text{bike}$ shortfall or surplus |
| $M^+_{jk}, M^-_{jk}$ | Arc-specific asymmetric load Big-M | $M^+_{jk} = Q + U_j$, $M^-_{jk} = Q + L_j$ |

### Decision Variables

* $x_{ijk} \in \{0, 1\}$: Equals 1 if truck $k$ travels directly from node $i$ to node $j$; 0 otherwise.
* $z_{ik} \in \mathbb{Z} \cap [-L_i, U_i]$: Sign-locked net bikes delivered to station $i$ by truck $k$ ($z_{ik} \ge 0$ if $G_i > 0$; $z_{ik} \le 0$ if $G_i < 0$).
* $l_{ik} \in [0, Q]$: Load of truck $k$ immediately after visiting station $i$.
* $y_{ik} \in [0, N_{\max, k}]$: Lifted MTZ continuous visit-order variable for subtour elimination.

---

## 3. Key Constraints Summary

1. **Depot Flow & Single Visit Constraints** (with $v_{ik} = \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{jik}$):
   $$\sum_{j \in \mathcal{I}^-} x_{S_k j k} = \sum_{i \in \mathcal{I}^+} x_{i S_k k} \le 1, \quad \forall k \in \mathcal{K}$$
   $$\sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k} = \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{i j k}, \quad \forall i \in \mathcal{I}_{\text{active}}, k \in \mathcal{K}$$
   $$\sum_{k \in \mathcal{K}} v_{ik} \le 1, \quad \forall i \in \mathcal{I}_{\text{active}}$$

2. **Sign-Locked Transfer Bounds & Minimum-Transfer Cuts**:
   $$v_{ik} \le z_{ik} \le U_i v_{ik}, \quad \forall i \in \mathcal{I}^+, k \in \mathcal{K}$$
   $$-L_i v_{ik} \le z_{ik} \le -v_{ik}, \quad \forall i \in \mathcal{I}^-, k \in \mathcal{K}$$

3. **Truck Load Propagation & Return Conditions**:
   $$l_{S_k k} = 0, \quad 0 \le l_{ik} \le Q v_{ik}, \quad \forall i \in \mathcal{I}_{\text{active}}, k \in \mathcal{K}$$
   $$-M^-_{jk}(1 - x_{ijk}) \le l_{jk} - l_{ik} + z_{jk} \le M^+_{jk}(1 - x_{ijk}), \quad \forall i \in \mathcal{V}_k \setminus \{j\}, j \in \mathcal{I}_{\text{active}}, k \in \mathcal{K}$$
   $$l_{ik} \le Q(1 - x_{i S_k k}), \quad \forall i \in \mathcal{I}^+, k \in \mathcal{K}$$
   $$\sum_{i \in \mathcal{I}_{\text{active}}} z_{ik} = 0, \quad \forall k \in \mathcal{K}$$

4. **Distance Budget, Metric Tour Lower Bounds & Lifted MTZ Subtour Elimination**:
   $$\sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} \le W_k, \quad \sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} \ge \sum_{i \in \mathcal{I}_{\text{active}}} d^{\min}_i v_{ik}, \quad \forall k \in \mathcal{K}$$
   $$2 v_{ik} - x_{S_k i k} \le y_{ik} \le N_{\max, k} v_{ik} - (N_{\max, k} - 1) x_{S_k i k}, \quad \forall i \in \mathcal{I}_{\text{active}}, k \in \mathcal{K}$$
   $$y_{jk} \ge y_{ik} + 1 - N_{\max, k}(1 - x_{ijk}) + (N_{\max, k} - 2) x_{jik}, \quad \forall i, j \in \mathcal{I}_{\text{active}}, i \neq j, k \in \mathcal{K}$$

---

## 4. Optimization Control & Warm-Start Pipeline

Before calling `model.optimize()`, the solver executes `_build_clarke_wright_warm_start` to inject a feasible initial solution (`var.Start`), followed by `create_stagnation_callback`:

```
                 ┌──────────────────────────────────────────┐
                 │  Multi-Start Clarke-Wright + Exact DP    │
                 │  & 2-Opt / Insertion Local Search Start  │
                 └────────────────────┬─────────────────────┘
                                      │ (Inject Feasible .Start)
                                      ▼
                        ┌────────────────────────────┐
                        │   Phase 0: NoRel Heuristic │
                        └─────────────┬──────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
       New Best Solution                             No Improvement for
       Found (< 50s stall)                              50 Seconds
                │                                           │
                ▼                                           ▼
      Reset Stagnation Timer                      model.cbProceed()
                │                        (Advance to Branch & Cut Search)
                └─────────────────────┬─────────────────────┘
                                      │
                                      ▼
                        ┌────────────────────────────┐
                        │ Phase 1 & 2: MIP Search    │
                        └─────────────┬──────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
       New Best Solution                             No Improvement for
       Found (< 100s stall)                             100 Seconds
                │                                           │
                ▼                                           ▼
      Reset Stagnation Timer                       model.terminate()
                                                (Return Best Incumbent)
```

### Stagnation Parameters
* **No Relaxation Heuristic Stagnation (50s)**: If NoRel (`GRB.Callback.MIP_PHASE == 0`) fails to discover a better integer feasible solution for 50 consecutive seconds, `model.cbProceed()` forces Gurobi to transition into LP relaxation and Branch-and-Cut MIP search immediately.
* **MIP Search Stagnation (100s)**: If standard MIP search (`GRB.Callback.MIP_PHASE >= 1`) makes no progress on the best incumbent solution for 100 consecutive seconds, `model.terminate()` cleanly stops solver execution and exports the current best incumbent solution.

---

## 5. Setup & Installation

### Step 1: Install Dependencies

Install the required Python environment dependencies:

```bash
pip install gurobipy contextily pyproj matplotlib pandas requests numpy
```

Or via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Gurobi License

Ensure a valid Gurobi license file (`gurobi.lic`) is located in your user home directory or in the root project folder:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

---

## 6. Running the Solver & CLI Usage

Run the optimization pipeline using `main.py`. The script provides command-line flags to adjust execution time limits, truck capacity, max distance allowances, objective penalties, fleet size, and optional $k$-nearest neighbor arc pruning.

```bash
python main.py [OPTIONS]
```

### Supported CLI Flags

| Parameter Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p`, `--time-limit` | `float` | `600.0` | Maximum solver time limit in seconds. |
| `-Q`, `--truck-capacity` | `int` | `25` | Maximum bike payload per truck ($Q$). |
| `-w`, `--max-distance` | `float` | `25.0` | Travel distance limit per truck in kilometers ($W_k$). |
| `-a`, `--alpha` | `float` | `1.5` | Travel cost per kilometer ($\alpha$). |
| `-b`, `--beta` | `float` | `5.0` | Inventory mismatch penalty per bike ($\beta$). |
| `-k`, `--max-neighbors` | `int` | `None` | Optional $k$-nearest neighbor arc limit (`None` = exact full graph). |
| `--no-warm-start` | `flag` | `False` | Disable the Clarke-Wright + local search MIP warm start. |
| `-m`, `--num-trucks` | `int` | `3` | Number of rebalancing trucks / depots ($|\mathcal{K}|$). |

### Execution Examples

* **Solve Exact Midtown Model with Clarke-Wright Warm Start:**
  ```bash
  python main.py -p 1200
  ```

* **Fast Sparse-Graph Run (Top-25 Nearest Neighbors per Station):**
  ```bash
  python main.py -p 600 -k 25
  ```

* **Custom Fleet Size, Payload, and Travel Allowances:**
  ```bash
  python main.py -p 1200 -m 3 -Q 30 -w 30.0 -a 2.0 -b 6.0
  ```

---

## 7. Empirical Results & Benchmark Performance

Evaluating the MILP model on the 165-station Midtown dataset with the 3-truck fleet demonstrates significant reductions in target inventory mismatches:

| Metric | Baseline (No Rebalancing) | Gurobi MILP | Absolute Change | % Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Abs. Inventory Mismatch** | $884\text{ bikes}$ | **$88\text{ bikes}$** | $-796\text{ bikes}$ | **$-90.0\%$** |
| **Approx. Total Travel Distance** | $0.000\text{ km}$ | **$72.286\text{ km}$** | $+72.286\text{ km}$ | N/A |
| **Travel Cost Term** ($\alpha = 1.5$) | $0.0000$ | **$108.4290$** | $+108.4290$ | N/A |
| **Mismatch Penalty Term** ($\beta = 5.0$) | $4,420.0000$ | **$440.0000$** | $-3,980.0000$ | **$-90.0\%$** |
| **Total Objective Value** | **$4,420.0000$** | **$548.4290$** | **$-3,871.5710$** | **$-87.6\%$** |

### Vehicle Performance Summary

| Truck ID | Depot Location | Coordinates | Stops Made | Total Dist. (km) | Budget Used | Bikes Picked Up / Delivered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Truck 1** | Broadway & W 25 St | $(40.7429, -73.9892)$ | $41$ | $23.999\text{ km}$ | $96.0\%$ | $170\text{ bikes}$ |
| **Truck 2** | W 47 St & 6 Ave | $(40.7582, -73.9820)$ | $42$ | $24.810\text{ km}$ | $99.2\%$ | $114\text{ bikes}$ |
| **Truck 3** | Port Authority Bus Terminal (W 41 St & 8 Ave) | $(40.7564, -73.9900)$ | $35$ | $23.477\text{ km}$ | $93.9\%$ | $114\text{ bikes}$ |
| **Total** | — | — | **$118$** | **$72.286\text{ km}$** | — | **$398\text{ bikes}$** |

> **Key Operational Finding:** Distance allowance $W_k = 25.0\text{ km}$ acts as the binding constraint limiting further mismatch reduction, rather than instantaneous vehicle payload capacity $Q$.

---

## 8. Operational Limitations & Future Extensions

1. **Street-Network Routing & Travel Time Constraints**:
   Current distances rely on Manhattan grid projections ($D_{ij}$). Integrating real-world street network directions (e.g., OpenStreetMap / OSRM) and travel/service time estimates will improve route schedule fidelity before the 06:00 AM cutoff.
2. **Dynamic Intra-Day Rebalancing**:
   The current framework optimizes static overnight rebalancing. Future work involves extending the formulation into a multi-period dynamic MIP that accounts for real-time customer trip activity during operational hours.