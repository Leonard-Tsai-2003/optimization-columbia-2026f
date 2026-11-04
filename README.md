# Citi Bike Overnight Rebalancing Optimization System (Midtown Manhattan)

An operational optimization framework developed in Python using **Gurobi** (`gurobipy`) to solve the overnight bike rebalancing problem across 165 stations in Midtown Manhattan. The system integrates historical GBFS feed snapshots, regularized Poisson regression demand forecasting, a fleet-conserving dynamic programming target allocation, Mixed-Integer Linear Programming (MILP) with Miller-Tucker-Zemlin (MTZ) subtour elimination, and geographic route visualization using `matplotlib`, `contextily`, and `pyproj`.

---

## 1. Project Overview & Key Upgrades

This project addresses the overnight Citi Bike rebalancing problem between 14:00 PM and 06:00 AM prior to the morning peak (06:00 AM – 10:00 AM). 

### Key Features & Model Refinements
* **Midtown Scope (14th St to 59th St)**: Comprehensive inclusion of all 165 active stations in Midtown Manhattan without downsampling, selected due to peak activity volume, activity per station, and absolute net flow density across historical trip records.
* **Effective Dock Capacity ($C_i$)**: Calculated as available bikes + available docks at midnight. Unavailable or damaged bikes/docks are excluded as they cannot recover overnight.
* **Demand Forecasting & Fleet-Conserving Allocation**: Combines a regularized Poisson regression model for morning pickups ($P_i$) and station-weekday averages for returns ($R_i$). Target inventories ($I_i^*$) are optimized via dynamic programming to minimize replayed blocked trips while strictly preserving total fleet inventory ($\sum_{i} I_i^* = \sum_{i} I_i^0$).
* **Optimized Depot Locations**: Discrete 2-median facility placement over station coordinates yields dual depot hubs at **Broadway & W 25th St** (Truck 1) and **W 47th St & 6th Ave** (Truck 2).
* **MTZ Subtour Elimination**: Polynomial Miller-Tucker-Zemlin formulations for exact subtour prevention across multi-stop routing tours.

---

## 2. Mathematical Model Formulation

### Objective Function

The MILP model minimizes total operational cost, trading off truck travel distance against penalties for station target inventory mismatches:

$$\min \alpha \sum_{k \in \mathcal{K}} \sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} + \beta \sum_{i \in \mathcal{I}} u_i$$

### Sets and Parameters

| Notation | Description | Value / Definition |
| :--- | :--- | :--- |
| $\mathcal{I}$ | Set of Midtown Manhattan stations | $n = 165$ stations |
| $\mathcal{K}$ | Fleet of rebalancing trucks | $|\mathcal{K}| = 2$ (`Truck_1`, `Truck_2`) |
| $S_k$ | Starting and ending depot for truck $k$ | Depot 1: $(40.7429^\circ \text{N}, -73.9892^\circ \text{W})$, Depot 2: $(40.7582^\circ \text{N}, -73.9820^\circ \text{W})$ |
| $\mathcal{V}_k$ | Node set for truck $k$ | $\mathcal{V}_k = \mathcal{I} \cup \{S_k\}$ |
| $\mathcal{A}_k$ | Feasible directed arcs for truck $k$ | $\mathcal{A}_k = \{(i,j) \mid i, j \in \mathcal{V}_k, i \neq j\}$ |
| $I_i^0$ | Initial available inventory at midnight | Extracted from GBFS status snapshot |
| $C_i$ | Effective station dock capacity | Available bikes + available docks at midnight |
| $I_i^*$ | Target inventory at station $i$ | Prepared via fleet-conserving dynamic programming |
| $G_i$ | Net required inventory change | $G_i = I_i^* - I_i^0$ ($G_i > 0$ deficit; $G_i < 0$ surplus) |
| $D_{ij}$ | Grid distance approximation (km) | $D_{ij} = \text{round}\left(84 \cdot |\text{lon}_i - \text{lon}_j| + 111 \cdot |\text{lat}_i - \text{lat}_j|, 3\right)$ |
| $Q$ | Maximum truck payload capacity | $Q = 25$ bikes |
| $W_k$ | Maximum distance allowance per truck | $W_k = 25.0$ km |
| $\alpha$ | Distance travel cost coefficient | $\$1.50 / \text{km}$ |
| $\beta$ | Target mismatch penalty coefficient | $\$5.00 / \text{bike}$ shortfall or surplus |
| $M$ | Load-tracking Big-M constant | $M = 170$ ($M = 2Q + \max_{i \in \mathcal{I}} C_i$) |

### Decision Variables

* $x_{ijk} \in \{0, 1\}$: Equals 1 if truck $k$ travels directly from node $i$ to node $j$; 0 otherwise.
* $z_{ik} \in \mathbb{Z}, -Q \le z_{ik} \le Q$: Net bikes delivered to station $i$ by truck $k$ ($z_{ik} < 0$ indicates pickup).
* $l_{ik} \in [0, Q]$: Load of truck $k$ immediately after visiting station $i$.
* $y_{ik} \in [0, n]$: MTZ continuous visit-order variable for subtour elimination.
* $u_i \in \mathbb{Z}_{\ge 0}$: Absolute inventory deviation from target stock at station $i$.
* $v_{ik} = \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{jik}$: Auxiliary binary indicator for a visit to station $i$ by truck $k$.

---

## 3. Key Constraints Summary

1. **Depot Flow & Single Visit Constraints**:
   $$\sum_{j \in \mathcal{I}} x_{S_k j k} = \sum_{i \in \mathcal{I}} x_{i S_k k} \le 1, \quad \forall k \in \mathcal{K}$$
   $$\sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k} = \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{i j k}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$\sum_{k \in \mathcal{K}} v_{ik} \le 1, \quad \forall i \in \mathcal{I}$$

2. **Inventory Transfer Limits & Mismatch Tracking**:
   $$-I_i^0 v_{ik} \le z_{ik} \le (C_i - I_i^0) v_{ik}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$u_i \ge G_i - \sum_{k \in \mathcal{K}} z_{ik}, \quad u_i \ge -G_i + \sum_{k \in \mathcal{K}} z_{ik}, \quad \forall i \in \mathcal{I}$$

3. **Truck Load Propagation & Return Conditions**:
   $$l_{S_k k} = 0, \quad 0 \le l_{ik} \le Q v_{ik}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$-M(1 - x_{ijk}) \le l_{jk} - l_{ik} + z_{jk} \le M(1 - x_{ijk}), \quad \forall i \in \mathcal{V}_k \setminus \{j\}, j \in \mathcal{I}, k \in \mathcal{K}$$
   $$l_{ik} \le Q(1 - x_{i S_k k}), \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$\sum_{i \in \mathcal{I}} z_{ik} = 0, \quad \forall k \in \mathcal{K}$$

4. **Distance Budget & Subtour Elimination**:
   $$\sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} \le W_k, \quad \forall k \in \mathcal{K}$$
   $$0 \le y_{ik} \le n v_{ik}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$y_{ik} \ge 1 - n(1 - x_{S_k i k}), \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$y_{jk} \ge y_{ik} + 1 - (n+1)(1 - x_{ijk}), \quad \forall i, j \in \mathcal{I}, i \neq j, k \in \mathcal{K}$$

---

## 4. Setup & Installation

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

## 5. Running the Solver & CLI Usage

Run the optimization pipeline using `main.py`. The script provides command-line flags to adjust execution time limits, truck capacity, max distance allowances, and objective penalties.

```bash
python main.py [OPTIONS]
```

### Supported CLI Flags

| Parameter Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p`, `--time_limit` | `int` | `600` | Gurobi time limit in seconds. |
| `-Q`, `--truck_capacity` | `int` | `25` | Maximum bike payload per truck ($Q$). |
| `-w`, `--max_distance` | `float` | `25.0` | Travel distance limit per truck in kilometers ($W_k$). |
| `--alpha` | `float` | `1.5` | Travel cost per kilometer ($\alpha$). |
| `--beta` | `float` | `5.0` | Inventory mismatch penalty per bike ($\beta$). |

### Execution Examples

* **Solve 165-Station Midtown Model (1800s time limit):**
  ```bash
  python main.py -p 1800 -Q 25 -w 25.0
  ```

* **Fast Benchmark Run (600s time limit):**
  ```bash
  python main.py -p 600
  ```

---

## 6. Empirical Results & Benchmark Performance

Evaluating the MILP model on the 165-station Midtown dataset demonstrates significant reductions in target inventory mismatches:

| Metric | Baseline (No Rebalancing) | MILP Optimized Model | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Absolute Inventory Mismatch** | `884 bikes` | **`294 bikes`** | **$-66.7\%$** ($-590$ bikes) |
| **Approx. Total Distance** | `0.00 km` | **`47.73 km`** | $+47.73$ km |
| **Travel Cost Term ($\alpha \cdot D$)** | `$0.00` | **`$71.60`** | $+71.60$ |
| **Mismatch Penalty ($\beta \cdot u$)** | `$4,420.00` | **`$1,470.00`** | $-66.7\%$ |
| **Total Objective Value** | `$4,420.00` | **`$1,541.60`** | **$-65.1\%$** |

### Vehicle Performance Summary

* **Truck 1 (Broadway & W 25th St)**: Serves **48 stops**, covers **24.735 km** ($98.9\%$ of distance budget), and transfers **176 bikes**.
* **Truck 2 (W 47th St & 6th Ave)**: Serves **40 stops**, covers **22.995 km** ($92.0\%$ of distance budget), and transfers **119 bikes**.
* **Total Operations**: Visits **88 stations** and moves **295 bikes** across Midtown Manhattan.

> **Key Operational Finding:** Distance allowance $W_k = 25.0\text{ km}$ acts as the binding constraint limiting further mismatch reduction, rather than instantaneous vehicle payload capacity $Q$.

---

## 7. Operational Limitations & Future Extensions

1. **Street-Network Routing & Travel Time Constraints**:
   Current distances rely on Manhattan grid projections ($D_{ij}$). Integrating real-world street network directions (e.g., OpenStreetMap / OSRM) and travel/service time estimates will improve route schedule fidelity before the 06:00 AM cutoff.
2. **Dynamic Intra-Day Rebalancing**:
   The current framework optimizes static overnight rebalancing. Future work involves extending the formulation into a multi-period dynamic MIP that accounts for real-time customer trip activity during operational hours.