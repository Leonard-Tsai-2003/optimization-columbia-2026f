# Citi Bike Overnight Rebalancing Optimization System (Lower Manhattan)

An automated operational optimization framework developed in Python using **Gurobi** (`gurobipy`) to solve the overnight bike rebalancing problem in Lower Manhattan. The system integrates live **GBFS (General Bikeshare Feed Specification)** API data, historical trip datasets, a Mixed-Integer Linear Programming (MILP) model, and geographic route visualizers using `matplotlib`, `contextily`, and `pyproj`.

---

## 1. Prerequisites & Setup

Before running or compiling `main.py`, ensure your local environment satisfies the dependencies and holds a valid Gurobi license.

### Step 1: Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install gurobipy contextily pyproj matplotlib pandas requests numpy
```

Or if you have a `requirements.txt` file in your workspace:

```bash
pip install -r requirements.txt
```

### Step 2: Add Gurobi License File

A valid Gurobi license file (`gurobi.lic`) is required to run the optimization solver.
* Copy your `gurobi.lic` file directly into the **project root directory** (the same folder containing `main.py` and `optimization_model.py`).

---

## 2. System Parameter Configurations

### Target Inventory ($I_i^*$)

For each station $i$, the target overnight inventory $I_i^*$ is calculated using the morning peak demand balance formula:

$$
I_i^* = \min\left\{C_i, \max\left\{0, \text{round}\left(I_i^0 + P_i - R_i\right)\right\}\right\}
$$

Where:
* $C_i$: Total dock capacity at station $i$.
* $I_i^0$: Initial observed inventory at station $i$ (scraped from live GBFS feeds).
* $P_i$: Average weekday morning peak (6:00 AM – 10:00 AM) pickups/departures.
* $R_i$: Average weekday morning peak (6:00 AM – 10:00 AM) returns/arrivals.
* $G_i = I_i^* - I_i^0$: Net target delivery/pickup quantity required at station $i$.
  * If $G_i < 0$: Surplus station (pickup site).
  * If $G_i > 0$: Deficit station (delivery site).
  * If $G_i = 0$: Balanced station (no inventory transfer required).

Historical trip counts ($P_i$ and $R_i$) are aggregated from October 2025 trip data using a multi-key mapping dictionary that matches GBFS `station_id`, `legacy_id`, and `short_name`.

### Truck & Fleet Parameters

| Parameter | Notation | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Number of Vehicles** | $\vert K \vert$ | `2` | Total rebalancing trucks available (`Truck_1` and `Truck_2`). |
| **Vehicle Capacity** | $Q$ | `25 bikes` | Maximum number of bikes a truck can carry simultaneously. |
| **Max Distance Allowance** | $W_k$ | `25.0 km` | Upper limit on total travel distance per truck per shift. |
| **Travel Cost Coefficient** | $\alpha$ | `$1.50 / km` | Operational cost multiplier per kilometer traveled. |
| **Penalty Cost Coefficient** | $\beta$ | `$5.00 / bike` | Penalty cost per unit of unmet target inventory shortfall. |

### Depot Locations

The fleet operates out of two strategically designated depots in Lower Manhattan:
1. **Financial District Depot (`Truck_1`)**: $(40.708^\circ \text{N}, -74.011^\circ \text{W})$
2. **Lower East Side Depot (`Truck_2`)**: $(40.720^\circ \text{N}, -73.985^\circ \text{W})$

Each truck starts its tour at its assigned depot empty ($L_0 = 0$), visits stations to pick up or deliver bikes within capacity $Q$, and returns empty to its home depot upon completing its route.

---

## 3. Running `main.py` & Command-Line Arguments

The main execution script supports command-line interface (CLI) arguments using `argparse`, allowing dynamic control over station sampling, dataset paths, and operational parameters.

```bash
python main.py [OPTIONS]
```

### Supported Command-Line Parameters

| Flag | Long Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `-p` | `--sample_pct` | `float` | `None` | Percentage of active Lower Manhattan stations to include in the model (e.g., `10` for 10%, `25` for 25%, `100` for 100%). |
| `-n` | `--num_stations` | `int` | `12` | Fixed count of stations to sample. Used if `--sample_pct` is omitted. |
| `-t` | `--trip_data` | `str` | `202510-citibike-tripdata` | Folder directory or specific CSV/ZIP file path containing historical trip data. |
| `-Q` | `--truck_capacity` | `int` | `25.0` | Maximum bikes carried by one truck. |
| `-w` | `--max_dist` | `float` | `25.0` | Maximum route travel distance limit per truck in kilometers ($W_k$). |
| `--alpha` | `--alpha` | `float` | `1.5` | Operating transportation cost in dollars per kilometer ($\alpha$). |
| `--beta` | `--beta` | `float` | `5.0` | Penalty cost in dollars per unit unmet target bike shortfall ($\beta$). |

### Command Execution Examples

* **Sample 25% of Lower Manhattan Stations:**
  ```bash
  python main.py --sample_pct 25
  ```

* **Run Full Scale (100% of Lower Manhattan Stations):**
  ```bash
  python main.py --sample_pct 100
  ```

* **Process Custom Trip Data Directory & Custom Distance/Penalty Parameters:**
  ```bash
  python main.py --trip_data ./my_trip_data_folder --sample_pct 15 --max_dist 30.0 --beta 8.0
  ```

---

## 4. Data Pipeline & Geographic Filtering

To prevent cross-river routing errors (e.g., placing stations across the East River in Brooklyn), the preprocessing pipeline enforces a two-step boundary filter:
1. **GBFS Region Check**: Verifies station membership via `region_id` (Manhattan `region_id = 71`).
2. **Ray-Casting Polygon Mask**: Applies a precise landmass polygon covering Lower Manhattan south of 14th Street to strictly exclude waterfronts, East River segments, and Brooklyn neighborhood overlaps (DUMBO, Brooklyn Heights, Navy Yard).

---

## 5. WHAT IS NEEDED TO DO

The following enhancements are identified for future development and model extensions:

### 1. Enhanced Target Inventory Prediction
Instead of estimating morning peak demand from simple historical averages ($I_i^* = \min\{C_i, \max\{0, \text{round}(I_i^0 + P_i - R_i)\}\}$), implement an **advanced predictive machine learning pipeline** (e.g., XGBoost, LSTM, or Spatial-Temporal Graph Neural Networks). The predictive model should incorporate:
* Real-time weather forecasts (temperature, precipitation).
* Day-of-week and seasonal variations.
* Local event schedules and public transit outage patterns.

### 2. Rigorous Depot Location Justification
Currently, depot locations are statically assigned to Financial District and Lower East Side coordinates. Future work must **mathematically justify depot placement and fleet allocation**:
* Formulate a facility location model ($p$-median or $p$-center problem) to find optimal depot sites that minimize total deadhead travel distance.
* Evaluate the trade-offs between single-depot vs. multi-depot configurations under varying fleet sizes ($\vert K \vert$).

### 3. Geographic Scope & Sampling Validity Analysis
The current model scope is restricted to Lower Manhattan (South of 14th St). Key research questions to address:
* **Full-System Inclusion vs. Sub-Region Decomposition**: Evaluate whether solving all NYC stations simultaneously in a single MILP formulation is computationally tractable, or if spatial clustering (e.g., $k$-means / spectral clustering by neighborhood) is necessary.
* **Sampling Validity**: Perform sensitivity analysis to determine whether random station sampling ($n\%$) statistically preserves the true net demand distribution, or if graph-based clustering provides superior sub-problem approximations.