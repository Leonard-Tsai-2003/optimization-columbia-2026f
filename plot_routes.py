import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import contextily as cx
    from pyproj import Transformer
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    logging.warning("contextily or pyproj not found. Install via: pip install contextily pyproj")


DEPOT_NAMES = {
    'Truck_1': 'Financial District Depot',
    'Truck_2': 'Lower East Side Depot'
}


def plot_rebalancing_solution(
    dataset: Dict[str, Any],
    depots: List[Dict[str, Any]],
    solution: Dict[str, Any],
    save_filename: str = "rebalancing_routes_map.png",
    sample_pct: float = None
):
    """
    Plots Lower Manhattan stations, truck depots, and active truck routes
    overlaying a 50% transparent geographic street map background.
    - Title includes the current timestamp (yyyy-mm-dd hh-mm).
    - Station names are shown ONLY if traversed by a truck.
    - Station dots and labels are smaller to prevent clutter.
    - Depots are named 'Financial District Depot' and 'Lower East Side Depot'.
    - Legend displays route distances per truck and total unmet target shortfall.
    """
    stations = dataset['stations']
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Custom User-Agent header required by tile providers
    HTTP_HEADERS = {'User-Agent': 'CitiBikeRebalancingProject/1.0 (Columbia_University_IEOR_E4004)'}

    # Coordinate Transformers
    if HAS_CONTEXTILY:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        inv_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        def to_mercator(lon, lat):
            return transformer.transform(lon, lat)
    else:
        def to_mercator(lon, lat):
            return lon, lat

    # Pre-project station coordinates to Web Mercator meters
    st_coords = {}
    surplus_x, surplus_y = [], []
    deficit_x, deficit_y = [], []
    balanced_x, balanced_y = [], []

    for s in stations:
        x, y = to_mercator(s['lon'], s['lat'])
        st_coords[s['id']] = (x, y)
        if s['G'] < 0:
            surplus_x.append(x)
            surplus_y.append(y)
        elif s['G'] > 0:
            deficit_x.append(x)
            deficit_y.append(y)
        else:
            balanced_x.append(x)
            balanced_y.append(y)

    # Plot compact station dots (s=30 - 35)
    ax.scatter(surplus_x, surplus_y, c='#1f4e78', edgecolor='white', linewidth=0.8, s=35, 
               label='Surplus Station (G_i < 0, Pickup)', zorder=4)
    ax.scatter(deficit_x, deficit_y, c='#d9381e', edgecolor='white', linewidth=0.8, s=35, 
               label='Deficit Station (G_i > 0, Delivery)', zorder=4)
    if balanced_x:
        ax.scatter(balanced_x, balanced_y, c='gray', edgecolor='white', linewidth=0.6, s=20, alpha=0.6, 
                   label='Balanced Station (G_i = 0)', zorder=4)

    # Identify nodes traversed by any truck
    visited_nodes = set()
    for truck_id, rinfo in solution['routes'].items():
        path = rinfo.get('path', [])
        if path and len(path) > 1:
            for node in path:
                if not node.startswith('depot_'):
                    visited_nodes.add(node)

    # Pre-project depots
    depot_coords = {}
    for d in depots:
        dk = f"depot_{d['truck_id']}"
        dx, dy = to_mercator(d['lon'], d['lat'])
        depot_coords[dk] = (dx, dy)

    # Draw active truck routes with directed arrows and record distance in legend
    colors = ['#27ae60', '#8e44ad', '#d35400', '#16a085']
    for idx, (truck_id, rinfo) in enumerate(solution['routes'].items()):
        path = rinfo.get('path', [])
        dist_km = rinfo.get('distance_km', 0.0)
        dname = DEPOT_NAMES.get(truck_id, truck_id)
        
        if not path or len(path) <= 1:
            continue
            
        color = colors[idx % len(colors)]
        transfers = rinfo['transfers']
        
        # Plot route line entry for legend
        ax.plot([], [], color=color, lw=2.5, label=f"Route: {dname} ({dist_km:.2f} km)")
        
        for i in range(len(path) - 1):
            curr_node, next_node = path[i], path[i + 1]
            c1 = depot_coords[curr_node] if curr_node in depot_coords else st_coords[curr_node]
            c2 = depot_coords[next_node] if next_node in depot_coords else st_coords[next_node]
            
            ax.annotate("", xy=c2, xytext=c1,
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", 
                                        color=color, lw=2.8, alpha=0.9), zorder=3)
            
            if next_node in transfers:
                val = transfers[next_node]
                action_str = f"+{val}" if val > 0 else f"{val}"
                ax.annotate(f"{action_str} bikes", xy=c2, textcoords="offset points", 
                            xytext=(6, -12), fontsize=7.5, fontweight='bold', color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.2, alpha=0.9), zorder=6)

    # Station NAME & target demand annotations — ONLY FOR VISITED STATIONS (fontsize=5.5)
    for s in stations:
        if s['id'] in visited_nodes:
            sx, sy = st_coords[s['id']]
            st_name = s.get('name', f"Station_{s['id'][:8]}")
            
            if len(st_name) > 20 and ' & ' in st_name:
                st_name = st_name.replace(' & ', ' &\n')
                
            ax.annotate(f"{st_name}\n(G={s['G']})", (sx, sy), textcoords="offset points", 
                        xytext=(0, 6), ha='center', fontsize=5.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", lw=0.5, alpha=0.85), zorder=7)

    # PLOT NAMED DEPOTS ON TOP-MOST LAYER (zorder=10)
    for d in depots:
        tid = d['truck_id']
        dname = d.get('name', DEPOT_NAMES.get(tid, tid))
        dx, dy = depot_coords[f"depot_{tid}"]
        
        ax.scatter(dx, dy, c='#f1c40f', edgecolor='black', linewidth=1.8, marker='*', s=450, 
                   label=f"Depot: {dname}", zorder=10)
        ax.annotate(f"{dname}", (dx, dy), textcoords="offset points", 
                    xytext=(0, 12), ha='center', fontsize=7.5, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f1c40f", ec="black", lw=1.2, alpha=0.95), zorder=10)

    # Add Total Unmet Target Shortfall to Legend
    summary = solution.get('summary', {})
    unmet_bikes = summary.get('total_unmet_bikes', 0)
    unmet_patch = mpatches.Patch(color='none', label=f"Total Unmet Target Shortfall: {unmet_bikes} bikes")

    # Add 50% transparent high-resolution map tiles
    HIGH_RES_ZOOM = 15
    MAP_ALPHA = 0.5
    
    if HAS_CONTEXTILY:
        map_loaded = False
        try:
            cx.add_basemap(
                ax, 
                source=cx.providers.Esri.WorldStreetMap, 
                zoom=HIGH_RES_ZOOM, 
                alpha=MAP_ALPHA,
                headers=HTTP_HEADERS
            )
            logging.info("Successfully added 50% transparent map background via Esri WorldStreetMap.")
            map_loaded = True
        except Exception as e:
            logging.warning(f"Esri map fetch failed ({e}). Trying CartoDB Voyager...")

        if not map_loaded:
            try:
                cx.add_basemap(
                    ax, 
                    source=cx.providers.CartoDB.Voyager, 
                    zoom=HIGH_RES_ZOOM, 
                    alpha=MAP_ALPHA,
                    headers=HTTP_HEADERS
                )
                logging.info("Successfully added 50% transparent map background via CartoDB Voyager.")
                map_loaded = True
            except Exception as e:
                logging.warning(f"CartoDB map fetch failed ({e}). Trying OpenStreetMap...")

        if not map_loaded:
            try:
                cx.add_basemap(
                    ax, 
                    source=cx.providers.OpenStreetMap.Mapnik, 
                    zoom=HIGH_RES_ZOOM, 
                    alpha=MAP_ALPHA,
                    headers=HTTP_HEADERS
                )
                logging.info("Successfully added 50% transparent map background via OpenStreetMap.")
                map_loaded = True
            except Exception as e:
                logging.error(f"All map tile providers failed ({e}). Falling back to grid view.")
                ax.grid(True, linestyle="--", alpha=0.5)

        # Format Axis Ticks to Lat/Lon degrees
        def format_lon(x, pos):
            lon, _ = inv_transformer.transform(x, 0)
            return f"{lon:.4f}"

        def format_lat(y, pos):
            _, lat = inv_transformer.transform(0, y)
            return f"{lat:.4f}"

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    else:
        ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel("Longitude (°W)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)

    # Generate current timestamp (yyyy-mm-dd hh-mm)
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build Header Title with Current Timestamp & Sample Percentage
    if sample_pct is not None:
        title_text = (
            f"Citi Bike Overnight Rebalancing Routes - Lower Manhattan Map ({current_time_str})\n"
            f"[ Selection: {sample_pct}% of Stations ({len(stations)} Input Stations) ]"
        )
    else:
        title_text = (
            f"Citi Bike Overnight Rebalancing Routes - Lower Manhattan Map ({current_time_str})\n"
            f"[ Selection: {len(stations)} Input Stations ]"
        )

    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=12)

    # Format Legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(unmet_patch)
    ax.legend(handles=handles, loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)
    
    plt.tight_layout()

    # Ensure the output folder exists
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_filename), dpi=300)
    print(f"Route map saved successfully to '{os.path.join(output_dir, save_filename)}'.")
    plt.show()