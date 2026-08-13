import glob
import json
import lxml.etree as ET
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
from math import asin, atan2, cos, radians, sin, sqrt
from pathlib import Path

from branca.element import MacroElement
from jinja2 import Template

# Resolve paths relative to the script location so the output always stays in the project folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Directory containing TCX files (organized by year)
TCX_BASE_DIR = SCRIPT_DIR / "tcx"

# Maximum speed to display in color gradient (knots)
MAX_DISPLAY_SPEED = 15.0  # adjust as needed
OUTPUT_FILE = SCRIPT_DIR / "sailing_tracks_speed_arrows.html"


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def parse_tcx_file(file_path):
    """Parse a TCX file into a list of (lat, lon, speed) points."""
    print(f"Processing {file_path}")
    try:
        tree = ET.parse(file_path)
        ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
        points = []
        last_time, last_lat, last_lon = None, None, None

        for tp in tree.findall(".//ns:Trackpoint", ns):
            lat = tp.findtext("ns:Position/ns:LatitudeDegrees", namespaces=ns)
            lon = tp.findtext("ns:Position/ns:LongitudeDegrees", namespaces=ns)
            time = tp.findtext("ns:Time", namespaces=ns)

            if lat and lon and time:
                lat, lon = float(lat), float(lon)
                t = datetime.fromisoformat(time.replace("Z", "+00:00"))

                speed = 0.0
                if last_time is not None:
                    dt = (t - last_time).total_seconds()
                    if dt > 0:
                        dist = haversine(last_lat, last_lon, lat, lon)  # meters
                        speed = (dist / dt) * 1.94384  # m/s → knots

                points.append((lat, lon, speed))
                last_lat, last_lon, last_time = lat, lon, t

        return points
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return []


runs = []  # List of (year, points) tuples
all_points = []  # (lat, lon, speed_in_knots)
all_years = set()

# Collect TCX files from year subdirectories
if TCX_BASE_DIR.exists():
    for year_dir in sorted(TCX_BASE_DIR.glob("*/"), key=lambda x: x.name):
        year = year_dir.name
        try:
            # Validate year is numeric
            int(year)
            all_years.add(year)
            for file in sorted(year_dir.glob("*.tcx")):
                points = parse_tcx_file(file)
                if points:
                    runs.append((year, points))
                    all_points.extend(points)
        except ValueError:
            # Skip non-numeric directory names
            continue

all_years = sorted(all_years)

# Initialize map
m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)

# Add a real satellite basemap and a street map option
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satellite",
    control=True,
    prefer_canvas=True,
).add_to(m)

folium.TileLayer(
    tiles="CartoDB positron",
    name="Street",
    control=True,
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Draw tracks colored by speed, with arrows for direction
if all_points:
    import matplotlib

    cmap = matplotlib.colormaps["jet"]

    for year, run in runs:
        for i in range(len(run) - 1):
            lat1, lon1, s1 = run[i]
            lat2, lon2, s2 = run[i + 1]
            avg_speed = (s1 + s2) / 2
            scaled_speed = min(avg_speed / MAX_DISPLAY_SPEED, 1.0) ** 0.5
            color = mcolors.to_hex(cmap(scaled_speed))

            # Add line with year embedded in options
            line = folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=4,
                opacity=1.0,
                tooltip=f"Year {year}: {avg_speed:.1f} kn",
            )
            line.options['_year'] = year
            line.add_to(m)

            if i % 50 == 0:
                rotation_deg = (180 / 3.14159) * atan2(lat2 - lat1, lon2 - lon1)
                marker = folium.RegularPolygonMarker(
                    location=(lat2, lon2),
                    fill_color=color,
                    number_of_sides=3,
                    radius=6,
                    rotation=rotation_deg,
                    fill_opacity=1.0,
                )
                marker.options['_year'] = year
                marker.options['Year'] = str(year)
                marker.options['year'] = str(year)
                marker.add_to(m)

    min_lat = min(point[0] for point in all_points)
    max_lat = max(point[0] for point in all_points)
    min_lon = min(point[1] for point in all_points)
    max_lon = max(point[1] for point in all_points)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(20, 20))

# Add legend
colormap_html = f"""
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 200px; height: 20px;
     z-index:9999;
     background: linear-gradient(to right, blue, cyan, green, yellow, red);
     border:1px solid black;
     ">
</div>
<div style="position: fixed;
     bottom: 30px; left: 50px; width: 200px;
     z-index:9999;
     text-align: justify;
     font-size:12px;
     ">
     <span style="float:left;">0 kn</span>
     <span style="float:right;">{MAX_DISPLAY_SPEED:.1f} kn</span>
</div>
"""

legend = MacroElement()
legend._template = Template(f"""{{% macro html(this, kwargs) %}}{colormap_html}{{% endmacro %}}""")
m.get_root().add_child(legend)

# Save map
m.save(str(OUTPUT_FILE))

# Inject animation script with year filtering
if runs:
    # Organize route data by year
    route_data_by_year = {}
    for year, run in runs:
        if year not in route_data_by_year:
            route_data_by_year[year] = []
        route_data_by_year[year].append([[lat, lon] for lat, lon, _ in run])
    
    html_path = OUTPUT_FILE
    html = html_path.read_text(encoding="utf-8")
    map_name = m.get_name()
    
    animation_script = f"""
<script>
window.addEventListener('load', function() {{
    const map = {map_name};
    const routeDataByYear = {json.dumps(route_data_by_year)};
    const allYears = {json.dumps(all_years)};
    if (!map || !window.L || !routeDataByYear || !allYears.length) {{
        return;
    }}

    // Build mapping of layers to years
    const layersByYear = {{}};
    allYears.forEach(year => {{
        layersByYear[year] = [];
    }});
    
    // Populate layersByYear by checking all layers; marker objects may serialize as 'Year' instead of '_year'.
    Object.values(map._layers).forEach(layer => {{
        if (layer.options) {{
            const yearValue = layer.options._year ?? layer.options.Year ?? layer.options.year;
            if (yearValue !== undefined && yearValue !== null) {{
                const year = yearValue.toString();
                if (layersByYear[year]) {{
                    layersByYear[year].push(layer);
                }}
            }}
        }}
    }});

    const marker = L.circleMarker([0, 0], {{
        radius: 8,
        color: '#ff3b30',
        fillColor: '#ff3b30',
        fillOpacity: 0.95,
        weight: 2
    }}).addTo(map);

    let runIndex = 0;
    let pointIndex = 0;
    let timer = null;
    let paused = false;
    let selectedYear = allYears.length > 0 ? allYears[0] : null;
    let currentRuns = routeDataByYear[selectedYear] || [];

    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.right = '20px';
    overlay.style.bottom = '20px';
    overlay.style.zIndex = '1000';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.gap = '8px';
    overlay.style.alignItems = 'flex-end';
    document.body.appendChild(overlay);

    const status = document.createElement('div');
    status.style.background = 'rgba(255,255,255,0.95)';
    status.style.padding = '6px 10px';
    status.style.borderRadius = '8px';
    status.style.boxShadow = '0 2px 8px rgba(0,0,0,0.25)';
    status.style.fontSize = '12px';
    status.textContent = 'Year: ' + selectedYear + ' • Animating run 1 / ' + currentRuns.length;
    overlay.appendChild(status);

    const yearButtonsContainer = document.createElement('div');
    yearButtonsContainer.style.display = 'flex';
    yearButtonsContainer.style.gap = '4px';
    yearButtonsContainer.style.flexWrap = 'wrap';
    yearButtonsContainer.style.justifyContent = 'flex-end';
    yearButtonsContainer.style.background = 'rgba(255,255,255,0.95)';
    yearButtonsContainer.style.padding = '8px';
    yearButtonsContainer.style.borderRadius = '8px';
    yearButtonsContainer.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    overlay.appendChild(yearButtonsContainer);

    allYears.forEach(year => {{
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = year;
        btn.style.background = selectedYear === year ? '#007AFF' : '#ffffff';
        btn.style.color = selectedYear === year ? '#ffffff' : '#000000';
        btn.style.border = '1px solid #d0d7de';
        btn.style.borderRadius = '6px';
        btn.style.padding = '4px 8px';
        btn.style.cursor = 'pointer';
        btn.style.fontSize = '11px';
        btn.style.fontWeight = selectedYear === year ? 'bold' : 'normal';
        btn.addEventListener('click', function() {{
            selectedYear = year;
            currentRuns = routeDataByYear[year] || [];
            runIndex = 0;
            pointIndex = 0;
            updateYearButtons();
            updateLayerOpacity();
            updateStatusLabel();
            if (currentRuns.length === 0) {{
                marker.setLatLng([0, 0]);
                paused = true;
                pausePlayButton.textContent = 'Play';
            }} else {{
                advance();
            }}
        }});
        yearButtonsContainer.appendChild(btn);
    }});

    function updateYearButtons() {{
        const buttons = yearButtonsContainer.querySelectorAll('button');
        buttons.forEach(btn => {{
            if (btn.textContent === selectedYear) {{
                btn.style.background = '#007AFF';
                btn.style.color = '#ffffff';
                btn.style.fontWeight = 'bold';
            }} else {{
                btn.style.background = '#ffffff';
                btn.style.color = '#000000';
                btn.style.fontWeight = 'normal';
            }}
        }});
    }}

    function updateLayerOpacity() {{
        // Update opacity for all layers based on selected year
        allYears.forEach(year => {{
            const layers = layersByYear[year] || [];
            const isActive = year === selectedYear;
            const opacity = isActive ? 1.0 : 0.10;
            layers.forEach(layer => {{
                const isMarker = !!(layer.options && layer.options.numberOfSides && layer.options.radius && layer._path);

                if (layer.setOpacity) {{
                    if (isMarker) {{
                        layer.setOpacity(isActive ? 1.0 : 0.0);
                    }} else {{
                        layer.setOpacity(opacity);
                    }}
                }}
                if (layer.setStyle) {{
                    if (isMarker) {{
                        layer.setStyle({{ opacity: isActive ? 1.0 : 0.0, fillOpacity: isActive ? 1.0 : 0.0, strokeOpacity: isActive ? 1.0 : 0.0 }});
                    }} else {{
                        layer.setStyle({{ opacity: opacity, fillOpacity: opacity, strokeOpacity: opacity }});
                    }}
                }}
                if (layer.options) {{
                    layer.options.opacity = isMarker ? (isActive ? 1.0 : 0.0) : opacity;
                    layer.options.fillOpacity = isMarker ? (isActive ? 1.0 : 0.0) : opacity;
                    layer.options.strokeOpacity = isMarker ? (isActive ? 1.0 : 0.0) : opacity;
                }}

                // Keep the faded route visible for inactive years, but hide only the triangle direction markers.
                if (layer._path && isMarker) {{
                    layer._path.style.display = isActive ? '' : 'none';
                    layer._path.style.opacity = isActive ? 1.0 : 0.0;
                    layer._path.style.fillOpacity = isActive ? 1.0 : 0.0;
                    layer._path.style.strokeOpacity = isActive ? 1.0 : 0.0;
                }}
            }});
        }});
    }}

    const pausePlayButton = document.createElement('button');
    pausePlayButton.type = 'button';
    pausePlayButton.textContent = 'Pause';
    pausePlayButton.style.background = '#ffffff';
    pausePlayButton.style.border = '1px solid #d0d7de';
    pausePlayButton.style.borderRadius = '999px';
    pausePlayButton.style.padding = '6px 12px';
    pausePlayButton.style.cursor = 'pointer';
    pausePlayButton.style.fontSize = '12px';
    pausePlayButton.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    overlay.appendChild(pausePlayButton);

    function updateStatusLabel() {{
        const total = currentRuns.length;
        const label = 'Year: ' + selectedYear + ' • Animating run ' + (runIndex + 1) + ' / ' + total + ' • point ' + (pointIndex + 1);
        status.textContent = label;
    }}

    function scheduleNext() {{
        if (paused) {{
            return;
        }}
        timer = window.setTimeout(advance, 120);
    }}

    function advance() {{
        if (paused) {{
            return;
        }}

        if (!currentRuns || currentRuns.length === 0) {{
            runIndex = 0;
            pointIndex = 0;
            return;
        }}

        const currentRun = currentRuns[runIndex];
        if (!currentRun || !currentRun[pointIndex]) {{
            runIndex = 0;
            pointIndex = 0;
            return;
        }}

        const point = currentRun[pointIndex];
        marker.setLatLng([point[0], point[1]]);

        updateStatusLabel();

        pointIndex += 1;
        if (pointIndex >= currentRun.length) {{
            pointIndex = 0;
            runIndex = (runIndex + 1) % currentRuns.length;
        }}

        scheduleNext();
    }}

    pausePlayButton.addEventListener('click', function() {{
        paused = !paused;
        pausePlayButton.textContent = paused ? 'Play' : 'Pause';
        if (!paused) {{
            scheduleNext();
        }}
    }});

    // Initialize layer opacity
    updateLayerOpacity();

    if (currentRuns.length > 0) {{
        advance();
    }}
}});
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", animation_script + "\n</body>")
        html_path.write_text(html, encoding="utf-8")

print(f"Speed-colored tracks with arrows and animation saved to {OUTPUT_FILE}")

