import json
import re
import statistics
import lxml.etree as ET
import folium
from datetime import datetime, timedelta, timezone
from math import asin, atan2, ceil, cos, radians, sin, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

from branca.element import MacroElement
from jinja2 import Template

# Resolve paths relative to the script location so the output always stays in the project folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Directory containing TCX files (organized by year)
TCX_BASE_DIR = SCRIPT_DIR / "tcx"
SRT_BASE_DIR = SCRIPT_DIR / "srt"

# Maximum speed for the color gradient (knots). Used as-is when AUTO_MAX_DISPLAY_SPEED
# is False, or as a fallback if there's no data to measure. When AUTO_MAX_DISPLAY_SPEED
# is True (the default), this gets replaced by the fastest speed actually recorded
# across all tracks - a fixed guess here tends to sit well above what any real
# session hits, which flattens every genuinely fast section into yellow/green.
MAX_DISPLAY_SPEED = 15.0
AUTO_MAX_DISPLAY_SPEED = True
OUTPUT_FILE = SCRIPT_DIR / "sailing_tracks_speed_arrows.html"

# --- Efficiency / file-size knobs -------------------------------------------------
# Ramer-Douglas-Peucker simplification tolerances. A point is kept if it deviates
# from the straight line between its neighbors by more than EITHER of these -
# positionally, or in speed. The speed check matters a lot for sailing: a boat
# sitting nearly still, or slowing/accelerating on an otherwise straight line,
# barely moves position-wise, so a position-only simplifier throws those points
# away and erases exactly the slow/stopped moments you want to see. Set either
# to 0 to disable that half of the check; set both to 0 to disable simplification.
SIMPLIFY_TOLERANCE_M = 4.0
SIMPLIFY_SPEED_TOLERANCE_KN = 0.75

# Rounding applied to values before they're embedded as JSON. 6 decimal places of
# lat/lon is ~0.11m precision, which is far tighter than GPS accuracy anyway.
COORD_PRECISION = 6
SPEED_PRECISION = 2
# ------------------------------------------------------------------------------------


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def clean_speed_points(points):
    """Drop clearly invalid sailing speeds based on practical thresholds and outlier analysis."""
    if not points:
        return [], 0

    speeds = [speed for _, _, speed in points if speed > 0]
    if not speeds:
        return points, 0

    mean_speed = statistics.fmean(speeds)
    std_speed = statistics.pstdev(speeds) if len(speeds) > 1 else 0.0
    sigma_cutoff = mean_speed + 3.0 * std_speed

    cleaned = []
    removed_count = 0
    for index, (lat, lon, speed) in enumerate(points):
        # Hard cap for impossible points.
        if speed > 20.0:
            removed_count += 1
            continue

        # Anything above 15 kn is highly suspect, and more so at the route edges.
        if speed > 15.0:
            removed_count += 1
            continue

        # General suspicion threshold for very fast but still possible points.
        if speed > 12.0:
            if index <= 2 or index >= len(points) - 3:
                removed_count += 1
                continue
            if speed > sigma_cutoff:
                removed_count += 1
                continue

        cleaned.append((lat, lon, speed))

    return cleaned, removed_count


def _segment_deviation(point, seg_start, seg_end):
    """Returns (positional_deviation_m, speed_deviation_kn) of `point` from the
    straight line seg_start -> seg_end. Speed deviation compares the point's actual
    speed against the speed you'd expect from linearly interpolating between the
    segment's endpoints at the point's projected position along that line."""
    lat0 = radians((seg_start[0] + seg_end[0]) / 2)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * cos(lat0)

    def to_xy(p):
        return (p[1] * m_per_deg_lon, p[0] * m_per_deg_lat)

    ax, ay = to_xy(seg_start)
    bx, by = to_xy(seg_end)
    px, py = to_xy(point)

    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        pos_dist = sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = 0.0
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x, proj_y = ax + t * dx, ay + t * dy
        pos_dist = sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    expected_speed = seg_start[2] + t * (seg_end[2] - seg_start[2])
    speed_dev = abs(point[2] - expected_speed)

    return pos_dist, speed_dev


def simplify_track(points, tolerance_m, speed_tolerance_kn=0.0):
    """Ramer-Douglas-Peucker simplification of (lat, lon, speed) points. A point is
    kept if it deviates from the straight line between the current segment's
    endpoints by more than `tolerance_m` positionally, OR by more than
    `speed_tolerance_kn` in speed relative to the interpolated expectation. The OR
    keeps points that mark a genuine slowdown/stop or a gust even when the track
    itself barely bends there."""
    if len(points) < 3 or (tolerance_m <= 0 and speed_tolerance_kn <= 0):
        return points

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        seg_start, seg_end = points[start], points[end]

        # Track the worst offender by how far it exceeds its own tolerance (as a
        # ratio), so whichever axis - position or speed - is more "wrong" wins.
        worst_ratio, worst_index = -1.0, -1
        for i in range(start + 1, end):
            pos_dist, speed_dev = _segment_deviation(points[i], seg_start, seg_end)
            pos_ratio = (pos_dist / tolerance_m) if tolerance_m > 0 else 0.0
            speed_ratio = (speed_dev / speed_tolerance_kn) if speed_tolerance_kn > 0 else 0.0
            ratio = max(pos_ratio, speed_ratio)
            if ratio > worst_ratio:
                worst_ratio, worst_index = ratio, i

        if worst_ratio > 1.0:
            keep[worst_index] = True
            stack.append((start, worst_index))
            stack.append((worst_index, end))

    return [p for p, k in zip(points, keep) if k]


LOCAL_TZ = ZoneInfo("America/New_York")
# DJI SRT captions appear to be stamped ~3 hours early relative to the sailing route clock.
# The remaining drift is small enough to be handled as a tolerance window rather than a
# hard-coded extra minute offset, so 2026-07-29 15:28 EDT and 2026-07-30 15:27 EDT match
# the drone footage without pushing it out a few minutes on the wrong side of the route.
DRONE_TIME_OFFSET = timedelta(hours=3)
DRONE_MATCH_TOLERANCE = timedelta(minutes=15)


def normalize_datetime(dt):
    """Return a naive UTC datetime for reliable cross-source comparisons."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def format_edt_timestamp(iso_string):
    """Convert an ISO8601 UTC timestamp to Eastern Daylight Time for display."""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S EDT")
    except Exception:
        return iso_string


def format_activity_name(base_name, route_times, has_drone):
    """Convert a route to an EDT-local, human-readable label and prefix D for drone overlap."""
    if not route_times:
        return base_name

    start_dt = min(route_times)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    local_start = start_dt.astimezone(LOCAL_TZ)
    label = local_start.strftime("%Y-%m-%d %H:%M") + " EDT"
    if has_drone:
        return f"D {label}"
    return label


def parse_srt_file(file_path):
    """Parse a DJI SRT caption file into [(timestamp, lat, lon), ...] points."""
    print(f"Processing drone track {file_path}")
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        points = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                ts = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=LOCAL_TZ)
            except ValueError:
                i += 1
                continue

            lat = None
            lon = None
            for j in range(i + 1, min(i + 8, len(lines))):
                next_line = lines[j]
                lat_match = re.search(r"\[latitude:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]", next_line)
                lon_match = re.search(r"\[longitude:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]", next_line)
                if lat_match and lon_match:
                    lat = float(lat_match.group(1))
                    lon = float(lon_match.group(1))
                    break
            if lat is not None and lon is not None:
                adjusted_ts = ts + DRONE_TIME_OFFSET
                points.append((normalize_datetime(adjusted_ts), lat, lon))
            i += 1

        return points
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return []


def parse_tcx_file(file_path):
    """Parse a TCX file into a list of (lat, lon, speed) points and route timestamps."""
    print(f"Processing {file_path}")
    try:
        tree = ET.parse(file_path)
        ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
        points = []
        timestamps = []
        last_time, last_lat, last_lon = None, None, None

        for tp in tree.findall(".//ns:Trackpoint", ns):
            lat = tp.findtext("ns:Position/ns:LatitudeDegrees", namespaces=ns)
            lon = tp.findtext("ns:Position/ns:LongitudeDegrees", namespaces=ns)
            time = tp.findtext("ns:Time", namespaces=ns)

            if lat and lon and time:
                lat, lon = float(lat), float(lon)
                t = normalize_datetime(datetime.fromisoformat(time.replace("Z", "+00:00")))
                timestamps.append(t)

                speed = 0.0
                if last_time is not None:
                    dt = (t - last_time).total_seconds()
                    if dt > 0:
                        dist = haversine(last_lat, last_lon, lat, lon)  # meters
                        speed = (dist / dt) * 1.94384  # m/s -> knots

                points.append((lat, lon, speed))
                last_lat, last_lon, last_time = lat, lon, t

        cleaned_points, removed_count = clean_speed_points(points)
        if removed_count:
            valid_speeds = [s for _, _, s in points if s > 0]
            mean_speed = statistics.fmean(valid_speeds)
            std_speed = statistics.pstdev(valid_speeds) if len(valid_speeds) > 1 else 0.0
            print(f"    cleaned {removed_count} outlier speed points (hard cap 20 kn, suspect >12 kn, sigma cutoff {mean_speed + 3.0 * std_speed:.1f} kn)")

        max_speed = max((s for _, _, s in cleaned_points), default=0.0)

        simplified_points = simplify_track(cleaned_points, SIMPLIFY_TOLERANCE_M, SIMPLIFY_SPEED_TOLERANCE_KN)
        if len(simplified_points) < len(cleaned_points):
            pct = 100 * (1 - len(simplified_points) / len(cleaned_points)) if cleaned_points else 0
            print(f"    simplified {len(cleaned_points)} -> {len(simplified_points)} points ({pct:.0f}% reduction, "
                  f"tolerance {SIMPLIFY_TOLERANCE_M}m / {SIMPLIFY_SPEED_TOLERANCE_KN}kn)")

        return simplified_points, removed_count, max_speed, timestamps
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return [], 0, 0.0, []


def _round_point(point):
    lat, lon, speed = point
    return [round(lat, COORD_PRECISION), round(lon, COORD_PRECISION), round(speed, SPEED_PRECISION)]


runs = []  # List of (year, points, route_name, removed_points, start_time, end_time) tuples
all_points = []  # (lat, lon, speed_in_knots) - used only for map bounds
all_years = set()
observed_max_speed = 0.0
all_drone_points = []  # (datetime, lat, lon) points from all SRT files

# Collect drone SRT tracks once so we can filter for overlap with boat timings.
if SRT_BASE_DIR.exists():
    for srt_file in sorted(SRT_BASE_DIR.glob("*.SRT"), key=lambda p: p.name):
        all_drone_points.extend(parse_srt_file(srt_file))

# Collect TCX files from year subdirectories
if TCX_BASE_DIR.exists():
    for year_dir in sorted(TCX_BASE_DIR.glob("*/"), key=lambda x: x.name):
        year = year_dir.name
        try:
            # Validate year is numeric
            int(year)
            all_years.add(year)
            for file in sorted(year_dir.glob("*.tcx")):
                points, removed_count, max_speed, route_timestamps = parse_tcx_file(file)
                observed_max_speed = max(observed_max_speed, max_speed)
                if points:
                    route_name = file.stem
                    start_time = min(route_timestamps) if route_timestamps else None
                    end_time = max(route_timestamps) if route_timestamps else None
                    runs.append((year, points, route_name, removed_count, start_time, end_time))
                    all_points.extend(points)
        except ValueError:
            # Skip non-numeric directory names
            continue

all_years = sorted(all_years)

if AUTO_MAX_DISPLAY_SPEED and observed_max_speed > 0:
    # Round up to a clean-looking number so the legend label isn't an odd decimal.
    MAX_DISPLAY_SPEED = float(ceil(observed_max_speed))
    print(f"Auto-scaled color range to {MAX_DISPLAY_SPEED:.1f} kn (fastest recorded speed: {observed_max_speed:.1f} kn)")

# Initialize map
m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)

# Add a real satellite basemap and a street map option
# Note: In Folium, the last TileLayer added is the default visible one
folium.TileLayer(
    tiles="CartoDB positron",
    name="Street",
    control=True,
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satellite",
    control=True,
    prefer_canvas=True,
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# NOTE: tracks are no longer drawn as individual Folium PolyLine objects (and
# direction-arrow markers have been dropped entirely - they added noise without
# being reliably correct). Folium serialized each segment as its own JS statement
# block (unique variable, option dict, tooltip binding), which is what made the
# old output huge for tracks with many points. Instead, the point data is
# embedded once as compact JSON below, and a small injected script draws the
# colored track lines onto a single shared Leaflet canvas layer at load time.

if all_points:
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
     background: linear-gradient(to right, green, yellow, orange, red);
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

# Inject drawing + animation script with year/route filtering
if runs:
    # Organize route data by year - this single JSON payload is now the *only* copy
    # of point data in the file (previously it was duplicated: once inside the
    # per-segment Folium objects, and again here for the animation).
    route_data_by_year = {}
    for year, run, route_name, removed_count, start_time, end_time in runs:
        if year not in route_data_by_year:
            route_data_by_year[year] = []
        route_start_utc = normalize_datetime(start_time) if start_time else None
        route_end_utc = normalize_datetime(end_time) if end_time else None
        route_data_by_year[year].append({
            "name": route_name,
            "points": [_round_point(p) for p in run],
            "removedPoints": removed_count,
            "startTime": route_start_utc.isoformat() if route_start_utc else None,
            "endTime": route_end_utc.isoformat() if route_end_utc else None,
            "dronePoints": [],
        })

    # Only show drone points when their timestamps overlap the currently selected
    # boat route. This keeps the blue overlay aligned to actual footage time windows,
    # not just a visual line drawn across the entire session.
    for year, routes in route_data_by_year.items():
        for route in routes:
            route_start = datetime.fromisoformat(route["startTime"]) if route["startTime"] else None
            route_end = datetime.fromisoformat(route["endTime"]) if route["endTime"] else None
            if route_start is None or route_end is None:
                continue
            drone_points = []
            for drone_ts, drone_lat, drone_lon in all_drone_points:
                if route_start - DRONE_MATCH_TOLERANCE <= drone_ts <= route_end + DRONE_MATCH_TOLERANCE:
                    drone_points.append([round(drone_lat, COORD_PRECISION), round(drone_lon, COORD_PRECISION)])
            if drone_points:
                route["dronePoints"] = drone_points
                route["name"] = format_activity_name(route["name"], [route_start.replace(tzinfo=timezone.utc), route_end.replace(tzinfo=timezone.utc)], True)
            else:
                route["name"] = format_activity_name(route["name"], [route_start.replace(tzinfo=timezone.utc), route_end.replace(tzinfo=timezone.utc)], False)

    html_path = OUTPUT_FILE
    html = html_path.read_text(encoding="utf-8")
    map_name = m.get_name()

    # separators=(',', ':') strips the default whitespace json.dumps would otherwise add
    route_data_json = json.dumps(route_data_by_year, separators=(",", ":"))
    all_years_json = json.dumps(all_years, separators=(",", ":"))

    animation_script = f"""
<script>
window.addEventListener('load', function() {{
    const map = {map_name};
    const routeDataByYear = {route_data_json};
    const allYears = {all_years_json};
    const MAX_DISPLAY_SPEED = {MAX_DISPLAY_SPEED};
    if (!map || !window.L || !routeDataByYear || !allYears.length) {{
        return;
    }}

    // ---- speed -> color, matching the legend gradient (green -> yellow -> orange -> red) ----
    const COLOR_STOPS = [
        [0.00, [0, 255, 0]],
        [0.33, [255, 255, 0]],
        [0.67, [255, 165, 0]],
        [1.00, [255, 0, 0]]
    ];
    function speedColor(avgSpeedKn) {{
        const t = Math.min(Math.max(avgSpeedKn, 0) / MAX_DISPLAY_SPEED, 1);
        for (let i = 0; i < COLOR_STOPS.length - 1; i++) {{
            const [t0, c0] = COLOR_STOPS[i];
            const [t1, c1] = COLOR_STOPS[i + 1];
            if (t >= t0 && t <= t1) {{
                const f = (t - t0) / (t1 - t0 || 1);
                const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
                const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
                const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
                return 'rgb(' + r + ',' + g + ',' + b + ')';
            }}
        }}
        return 'rgb(255,0,0)';
    }}

    function haversineMeters(lat1, lon1, lat2, lon2) {{
        const R = 6371000;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }}

    function formatEDT(isoValue) {{
        const date = new Date(isoValue);
        if (Number.isNaN(date.getTime())) {{
            return isoValue;
        }}
        const parts = new Intl.DateTimeFormat('en-US', {{
            timeZone: 'America/New_York',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZoneName: 'short'
        }}).formatToParts(date);
        const map = {{}};
        parts.forEach(part => {{
            if (part.type !== 'literal') {{
                map[part.type] = part.value;
            }}
        }});
        const tz = map.timeZoneName || 'EDT';
        return map.year + '-' + map.month + '-' + map.day + ' ' + map.hour + ':' + map.minute + ':' + map.second + ' ' + tz;
    }}

    function solarDirection(lat, lon, isoValue) {{
        const date = new Date(isoValue);
        if (Number.isNaN(date.getTime())) {{
            return {{label: 'Sun: unavailable', azimuth: 0, altitude: 0}};
        }}

        const rad = Math.PI / 180;
        const latRad = lat * rad;
        const lonRad = lon * rad;

        const localDate = new Date(date.toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
        const localHours = localDate.getHours() + localDate.getMinutes() / 60 + localDate.getSeconds() / 3600;
        const dayOfYear = Math.floor((Date.UTC(localDate.getFullYear(), localDate.getMonth(), localDate.getDate()) - Date.UTC(localDate.getFullYear(), 0, 0)) / 86400000);
        const B = 2 * Math.PI * (dayOfYear - 81) / 365;
        const declination = 23.44 * Math.sin(B) * rad;
        const eqTime = 9.87 * Math.sin(2 * B) - 7.53 * Math.cos(B) - 1.5 * Math.sin(B / 2);
        const solarNoon = 12 + (lon / 15) - eqTime / 60;
        const hourAngle = (localHours - solarNoon) * 15 * rad;
        const altitude = Math.asin(
            Math.sin(latRad) * Math.sin(declination) +
            Math.cos(latRad) * Math.cos(declination) * Math.cos(hourAngle)
        ) / rad;
        const azimuth = Math.atan2(
            Math.sin(hourAngle),
            Math.cos(hourAngle) * Math.sin(latRad) - Math.tan(declination) * Math.cos(latRad)
        );
        const normAz = (azimuth * 180 / Math.PI + 180 + 360) % 360;
        const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const direction = directions[Math.round(normAz / 45) % 8];
        const altitudeClamped = Math.max(-90, Math.min(90, altitude));
        const localTimeText = formatEDT(isoValue);
        return {{
            label: direction + ' • ' + Math.round(altitudeClamped) + '° alt • ' + localTimeText,
            azimuth: normAz,
            altitude: altitudeClamped,
            localTimeText
        }};
    }}

    // ---- Draw all tracks once, onto a single shared canvas renderer ----
    const sharedRenderer = L.canvas({{ padding: 0.5 }});
    const yearLayerGroups = {{}};
    const routeLayerGroups = {{}};

    function drawRoutes() {{
        allYears.forEach(year => {{
            const yearGroup = L.layerGroup();
            routeLayerGroups[year] = {{}};
            (routeDataByYear[year] || []).forEach(route => {{
                const routeGroup = L.layerGroup();
                const pts = route.points;
                for (let i = 0; i < pts.length - 1; i++) {{
                    const [lat1, lon1, s1] = pts[i];
                    const [lat2, lon2, s2] = pts[i + 1];
                    const avgSpeed = (s1 + s2) / 2;
                    const color = speedColor(avgSpeed);

                    L.polyline([[lat1, lon1], [lat2, lon2]], {{
                        renderer: sharedRenderer, color, weight: 4, opacity: 1
                    }}).bindTooltip(route.activityName + ' • ' + route.name + ' • Year ' + year + ': ' + avgSpeed.toFixed(1) + ' kn')
                      .addTo(routeGroup);
                }}

                if (route.dronePoints && route.dronePoints.length > 1) {{
                    L.polyline(route.dronePoints, {{
                        renderer: sharedRenderer,
                        color: '#1f77ff',
                        weight: 3,
                        opacity: 0.9,
                        dashArray: '8 6'
                    }}).bindTooltip(route.name + ' • Drone overlap')
                      .addTo(routeGroup);
                }}
                routeGroup.addTo(yearGroup);
                routeLayerGroups[year][route.name] = routeGroup;
            }});
            yearGroup.addTo(map);
            yearLayerGroups[year] = yearGroup;
        }});
    }}

    drawRoutes();

    const marker = L.circleMarker([0, 0], {{
        radius: 8,
        color: '#ff3b30',
        fillColor: '#ff3b30',
        fillOpacity: 0.95,
        weight: 2
    }}).addTo(map);

    const droneMarker = L.circleMarker([0, 0], {{
        radius: 7,
        color: '#1f77ff',
        fillColor: '#1f77ff',
        fillOpacity: 0.95,
        weight: 2
    }}).addTo(map);
    droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});

    let runIndex = 0;
    let pointIndex = 0;
    let timer = null;
    let paused = false;
    let selectedYear = allYears.includes('2026') ? '2026' : (allYears.length > 0 ? allYears[0] : null);
    let selectedRouteName = null;
    let currentRuns = [];

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
    status.style.maxWidth = '360px';
    status.style.lineHeight = '1.4';
    overlay.appendChild(status);

    const statsCard = document.createElement('div');
    statsCard.style.background = 'rgba(255,255,255,0.95)';
    statsCard.style.padding = '8px 10px';
    statsCard.style.borderRadius = '8px';
    statsCard.style.boxShadow = '0 2px 8px rgba(0,0,0,0.25)';
    statsCard.style.fontSize = '11px';
    statsCard.style.maxWidth = '360px';
    statsCard.style.lineHeight = '1.5';
    statsCard.style.border = '1px solid #d0d7de';
    overlay.appendChild(statsCard);

    const routePanel = document.createElement('div');
    routePanel.style.background = 'rgba(255,255,255,0.95)';
    routePanel.style.borderRadius = '8px';
    routePanel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    routePanel.style.padding = '8px';
    routePanel.style.maxWidth = '320px';
    routePanel.style.maxHeight = '50vh';
    routePanel.style.overflowY = 'auto';
    routePanel.style.fontSize = '11px';
    routePanel.style.border = '1px solid #d0d7de';
    overlay.appendChild(routePanel);

    const routePanelHeader = document.createElement('div');
    routePanelHeader.style.display = 'flex';
    routePanelHeader.style.alignItems = 'center';
    routePanelHeader.style.justifyContent = 'space-between';
    routePanelHeader.style.gap = '8px';
    routePanelHeader.style.marginBottom = '8px';
    routePanel.appendChild(routePanelHeader);

    const routePanelTitle = document.createElement('strong');
    routePanelTitle.textContent = 'Available routes';
    routePanelHeader.appendChild(routePanelTitle);

    const routePanelToggle = document.createElement('button');
    routePanelToggle.type = 'button';
    routePanelToggle.textContent = 'Collapse';
    routePanelToggle.style.border = '1px solid #d0d7de';
    routePanelToggle.style.borderRadius = '5px';
    routePanelToggle.style.background = '#ffffff';
    routePanelToggle.style.cursor = 'pointer';
    routePanelToggle.style.padding = '2px 6px';
    routePanelHeader.appendChild(routePanelToggle);

    const routePanelContent = document.createElement('div');
    routePanel.appendChild(routePanelContent);

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

    const routeButtonsByYear = {{}};

    function updateStatusLabel() {{
        const total = currentRuns.length;
        const routeLabel = selectedRouteName ? ' • ' + selectedRouteName : '';
        const currentRun = currentRuns[runIndex] || [];
        const point = currentRun[pointIndex] || currentRun[0] || [0, 0];
        const sun = solarDirection(point[0], point[1], currentRun.isoTimestamp || currentRun.activityTimestamp || new Date().toISOString());
        const label = 'Year: ' + selectedYear + routeLabel + ' • ' + sun.localTimeText + ' • Animating run ' + (total > 0 ? (runIndex + 1) : 0) + ' / ' + total + ' • point ' + (currentRuns[runIndex] ? (pointIndex + 1) : 0) + ' • Sun: ' + sun.label;
        status.textContent = label;
    }}

    function getSelectedRoutes() {{
        if (!selectedYear) {{
            return [];
        }}
        const routes = routeDataByYear[selectedYear] || [];
        if (!selectedRouteName) {{
            return routes;
        }}
        const chosen = routes.find(route => route.name === selectedRouteName);
        return chosen ? [chosen] : [];
    }}

    function computeSelectionStats(routes) {{
        let totalDistanceMeters = 0;
        let totalSpeedKnots = 0;
        let validPoints = 0;
        let maxSpeedKnots = 0;
        let totalSegments = 0;
        let filteredOut = 0;

        routes.forEach(route => {{
            const points = route.points || [];
            if (!points.length) {{
                return;
            }}
            for (let i = 1; i < points.length; i++) {{
                const prev = points[i - 1];
                const curr = points[i];
                totalDistanceMeters += haversineMeters(prev[0], prev[1], curr[0], curr[1]);
                totalSegments += 1;
            }}
            points.forEach(point => {{
                const speed = Number(point[2]) || 0;
                totalSpeedKnots += speed;
                validPoints += 1;
                maxSpeedKnots = Math.max(maxSpeedKnots, speed);
            }});

            filteredOut += route.removedPoints || 0;
        }});

        return {{
            routeCount: routes.length,
            averageSpeedKnots: validPoints ? totalSpeedKnots / validPoints : 0,
            maxSpeedKnots: maxSpeedKnots,
            averageDistanceMeters: totalSegments ? totalDistanceMeters / totalSegments : 0,
            totalDistanceMeters: totalDistanceMeters,
            totalDistanceKm: totalDistanceMeters / 1000,
            totalDistanceNm: totalDistanceMeters / 1852,
            filteredOut: filteredOut,
        }};
    }}

    function updateStatsDisplay() {{
        const selectedRoutes = getSelectedRoutes();
        const stats = computeSelectionStats(selectedRoutes);

        if (!selectedRoutes.length) {{
            statsCard.innerHTML = '<strong>No route selected</strong><br>Choose a year or one of the routes to inspect statistics.';
            return;
        }}

        const avgSpeedKnots = stats.averageSpeedKnots.toFixed(1);
        const maxSpeedKnots = stats.maxSpeedKnots.toFixed(1);
        const avgSpeedKph = (stats.averageSpeedKnots * 1.852).toFixed(1);
        const maxSpeedKph = (stats.maxSpeedKnots * 1.852).toFixed(1);
        const avgDist = (stats.averageDistanceMeters / 1000).toFixed(1);
        const totalDistKm = stats.totalDistanceKm.toFixed(1);
        const totalDistNm = stats.totalDistanceNm.toFixed(1);

        statsCard.innerHTML = [
            '<strong>Selected run stats</strong><br>',
            'Runs: ' + stats.routeCount + '<br>',
            'Avg speed: ' + avgSpeedKnots + ' kn (' + avgSpeedKph + ' km/h)<br>',
            'Max speed: ' + maxSpeedKnots + ' kn (' + maxSpeedKph + ' km/h)<br>',
            'Avg distance: ' + avgDist + ' km / leg<br>',
            'Total distance: ' + totalDistKm + ' km (' + totalDistNm + ' nm)<br>',
            'Filtered out: ' + stats.filteredOut + ' points'
        ].join('');
    }}

    function setCurrentRunsForSelection() {{
        if (!selectedYear) {{
            currentRuns = [];
            selectedRouteName = null;
            return;
        }}

        const routes = routeDataByYear[selectedYear] || [];
        if (!selectedRouteName) {{
            currentRuns = routes.map(route => route.points);
        }} else {{
            const chosen = routes.find(route => route.name === selectedRouteName);
            currentRuns = chosen ? [chosen.points] : [];
        }}
    }}

    function updateDroneMarkerForSelection() {{
        if (!selectedYear || !selectedRouteName) {{
            droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}

        const routes = routeDataByYear[selectedYear] || [];
        const chosen = routes.find(route => route.name === selectedRouteName);
        const dronePoints = chosen && chosen.dronePoints ? chosen.dronePoints : [];

        if (!dronePoints.length) {{
            droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}

        const point = dronePoints[0];
        droneMarker.setLatLng([point[0], point[1]]);
        droneMarker.setStyle({{ opacity: 1, fillOpacity: 0.95 }});
    }}

    function updateYearButtons() {{
        const buttons = yearButtonsContainer.querySelectorAll('button');
        buttons.forEach(btn => {{
            if (btn.dataset.year === selectedYear) {{
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

    function updateRouteButtons() {{
        Object.keys(routeButtonsByYear).forEach(year => {{
            const group = routeButtonsByYear[year];
            if (!group) return;
            Object.keys(group).forEach(routeName => {{
                if (routeName === 'allButton') return;
                const btn = group[routeName];
                const active = year === selectedYear && selectedRouteName === routeName;
                btn.style.background = active ? '#007AFF' : '#ffffff';
                btn.style.color = active ? '#ffffff' : '#000000';
                btn.style.fontWeight = active ? 'bold' : 'normal';
                btn.style.borderColor = active ? '#007AFF' : '#d0d7de';
            }});

            const allButton = group.allButton;
            if (allButton) {{
                const active = year === selectedYear && selectedRouteName === null;
                allButton.style.background = active ? '#dfeeff' : '#ffffff';
                allButton.style.color = '#000000';
                allButton.style.fontWeight = active ? 'bold' : 'normal';
                allButton.style.borderColor = active ? '#7aa7ff' : '#d0d7de';
            }}
        }});
    }}

    // Replaces the old "scan map._layers for matching option metadata" approach:
    // we already hold direct references to each year/route's layer group from drawRoutes().
    function updateLayerOpacity() {{
        const baseOpacity = 0.18;
        const accentOpacity = 1.0;

        allYears.forEach(year => {{
            const activeYear = year === selectedYear;
            const routes = routeLayerGroups[year] || {{}};
            Object.keys(routes).forEach(routeName => {{
                const isSelectedRoute = activeYear && selectedRouteName && routeName === selectedRouteName;
                const isAllSelectedYear = activeYear && !selectedRouteName;
                const opacity = isSelectedRoute || isAllSelectedYear ? accentOpacity : baseOpacity;

                routes[routeName].eachLayer(layer => {{
                    if (layer.setStyle) {{
                        layer.setStyle({{ opacity, fillOpacity: opacity }});
                    }}
                }});
            }});
        }});
    }}

    function applyRouteSelection(year, routeName) {{
        selectedYear = year;
        selectedRouteName = routeName;
        setCurrentRunsForSelection();
        runIndex = 0;
        pointIndex = 0;
        updateYearButtons();
        updateRouteButtons();
        updateLayerOpacity();
        updateDroneMarkerForSelection();
        updateStatusLabel();
        updateStatsDisplay();
        if (currentRuns.length === 0) {{
            marker.setLatLng([0, 0]);
            paused = true;
            pausePlayButton.textContent = 'Play';
        }} else {{
            paused = false;
            pausePlayButton.textContent = 'Pause';
            advance();
        }}
    }}

    allYears.forEach(year => {{
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.year = year;
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
            applyRouteSelection(year, null);
        }});
        yearButtonsContainer.appendChild(btn);
    }});

    let routePanelCollapsed = false;
    function setRoutePanelCollapsed(collapsed) {{
        routePanelCollapsed = collapsed;
        routePanelToggle.textContent = collapsed ? 'Expand' : 'Collapse';
        routePanelContent.style.display = collapsed ? 'none' : 'block';
    }}

    routePanelToggle.addEventListener('click', function() {{
        setRoutePanelCollapsed(!routePanelCollapsed);
    }});

    allYears.forEach(year => {{
        const section = document.createElement('div');
        section.style.marginBottom = '6px';

        const summary = document.createElement('div');
        summary.style.display = 'flex';
        summary.style.alignItems = 'center';
        summary.style.justifyContent = 'space-between';
        summary.style.gap = '6px';
        summary.style.marginBottom = '6px';
        summary.style.cursor = 'pointer';

        const title = document.createElement('strong');
        title.textContent = year + ' (' + (routeDataByYear[year] || []).length + ')';
        summary.appendChild(title);

        const allBtn = document.createElement('button');
        allBtn.type = 'button';
        allBtn.textContent = 'All';
        allBtn.style.border = '1px solid #d0d7de';
        allBtn.style.borderRadius = '5px';
        allBtn.style.background = '#ffffff';
        allBtn.style.cursor = 'pointer';
        allBtn.style.padding = '2px 6px';
        allBtn.addEventListener('click', function(event) {{
            event.stopPropagation();
            applyRouteSelection(year, null);
        }});
        summary.appendChild(allBtn);

        const list = document.createElement('div');
        list.style.display = 'flex';
        list.style.flexDirection = 'column';
        list.style.gap = '4px';
        list.style.paddingLeft = '8px';
        list.style.marginBottom = '8px';

        routeButtonsByYear[year] = {{ allButton: allBtn }};

        (routeDataByYear[year] || []).forEach(route => {{
            const routeButton = document.createElement('button');
            routeButton.type = 'button';
            routeButton.textContent = route.name;
            routeButton.style.border = '1px solid #d0d7de';
            routeButton.style.borderRadius = '5px';
            routeButton.style.background = '#ffffff';
            routeButton.style.padding = '4px 6px';
            routeButton.style.cursor = 'pointer';
            routeButton.style.textAlign = 'left';
            routeButton.addEventListener('click', function() {{
                applyRouteSelection(year, route.name);
            }});
            list.appendChild(routeButton);
            routeButtonsByYear[year][route.name] = routeButton;
        }});

        const yearToggle = document.createElement('span');
        yearToggle.textContent = '▾';
        yearToggle.style.fontSize = '12px';
        yearToggle.style.marginLeft = '4px';
        summary.appendChild(yearToggle);

        let yearExpanded = true;
        summary.addEventListener('click', function() {{
            yearExpanded = !yearExpanded;
            list.style.display = yearExpanded ? 'flex' : 'none';
            yearToggle.textContent = yearExpanded ? '▾' : '▸';
            if (selectedYear !== year && yearExpanded) {{
                applyRouteSelection(year, null);
            }}
        }});

        section.appendChild(summary);
        section.appendChild(list);
        routePanelContent.appendChild(section);
    }});

    const prevButton = document.createElement('button');
    prevButton.type = 'button';
    prevButton.textContent = '⏮';
    prevButton.style.background = '#ffffff';
    prevButton.style.border = '1px solid #d0d7de';
    prevButton.style.borderRadius = '999px';
    prevButton.style.padding = '6px 10px';
    prevButton.style.cursor = 'pointer';
    prevButton.style.fontSize = '12px';
    prevButton.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    overlay.appendChild(prevButton);

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

    const nextButton = document.createElement('button');
    nextButton.type = 'button';
    nextButton.textContent = '⏭';
    nextButton.style.background = '#ffffff';
    nextButton.style.border = '1px solid #d0d7de';
    nextButton.style.borderRadius = '999px';
    nextButton.style.padding = '6px 10px';
    nextButton.style.cursor = 'pointer';
    nextButton.style.fontSize = '12px';
    nextButton.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    overlay.appendChild(nextButton);

    function updateAnimationAtCurrentIndex() {{
        if (!currentRuns || currentRuns.length === 0) {{
            return;
        }}

        while (runIndex < 0) {{
            runIndex = currentRuns.length - 1;
        }}
        while (runIndex >= currentRuns.length) {{
            runIndex = 0;
        }}

        const currentRun = currentRuns[runIndex];
        if (!currentRun || !currentRun.length) {{
            return;
        }}

        while (pointIndex < 0) {{
            runIndex = (runIndex - 1 + currentRuns.length) % currentRuns.length;
            pointIndex += currentRuns[runIndex].length;
        }}

        while (pointIndex >= currentRun.length) {{
            pointIndex -= currentRun.length;
            runIndex = (runIndex + 1) % currentRuns.length;
            const nextRun = currentRuns[runIndex];
            if (!nextRun || !nextRun.length) {{
                pointIndex = 0;
                break;
            }}
        }}

        const nextRun = currentRuns[runIndex];
        if (!nextRun || !nextRun.length) {{
            return;
        }}

        const point = nextRun[pointIndex];
        if (!point) {{
            return;
        }}

        marker.setLatLng([point[0], point[1]]);

        if (selectedYear && selectedRouteName) {{
            const routes = routeDataByYear[selectedYear] || [];
            const chosen = routes.find(route => route.name === selectedRouteName);
            const dronePoints = chosen && chosen.dronePoints ? chosen.dronePoints : [];
            if (dronePoints.length && nextRun.length > 1) {{
                const progress = pointIndex / (nextRun.length - 1);
                const droneIndex = Math.round(progress * (dronePoints.length - 1));
                const dronePoint = dronePoints[droneIndex];
                droneMarker.setLatLng([dronePoint[0], dronePoint[1]]);
                droneMarker.setStyle({{ opacity: 1, fillOpacity: 0.95 }});
            }} else {{
                droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            }}
        }} else {{
            droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
        }}

        updateStatusLabel();
    }}

    function moveByStep(delta) {{
        if (!currentRuns || currentRuns.length === 0) {{
            return;
        }}

        pointIndex += delta;
        while (pointIndex < 0) {{
            runIndex = (runIndex - 1 + currentRuns.length) % currentRuns.length;
            pointIndex += currentRuns[runIndex].length;
        }}

        while (pointIndex >= (currentRuns[runIndex] || []).length) {{
            pointIndex -= (currentRuns[runIndex] || []).length;
            runIndex = (runIndex + 1) % currentRuns.length;
        }}

        updateAnimationAtCurrentIndex();
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
            droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
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

        if (selectedYear && selectedRouteName) {{
            const routes = routeDataByYear[selectedYear] || [];
            const chosen = routes.find(route => route.name === selectedRouteName);
            const dronePoints = chosen && chosen.dronePoints ? chosen.dronePoints : [];
            if (dronePoints.length && currentRun.length > 1) {{
                const progress = pointIndex / (currentRun.length - 1);
                const droneIndex = Math.round(progress * (dronePoints.length - 1));
                const dronePoint = dronePoints[droneIndex];
                droneMarker.setLatLng([dronePoint[0], dronePoint[1]]);
                droneMarker.setStyle({{ opacity: 1, fillOpacity: 0.95 }});
            }} else {{
                droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            }}
        }} else {{
            droneMarker.setStyle({{ opacity: 0, fillOpacity: 0 }});
        }}

        updateStatusLabel();

        pointIndex += 1;
        if (pointIndex >= currentRun.length) {{
            pointIndex = 0;
            runIndex = (runIndex + 1) % currentRuns.length;
        }}

        scheduleNext();
    }}

    prevButton.addEventListener('click', function() {{
        paused = true;
        pausePlayButton.textContent = 'Play';
        moveByStep(-1);
    }});

    nextButton.addEventListener('click', function() {{
        paused = true;
        pausePlayButton.textContent = 'Play';
        moveByStep(1);
    }});

    pausePlayButton.addEventListener('click', function() {{
        paused = !paused;
        pausePlayButton.textContent = paused ? 'Play' : 'Pause';
        if (!paused) {{
            scheduleNext();
        }}
    }});

    setRoutePanelCollapsed(false);
    updateRouteButtons();
    updateLayerOpacity();
    updateDroneMarkerForSelection();
    setCurrentRunsForSelection();
    updateStatusLabel();
    updateStatsDisplay();

    if (currentRuns.length > 0) {{
        advance();
    }}
}});
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", animation_script + "\n</body>")
        html_path.write_text(html, encoding="utf-8")

print(f"Speed-colored tracks with animation saved to {OUTPUT_FILE}")