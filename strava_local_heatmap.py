# Here is the code to create a dynamic map from strava output.
# This code runs slowly to generate the HTML file, but it parses through both GPX files and FIT files.
# See the README for more details on how to set up and run this program!

# If you only have gpx files, there is commented out code below that only deals with gpx files and runs quickly!

# Enjoy!

import os
import glob
import numpy as np
import folium
from branca.element import Element
import gzip
import pandas as pd

import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point

from fitparse import FitFile
from xml.etree import ElementTree as ET
from datetime import datetime
from argparse import ArgumentParser, Namespace

from geopy import distance
from geopy.distance import geodesic


# Simplification tolerance in meters. 3 m keeps the visible route shape
# very close to the original while removing large numbers of redundant GPS points.
ROUTE_SIMPLIFY_METERS = 10.0

def simplify_track(lat_lon_data: np.ndarray, tolerance_meters: float = ROUTE_SIMPLIFY_METERS) -> np.ndarray:
    """Reduce GPS points while preserving the shape of the route.

    Shapely's Douglas-Peucker simplification is applied in a local
    latitude/longitude coordinate system. The longitude tolerance is
    adjusted for latitude so the requested tolerance is approximately
    the same in meters in both directions. Endpoints and significant
    bends are retained.
    """
    if lat_lon_data is None or len(lat_lon_data) <= 2:
        return lat_lon_data

    # Remove immediately repeated points first.
    keep = np.ones(len(lat_lon_data), dtype=bool)
    if len(lat_lon_data) > 1:
        keep[1:] = np.any(np.diff(lat_lon_data, axis=0) != 0, axis=1)
    data = lat_lon_data[keep]

    if len(data) <= 2:
        return data

    mean_lat = float(np.mean(data[:, 0]))
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = max(111_320.0 * np.cos(np.radians(mean_lat)), 1.0)

    # Work in approximately-meter coordinates, simplify, then convert back.
    coords_xy = [(float(lon) * meters_per_degree_lon,
                  float(lat) * meters_per_degree_lat)
                 for lat, lon in data]

    simplified = LineString(coords_xy).simplify(
        tolerance_meters,
        preserve_topology=False
    )

    if simplified.is_empty:
        return data[[0, -1]]

    coords = np.asarray(simplified.coords, dtype=float)
    result = np.column_stack((
        coords[:, 1] / meters_per_degree_lat,
        coords[:, 0] / meters_per_degree_lon
    ))

    # Always retain the exact original endpoints.
    result[0] = data[0]
    result[-1] = data[-1]
    return result
START_COORDS= (43.7, -72.3)     # map center (e.g., New Hampshire)
ZOOM=8  #starting map zoom

def extract_gpx_info(gpx_file: str) -> tuple:
    # extract activity name and formatted date from GPX file
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    namespace = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    name = root.find('.//gpx:name', namespace).text
    time = root.find('.//gpx:metadata/gpx:time', namespace).text
    aType = root.find('.//gpx:type', namespace).text
    date_obj = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
    formatted_date = date_obj.strftime('%B %d, %Y %I:%M %p')
    return name, formatted_date, aType

def extract_fit_info(fit_file: str, activities_df: pd.DataFrame) -> tuple:
    # extract activity name, date, and activity ID from FIT file
    try:
        fit = FitFile(fit_file)
        # Look for the 'timestamp' in the FIT file
        timestamps = [msg.get_value('timestamp') for msg in fit.get_messages('record') if msg.get_value('timestamp')]
        if not timestamps:
            return "fail", "fail", "fail"
        start_time = min(timestamps)  # take earliest timestamp
        formatted_time = start_time.strftime('%B %d, %Y %I:%M %p')
        # match to CSV to get name and ID
        match = activities_df[activities_df["Activity Date"] == formatted_time]
        if match.empty:
            return "fail", "fail", "fail"
        activity_name = match.iloc[0]["Activity Name"]
        activity_id = str(match.iloc[0]["Activity ID"])
        activity_type = match.iloc[0]["Activity Type"]
        
        return activity_name, formatted_time, activity_id, activity_type
    except Exception as e:
        print(f"FIT read error {fit_file}: {e}")
        return "fail", "fail", "fail"


#Using no sampling to reduce file size
def read_fit_trackpoints(fit_file: str) -> np.ndarray:
    """Return lat/lon points from a FIT file"""
    try:
        fit = FitFile(fit_file)
        lat_lon_data = []
        for record in fit.get_messages('record'):
            lat = record.get_value('position_lat')
            lon = record.get_value('position_long')
            if lat is not None and lon is not None:
                # convert semicircles to degrees (from google)
                lat_deg = lat * (180 / 2**31)
                lon_deg = lon * (180 / 2**31)
                lat_lon_data.append([lat_deg, lon_deg])
        return np.array(lat_lon_data)
    except:
        return np.array([])

def main(args: Namespace) -> None:

    # load activities CSV
    linecolor="red"
    lineweight=1
    activities_df = pd.read_csv('activities.csv')
    activities_df["Activity Date"] = pd.to_datetime(
        activities_df["Activity Date"],
        format="%b %d, %Y, %I:%M:%S %p",
        errors="coerce"
    )
    # convert back to formatted string for matching
    activities_df["Activity Date"] = activities_df["Activity Date"].dt.strftime('%B %d, %Y %I:%M %p')

    # read GPX files
    gpx_files = glob.glob(f'{args.gpx_dir}/*.gpx')
    fit_files = glob.glob(f'{args.fit_dir}/*.fit.gz') # diff folder for fit fyi

    if not gpx_files and not fit_files:
        exit("ERROR: No GPX or FIT files found!")

    # custom mapbox map
    MAPBOX_TOKEN = "pk.eyJ1IjoibGJyZWdvdSIsImEiOiJjbWZyNXFnOWwwM2diMmlvcXB6M3M4bHdzIn0.ainJCdTcN4gJLONN7TBZHg"

    m = folium.Map(
        location=START_COORDS, 
        zoom_start=ZOOM,
        #tiles=f"https://api.mapbox.com/styles/v1/lbregou/cmfpl06zz00ht01qkfq3h2dp1/tiles/256/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        tiles=f"https://api.mapbox.com/styles/v1/lbregou/cmfpl0bnp00h801qhfsbn3mrw/tiles/256/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        attr='Mapbox - Bregou Custom Map'
    )

    # Custom tooltip CSS
    custom_css = """
    <style>
        .leaflet-tooltip.custom-tooltip-style {
            background-color: #007bff;
            color: yellow;
            border: 2px solid green;
            font-size: 16px;
            padding: 8px;
            border-radius: 3px;
        }
        .leaflet-interactive:focus {
            outline: none !important;
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(custom_css))

    # GPX routes are simplified with Douglas-Peucker rather than sampled at a fixed interval.
    # =========================================================
    # ADVENTURE MAP STATE
    # =========================================================

    # Python-side data used while building the HTML.
    ride_years = set()

    # JavaScript registry: every ride registers its invisible
    # touch target and its visible line here.
    m.get_root().html.add_child(Element("""
    <script>
    window.adventureRides = [];
    </script>
    """))

    # =========================================================
    # GPX ACTIVITIES
    # =========================================================

    for gpx_file in gpx_files:
        print(f"Reading GPX {os.path.basename(gpx_file)}")

        activity_name, activity_date, activity_type = extract_gpx_info(gpx_file)

        lat_lon_data = []

        with open(gpx_file, encoding="utf-8") as file:
            for line in file:
                if "<trkpt" in line:
                    l = line.split('"')
                    lat_lon_data.append([float(l[1]), float(l[3])])

        lat_lon_data = np.array(lat_lon_data)

        if lat_lon_data.size == 0:
            continue

        lat_lon_data = simplify_track(lat_lon_data)  # shape-preserving simplification

        # ---------------------------------------------------------
        # COLOR
        # ---------------------------------------------------------

        linecolor = "red"
        lineweight = 1

        match activity_type:
            case "AlpineSki" | "NordicSki" | "IceSkate" | "BackcountrySki" | "Snowboard" | "Snowshoe":
                linecolor = "fuchsia"
                lineweight = 1.5

        if ("fat" in activity_name.casefold()) and ("father" not in activity_name.casefold()):
            linecolor = "fuchsia"
            lineweight = 1.5

        # ---------------------------------------------------------
        # YEAR / HEATMAP DATA
        # ---------------------------------------------------------

        try:
            activity_year = datetime.strptime(
                str(activity_date),
                "%B %d, %Y %I:%M %p"
            ).year
            ride_years.add(activity_year)
        except (ValueError, TypeError):
            activity_year = None
# ---------------------------------------------------------
        # GEOJSON
        # ---------------------------------------------------------

        geojson_feature = {
            "type": "Feature",
            "properties": {
                "name": activity_name,
                "date": activity_date,
                "year": activity_year,
                "color": linecolor,
                "weight": lineweight
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [lon, lat] for lat, lon in lat_lon_data.tolist()
                ]
            }
        }

        # ---------------------------------------------------------
        # LARGE INVISIBLE TOUCH / HOVER TARGET
        # ---------------------------------------------------------

        hit_area = folium.GeoJson(
            geojson_feature,
            style_function=lambda x: {
                "color": x["properties"]["color"],
                "weight": 30,
                "opacity": 0.0,
            },
            highlight_function=lambda x: {
                "color": x["properties"]["color"],
                "weight": 30,
                "opacity": 0.0,
            },
            tooltip=folium.Tooltip(
                f"<strong>{activity_name}</strong><br>"
                f"{activity_date}<br>"
                f"Tap again for more info",
                sticky=True
            )
        ).add_to(m)

        # ---------------------------------------------------------
        # ACTUAL VISIBLE RIDE LINE
        # ---------------------------------------------------------

        polyline = folium.GeoJson(
            geojson_feature,
            style_function=lambda x: {
                "color": x["properties"]["color"],
                "weight": x["properties"]["weight"],
                "opacity": 1.0,
            },
            highlight_function=lambda x: {
                "color": "yellow",
                "weight": x["properties"]["weight"] + 2,
                "opacity": 1.0,
            }
        ).add_to(m)

        # ---------------------------------------------------------
        # STRAVA POPUP
        # ---------------------------------------------------------

        activity_id = os.path.splitext(os.path.basename(gpx_file))[0]
        strava_url = f"https://www.strava.com/activities/{activity_id}"

        popup_content = f"""
        <strong>Activity:</strong> {activity_name}<br>
        <strong>Date:</strong> {activity_date}<br>
        <b><a href="{strava_url}" target="_blank">View on Strava</a></b>
        """

        folium.Popup(
            popup_content,
            max_width=300
        ).add_to(hit_area)

        # ---------------------------------------------------------
        # REGISTER RIDE FOR DESKTOP + MOBILE INTERACTION
        # ---------------------------------------------------------

        hit_name = hit_area.get_name()
        line_name = polyline.get_name()

        m.get_root().html.add_child(Element(f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var hitLayer = {hit_name};
            var visibleLine = {line_name};

            var ride = {{
                hitLayer: hitLayer,
                visibleLine: visibleLine,
                originalColor: "{linecolor}",
                originalWeight: {lineweight},
                year: {activity_year if activity_year is not None else "null"},
                lastLatLng: null
            }};

            window.adventureRides.push(ride);

            hitLayer.eachLayer(function(layer) {{
                layer.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);

                    ride.lastLatLng = e.latlng;

                    if (window.adventureManager) {{
                        window.adventureManager.toggle(ride);
                    }}
                }});
            }});
        }});
        </script>
        """))

    # =========================================================
    # FIT ACTIVITIES
    # =========================================================

    for fit_file in fit_files:
        print(f"Reading FIT {os.path.basename(fit_file)}")

        try:
            # Decompress if needed
            if fit_file.endswith(".gz"):
                with gzip.open(fit_file, "rb") as f:
                    fit_bytes = f.read()

                fit_path = f"temp_{os.path.basename(fit_file)}"

                with open(fit_path, "wb") as temp_file:
                    temp_file.write(fit_bytes)
            else:
                fit_path = fit_file

            activity_name, activity_date, activity_id, activity_type = extract_fit_info(
                fit_path,
                activities_df
            )

            if activity_name == "fail":
                print(f"Failed to match {fit_file}")
                continue

            lat_lon_data = read_fit_trackpoints(fit_path)

            if lat_lon_data.size == 0:
                print(f"No trackpoints in {fit_file}")
                continue

            lat_lon_data = simplify_track(lat_lon_data)  # shape-preserving simplification

            # ---------------------------------------------------------
            # COLOR
            # ---------------------------------------------------------

            linecolor = "red"
            lineweight = 1

            match activity_type:
                case "Alpine Ski" | "Nordic Ski" | "Ice Skate" | "Backcountry Ski" | "Snowboard" | "Snowshoe":
                    linecolor = "fuchsia"
                    lineweight = 1.5

            if ("fat" in activity_name.casefold()) and ("father" not in activity_name.casefold()):
                linecolor = "fuchsia"
                lineweight = 1.5

            # ---------------------------------------------------------
            # YEAR / HEATMAP DATA
            # ---------------------------------------------------------

            try:
                activity_year = datetime.strptime(
                str(activity_date),
                "%B %d, %Y %I:%M %p"
            ).year
                ride_years.add(activity_year)
            except (ValueError, TypeError):
                activity_year = None
# ---------------------------------------------------------
            # GEOJSON
            # ---------------------------------------------------------

            geojson_feature = {
                "type": "Feature",
                "properties": {
                    "name": activity_name,
                    "date": activity_date,
                    "year": activity_year,
                    "color": linecolor,
                    "weight": lineweight
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [lon, lat] for lat, lon in lat_lon_data.tolist()
                    ]
                }
            }

            # ---------------------------------------------------------
            # LARGE INVISIBLE TOUCH / HOVER TARGET
            # ---------------------------------------------------------

            hit_area = folium.GeoJson(
                geojson_feature,
                style_function=lambda x: {
                    "color": x["properties"]["color"],
                    "weight": 30,
                    "opacity": 0.0,
                },
                highlight_function=lambda x: {
                    "color": x["properties"]["color"],
                    "weight": 30,
                    "opacity": 0.0,
                },
                tooltip=folium.Tooltip(
                    f"<strong>{activity_name}</strong><br>"
                    f"{activity_date}<br>"
                    f"Tap again for more info",
                    sticky=True
                )
            ).add_to(m)

            # ---------------------------------------------------------
            # ACTUAL VISIBLE RIDE LINE
            # ---------------------------------------------------------

            polyline = folium.GeoJson(
                geojson_feature,
                style_function=lambda x: {
                    "color": x["properties"]["color"],
                    "weight": x["properties"]["weight"],
                    "opacity": 1.0,
                },
                highlight_function=lambda x: {
                    "color": "yellow",
                    "weight": x["properties"]["weight"] + 2,
                    "opacity": 1.0,
                }
            ).add_to(m)

            # ---------------------------------------------------------
            # STRAVA POPUP
            # ---------------------------------------------------------

            strava_url = f"https://www.strava.com/activities/{activity_id}"

            popup_content = f"""
            <strong>Activity:</strong> {activity_name}<br>
            <strong>Date:</strong> {activity_date}<br>
            <b><a href="{strava_url}" target="_blank">View on Strava</a></b>
            """

            folium.Popup(
                popup_content,
                max_width=300
            ).add_to(hit_area)

            # ---------------------------------------------------------
            # REGISTER RIDE FOR DESKTOP + MOBILE INTERACTION
            # ---------------------------------------------------------

            hit_name = hit_area.get_name()
            line_name = polyline.get_name()

            m.get_root().html.add_child(Element(f"""
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                var hitLayer = {hit_name};
                var visibleLine = {line_name};

                var ride = {{
                    hitLayer: hitLayer,
                    visibleLine: visibleLine,
                    originalColor: "{linecolor}",
                    originalWeight: {lineweight},
                    year: {activity_year if activity_year is not None else "null"},
                    lastLatLng: null
                }};

                window.adventureRides.push(ride);

                hitLayer.eachLayer(function(layer) {{
                    layer.on('click', function(e) {{
                        L.DomEvent.stopPropagation(e);

                        ride.lastLatLng = e.latlng;

                        if (window.adventureManager) {{
                            window.adventureManager.toggle(ride);
                        }}
                    }});
                }});
            }});
            </script>
            """))

        except Exception as e:
            print(f"Error reading FIT {fit_file}: {e}")


    # =========================================================
    # NO YEAR SLIDER / REPLAY CONTROLS
    # =========================================================

    # Year slider and Replay functionality have been removed.
    # Routes remain on the map with their existing click/tap interaction.

    else:
        # No year controls are needed.
        pass


    m.save(outfile=args.output)
    print(f"Saved interactive map to {args.output}")

# CLI

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate an interactive map from GPX and FIT files")
    parser.add_argument("--test", type=bool, default=False, help="Test mode when true")
    parser.add_argument("--gpx_dir", default="gpx", help="Directory containing GPX files")
    parser.add_argument("--fit_dir", default="fit", help="Directory containing FIT files")
    parser.add_argument("--output", default="index.html", help="Output HTML map file")
    args = parser.parse_args()
    #args.test=True #force test mode if don't want to use command line switch --test TRUE
    if args.test==True:
        args.gpx_dir="gpxtest"
        args.fit_dir="fittest"
    main(args)

