from typing import Optional
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_travel_duration(start_location: str, destination_location: str, transportation_mode: Optional[str] = None) -> str:
    """Gets the travel time between two places using the Google Maps Routes API.

    Args:
        start_location: the place from which you start your ride
        destination_location: the place of arrival
        transportation_mode: The transportation mode, in 'driving', 'walking', 'bicycling', or 'transit'. Defaults to 'driving'.
    """
    import os
    import googlemaps
    from datetime import datetime

    gmaps = googlemaps.Client(key=os.getenv("GMAPS_API_KEY"))

    if transportation_mode is None:
        transportation_mode = "driving"
    try:
        # Using the Routes API instead of the legacy Directions API
        routes_result = gmaps.directions(
            origin=start_location,
            destination=destination_location,
            mode=transportation_mode,
            departure_time=datetime(2025, 6, 6, 11, 0),  # At 11, date far in the future
        )
        if not routes_result:
            return "No route found between these places with the required transportation mode."
        print(routes_result)
        return routes_result[0]["legs"][0]["duration"]["text"]
    except Exception as e:
        print(e)
        return str(e)

agent = CodeAgent(tools=[get_travel_duration], model=HfApiModel(), additional_authorized_imports=["datetime"])

agent.run("Can you give me a nice one-day trip around Paris with a few locations and the times? Could be in the city or outside, but should fit in one day. I'm travelling only with a rented bicycle.")