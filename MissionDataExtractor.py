from datetime import timedelta, datetime
from functools import lru_cache
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio

# Helper Functions
def translate_victory_type(rank):
    return ("World Record", "🏆") if rank == 1 else ("Personal Best", "🥇")

def get_emoji(rank):
    emojis = {1: "🏆", 2: "🥈", 3: "🥉", 4: "🎖️", 5: "🎉"}
    return emojis.get(rank, "🎉")

class MissionDataExtractor:
    """
    A utility class for extracting and processing mission and player data from Potato.tf's API.

    This class provides methods to fetch mission and player data, format it, and post-process the resulting dataframes.
    """
    def __init__(self):
        self.difficulty_mapping_inverted = {"Int ": "Intermediate", "Adv ": "Advanced", "Exp ": "Expert", "Rev ": "Reversed", "Reverse ": "Reversed"}

    @lru_cache(maxsize=None)
    def adjust_map_name(self, map_name):
        return map_name.replace('mvm_', '').title()

    @lru_cache(maxsize=None)
    def replace_player_name(self, player_name):
        return player_name.replace('่', 'Googlayz')

    @lru_cache(maxsize=None)
    def format_time(self, seconds):
        return str(timedelta(seconds=int(seconds)))

    @lru_cache(maxsize=None)
    def format_date(self, timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y')  # Corrected this line

    @lru_cache(maxsize=None)
    def get_nice_mission_name(self, mission):
        return mission.replace("_", " ").title()

    def extract_difficulty(self, mission_name):
        for key, value in self.difficulty_mapping_inverted.items():
            if key in mission_name:
                return mission_name.replace(key, '').strip(), value
        return mission_name, None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def fetch_data(self, session, url):
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    def format_speedrun_record(self, record):
        players_info = record.get("players", [])

        player_details = [{
            "SteamID": player.get("steamid", ""),
            "PersonaName": self.replace_player_name(player.get("personaname", "")),
            "ProfileURL": player.get("profileurl", ""),
            "AvatarURL": player.get("avatarmedium", ""),
            "Map": self.adjust_map_name(record.get("map", "")),
            "Mission": self.get_nice_mission_name(record.get("mission", "")),
        } for player in players_info]

        return {
            "Time": self.format_time(record.get("time", 0)),
            "Date": self.format_date(record.get("timeAdded", 0)),
            "Players": [player['PersonaName'] for player in player_details],
            "PlayerDetails": player_details
        }

    def process_data(self, speedrun_data, mission_data, map_name):
        mission_records = []
        player_records = []
        adjusted_map_name = self.adjust_map_name(map_name)

        for record in speedrun_data:
            formatted_record = self.format_speedrun_record(record)
            mission_name, difficulty = self.extract_difficulty(self.get_nice_mission_name(record["mission"]))
            mission_record = {"Map": adjusted_map_name, "Mission": mission_name, "Difficulty": difficulty, **formatted_record}
            mission_record['Total Players'] = len(formatted_record['Players'])  # Calculate the total players for each mission
            mission_records.append(mission_record)
            player_records.extend(formatted_record["PlayerDetails"])

        missions_df = pd.DataFrame(mission_records)
        missions_df['Rank'] = missions_df.groupby('Mission')['Time'].rank(method='dense').astype(int)
        missions_df['World Record'] = missions_df['Rank'] == 1

        for index, row in missions_df[missions_df['World Record']].iterrows():
            for player in row['PlayerDetails']:
                player['WorldRecord'] = True

        players_df = pd.DataFrame(player_records)
        return missions_df, players_df


    async def fetch_map_data(self, session, map_info_item):
        map_name = map_info_item['name']
        speedrun_url = f"https://potato.tf/api/speedrun?map={map_name}"
        mission_info_url = f"https://potato.tf/api/missioninfo?map={map_name}"
        speedrun_data, mission_data = await asyncio.gather(
            self.fetch_data(session, speedrun_url),
            self.fetch_data(session, mission_info_url)
        )
        if speedrun_data and mission_data:
            return self.process_data(speedrun_data, mission_data, map_name)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            map_info_url = "https://potato.tf/api/mapinfo"
            map_info = await self.fetch_data(session, map_info_url)
            if not map_info:
                print("No map names were found.")
                return None, None

            tasks = [self.fetch_map_data(session, item) for item in map_info]
            all_mission_frames = []
            all_player_frames = []
            for task in tqdm_asyncio.as_completed(tasks, desc="Fetching map data", total=len(map_info)):
                missions_df, players_df = await task
                if missions_df is not None:
                    all_mission_frames.append(missions_df)
                if players_df is not None:
                    all_player_frames.append(players_df)

            combined_missions_df = pd.concat(all_mission_frames, ignore_index=True) if all_mission_frames else pd.DataFrame()
            combined_players_df = pd.concat(all_player_frames, ignore_index=True) if all_player_frames else pd.DataFrame()

            combined_missions_df, combined_players_df = self.post_process_dataframe(combined_missions_df, combined_players_df)
            return combined_missions_df, combined_players_df

    def post_process_dataframe(self, missions_df, players_df):
        if not missions_df.empty:
            missions_df['Map'] = missions_df['Map'].str.replace('_', ' ')
            missions_df['Mission'] = missions_df['Mission'].str.replace('_', ' ')
            missions_df['Date'] = pd.to_datetime(missions_df['Date'], format='%d/%m/%Y').dt.strftime("%m/%d/%Y")
            missions_df.sort_values(['Mission', 'Time'], inplace=True)
    
        if not players_df.empty:
            players_df['WorldRecordsHeld'] = players_df.groupby('SteamID')['WorldRecord'].transform('sum')
            players_df = players_df.sort_values('WorldRecordsHeld', ascending=False).drop_duplicates('SteamID')
            players_df = players_df[['SteamID', 'PersonaName', 'ProfileURL', 'AvatarURL', 'WorldRecordsHeld']].reset_index(drop=True)
    
        return missions_df, players_df