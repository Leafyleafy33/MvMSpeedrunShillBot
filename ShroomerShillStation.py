import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag, SoupStrainer
import os
from functools import lru_cache
import asyncio
from MissionClusterer import MissionClusterer
from MissionDataExtractor import MissionDataExtractor
from MissionTitleUpdater import MissionTitleUpdater
from datetime import datetime
import streamlit.components.v1 as components
from bs4 import SoupStrainer

import warnings
warnings.filterwarnings('ignore', message='missing ScriptRunContext')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')
warnings.filterwarnings('ignore', module='concurrent.futures')
warnings.filterwarnings('ignore', message='.*ThreadPoolExecutor.*')
warnings.filterwarnings('ignore', module='streamlit')

class DataFetcher:
    @staticmethod
    @lru_cache(maxsize=128)
    def cached_requests_get(*args, **kwargs):
        return requests.get(*args, **kwargs)

    @staticmethod
    @st.cache_resource
    def fetch_description(url):
        try:
            response = DataFetcher.cached_requests_get(url)
            if response.ok:  # Checks for all 2XX status codes
                strainer = SoupStrainer('div', class_='profile_summary')
                soup = BeautifulSoup(response.content, 'lxml', parse_only=strainer)
                description_tag = soup.find()
                return UtilityFunctions.parse_description(description_tag) if description_tag else '<p>No description found</p>'
            else:
                return f'<p>Error: {response.status_code}</p>'
        except requests.RequestException as e:
            st.error(f'Request error in fetch_description: {e}')
            return ''
        except Exception as e:
            st.error(f'Unexpected error in fetch_description: {e}')
            return ''

    @staticmethod
    @st.cache_resource
    def fetch_background_url(url):
        try:
            response = DataFetcher.cached_requests_get(url)
            if response.ok:
                soup = BeautifulSoup(response.content, 'lxml')
                background_div = soup.find('div', class_='profile_animated_background')
                if not background_div:
                    background_div = soup.find('div', class_='no_header profile_page has_profile_background')
                if not background_div:
                    background_div = soup.find('div', class_='no_header profile_page has_profile_background full_width_background')
                if not background_div:
                    return 'https://community.cloudflare.steamstatic.com/public/images/profile/2020/bg_dots.png'

                if background_div:
                    video_tag = background_div.find('video', poster=True)
                    style = background_div.get('style', '')
                    if video_tag and 'poster' in video_tag.attrs:
                        # Ensure to return a URL without single quotes
                        return video_tag['poster'].replace("'", "")
                    elif 'url(' in style:
                        return style.split('url(')[1].split(')')[0].replace("'", "").replace("\"", "")
                return ''
            else:
                return f'<p>Error: {response.status_code}</p>'
        except requests.RequestException as e:
            st.error(f'Request error in fetch_background_url: {e}')
            return ''
        except Exception as e:
            st.error(f'Unexpected error in fetch_background_url: {e}')
            return ''

    @staticmethod
    async def async_fetch_data():
        extractor = MissionDataExtractor()
        missions_df, players_df = await extractor.run()
        mission_names = missions_df['Mission'].unique().tolist()
        
    
        clusterer = MissionClusterer(mission_names)
        clusterer.cluster_missions()
        
        title_updater = MissionTitleUpdater(clusterer)
        missions_df = title_updater.add_title_column(missions_df)
        missions_df = title_updater.add_ai_prompts_column(missions_df)
    
    
        return missions_df, players_df
    
    @staticmethod
    def fetch_data():
        return asyncio.run(DataFetcher.async_fetch_data())

    @staticmethod
    @st.cache_resource
    def get_data():
        return DataFetcher.fetch_data()

class UtilityFunctions:
    @staticmethod
    def parse_description(description_tag):
        components = []
        for element in description_tag.descendants:
            if isinstance(element, NavigableString):
                if element.strip() != '':
                    components.append(element.strip())
            elif isinstance(element, Tag):
                if element.name == 'br':
                    components.append('<br>')  # Change to HTML line break
                elif element.name == 'img':
                    # Keep the image tags as HTML and ensure they are displayed inline
                    image_url = element.get('src')
                    alt_text = element.get('alt', '')
                    components.append(f'<img src="{image_url}" alt="{alt_text}" style="display: inline; height: auto; max-width: 100%;">')
                elif element.name == 'a':
                    # Convert to HTML anchor tag
                    link_text = element.text
                    link_url = element.get('href')
                    components.append(f'<a href="{link_url}">{link_text}</a>')
        return ''.join(components)  # Return as HTML string

    @staticmethod
    def create_card_html(player, index):
        background_url = player['BackgroundURL']
        avatar_url = player['AvatarURL']
        persona_name = player['PersonaName']
        world_records_held = player['WorldRecordsHeld']
        description = player['Description']

        card_html = f"""
        <body style="background: linear-gradient(to bottom, #3e1e28 0%, #2c202b 20%, #1e222f 100%);"> 
        <div class="card-container" style="background-image: url('{background_url}');">
            <div class="card">
                <div class="flip-card-inner">
                    <div class="flip-card-front">
                        <img src="{avatar_url}" alt="{persona_name}">
                        <div class="card-info">
                            <h3>{persona_name}</h3>
                            <p>Rank: #{index + 1}</p>
                            <p>World Records: {world_records_held}</p>
                        </div>
                    </div>
                    <div class="flip-card-back">
                        <div class="card-back-content">
                            <h1><style="box-shadow:  5px 5px 19px rgba(0,0,0,0.4), -5px -5px 19px rgba(0,0,0,0.4);">{persona_name}'s Details</h1>
                            <p>{description}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        </body>
        <style>
            .card-container {{
                box-sizing: border-box;
                perspective: 1000px;
                width: calc(20% - 10px);
                margin: 10px 5px 20px;
                height: 330px;
                border: 2px solid #f0f0f0;
                border-radius: 10px;
                background-size: cover;
                background-position: center;
                display: inline-block;
                vertical-align: top;

            }}

            .card {{
                width: 100%;
                height: 100%;
                position: relative;
                transform-style: preserve-3d;
                transition: transform 0.6s;


            }}

            .card:hover .flip-card-inner {{
                transform: rotateY(180deg);
            }}

            .flip-card-front, .flip-card-back {{
                position: absolute;
                width: 100%;
                height: 100%;
                backface-visibility: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                background-color: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(2px);
            }}

            .flip-card-front {{
                background-size: cover;
                background-position: center;
            }}

            .card-info {{
                width: 100%;
                color: #fff;
                padding: 5px;
                margin-bottom: 4px;
                background-color: rgba(0, 0, 0, 0.1);
                text-align: center;
                border: 1px solid #f0f0f0;
                box-shadow:  5px 5px 19px rgba(0,0,0,0.4),
                 -5px -5px 19px rgba(0,0,0,0.4);
            }}
            .card-back-content {{
                width: 100%;
                color: #fff;
                font-size: 10px;
                overflow: auto;
                padding: 5px;
                margin-bottom: 8px;
            }}

            .flip-card-inner {{
                position: relative;
                width: 100%;
                height: 100%;
                text-align: center;
                transition: transform 0.6s;
                transform-style: preserve-3d;
            }}
            .flip-card-back {{
                background-size: cover;
                background-position: center;
                transform: rotateY(180deg);
            }}
            .flip-card-front {{
                z-index: 2; 
            }}

            .flip-card-back {{
                transform: rotateY(180deg); 
                z-index: 1;  
            }}

            .card-container:hover .flip-card-inner {{
                transform: rotateY(180deg); 
            }}

            .flip-card-front img {{
                width: auto;
                height: auto;
                max-width: 125px;
                border-radius: 25%; 
                margin-bottom: 17px;
                box-shadow:  5px 5px 19px rgba(0,0,0,0.4),
                 -5px -5px 19px rgba(0,0,0,0.4);
            }}

        </style>
        """

        return card_html

    @staticmethod
    def filter_by_player(df, player_name):
        df = df.dropna(subset=['Players'])
        df['Players'] = df['Players'].astype(str)
        return df[df['Players'].str.contains(player_name, case=False)]

    @staticmethod
    def update_top_players(players_df, selection_count):
        modified_df = players_df.copy()
        modified_df['WorldRecordsHeld'] = pd.to_numeric(players_df['WorldRecordsHeld'], errors='coerce').fillna(0).astype(int)
        modified_df['AvatarURL'] = modified_df['AvatarURL'].str.replace('_medium', '_full')
        return modified_df.nlargest(selection_count, 'WorldRecordsHeld')
    
    @staticmethod
    def search_and_display_by_date(df):
        with st.form("record_search_form"):
            st.subheader('Search and Display Records by Date')
            record_type = st.selectbox("Select the record type", ['All', 'World Record', 'Personal Best'], index=0).lower()
            date_input = st.date_input("Select a date", datetime.today())
            submit_button = st.form_submit_button(label='Search')

            if submit_button:
                search_date = pd.Timestamp(date_input)
                if record_type != 'all':
                    filtered_df = df[df['World Record'] == (record_type == "world record")].copy()
                else:
                    filtered_df = df.copy()
            
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%m/%d/%Y')
                exact_date_found = search_date in filtered_df['Date'].values

                if exact_date_found:
                    display_df = filtered_df[filtered_df['Date'] == search_date]
                else:
                    nearest_date = filtered_df.iloc[(filtered_df['Date'] - search_date).abs().argsort()[:1]]
                    display_df = filtered_df[filtered_df['Date'] == nearest_date['Date'].values[0]]
                    st.warning(f"No data found for {date_input.strftime('%m/%d/%Y')}. Showing results for the nearest date: {nearest_date['Date'].dt.strftime('%m/%d/%Y').values[0]}")
                
                for index, row in display_df.iterrows():
                    st.markdown(f'```markdown\n{row["AI_Prompt"]}\n```') 

    @staticmethod
    def display_cards(players_df, player_count_selection):
        all_cards_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center;'>"

        # Fetch descriptions and background URLs for unique ProfileURLs
        unique_urls = players_df['ProfileURL'].unique()
        description_cache = {url: DataFetcher.fetch_description(url) for url in unique_urls}
        background_cache = {url: DataFetcher.fetch_background_url(url) for url in unique_urls}

        # Modify the players_df DataFrame using vectorized operations
        players_df['Description'] = players_df['ProfileURL'].map(description_cache)
        players_df['BackgroundURL'] = players_df['ProfileURL'].map(background_cache)

        cards_html = [UtilityFunctions.create_card_html(player, index) for index, player in players_df.iterrows()]
        all_cards_html += ''.join(cards_html)
        all_cards_html += "</div>"

        # Pre-set values
        card_height = 340  # The height of a single card
        card_margin = 20   # The total vertical margin around a card
        cards_per_row = 5  # Number of cards in a single row

        rows = (player_count_selection / cards_per_row)

        # Calculate total height of the container based on rows, card height, and margins
        total_height = rows * (card_height + card_margin)

        st.components.v1.html(all_cards_html, height=total_height)

    @staticmethod
    def display_system_prompt():
        system_prompt = '''
### System Prompt for Dual Thematic Integration:
THIS IS A SYSTEM PROMPT, AWAIT THE DATA STRUCTURE ABOVE AND FOLLOW THE STEPS TO CRAFT THE ANNOUNCEMENT AFTER RECEIPT. Additonally, pay special attention to the CAPITALIZED PUNS, Humor, and the use of EMOJIS to concisely create HYPE and ENERGY.
If you understand this, it is important that you simply reply with "Recieved, I am a Masterfully Talented Social Media Manager that understands "Better to be low effort and HIGH ENERGY, than high effort and low energy": Awaiting Data..." and the data structure will be sent to you.
                    
#### Step 1: Data Reception
Await the following data structure:
- **Title:** (e.g., 🌲 TF2 MvM Speedrun | Potato.tf: Mechanical magic - Advanced | 🏆 World Record | [0:21:25])
- **Data:**
  - Game Mode
  - Map
  - Mission
  - Time
  - Date
  - Difficulty
  - World Record Status
  - Total Players
  - Dual Theme (e.g., Mushroom Hunting + Map & Mission)

#### Step 2: Theme Identification
Combine "Mushroom Hunting" with the mission theme (e.g., Mechanical Magic) for a dual theme. Take the dual theme to the extreme - imagine you are making the theme a meme, and use aliterations where possible (e.g., Mushroom Hunting + Mechanical Magic = "Magical Mecha-Mushroom Mayhem")

#### Step 3: Crafting the Announcement: Act as a Creative Director and design the announcement using the following structure: ((Inject maximum CAPITALIZED PUNS, and over-the-top humor to create EXAGGERATED HYPE and ENERGY)
- **Title Adaptation:** Use the received title exactly as recieved, only adding and modifying thematic emojis and flair.
- **Introduction:** Craft a hyper-enthusiastic intro using both themes, highlighting the game mode, map, and mission achievement.
- **Highlight:** Exaggerate Mann vs. Machine elements,, like the waves and challenges, using puns and references specific to TF2 and the dual theme. (Make good use of Puns and Humor)
- **Support Integration:** (Be very Concise and Specific here, use CAPITALS and PUNS for Humor, but Go all out with thematic party descriptors, wild merchandise descriptions, and tie the boost to the thematic event in the most flamboyant way possible.)
  - **Patreon**: Develop a thematic party descriptor, Be very Concise
  - **Merch**: Describe merchandise with dual thematic terms, Be very Concise and Specific
  - **Ko-fi**: Tie the boost to the thematic event, Be very Concise and Specific
- **Community Call-to-Action:** Encourage participation under a thematic banner, using engaging language.
- **SEO Keywords:** Include relevant keywords based on TF2, MvM, and the themes.

#### Step 4: Hype, Wit, and Energy
Ensure the language is lively, witty, and engaging, suitable for a gaming audience craving excitement and memes.

#### Example Based on "Mechanical Magic" and TF2 MvM:
- **Title:** 🌲 TF2 MvM Speedrun | Potato.tf: Mechanical Magic - Advanced | 🏆 World Record | [0:21:25] 🍄⚙️
- **Introduction:** "Dive into a MECHA-MUSHROOM mayhem in TF2 Mann vs. Machine on the Wizardry map - Mechanical Magic, where we've engineered a RECORD-BREAKING victory in just [0:21:25]! 🏆"
- **Highlight:** "Our team tackled the torrent of robotic waves with FUNGAL FERVOR and MECHANICAL MASTERY, outmaneuvering every gear and circuit in this epic battle of wit and will! 🍄⚙️ This isn't just a triumph; it's a STEAM-POWERED SPECTACLE in the world of esports!"
- **Support Integration**: Adapt Patreon, Merch, and Ko-fi sections with the dual theme, but remember brevity is the soul of wit.
- **Community Call-to-Action**: "Join our CYBER-SPORE SQUAD, LIKE, SUBSCRIBE, and SHARE to be a part of our innovative gaming and speedrunning universe, thriving under the MYCOTECH BANNER! 🙌🍄🤖"

#### Step 5: Review
Review the annoucement to ensure it is concise, and witty, maximizing the Humor of the DUAL-THEMED CAPITILIZED PUNS to ensure the post is high energy and engaging.
    '''
        st.markdown(f"```markdown\n{system_prompt}\n```")

class MainApp:
    def __init__(self):
        self.missions_df = None
        self.players_df = None
        self.load_css()
        self.main()

    def load_css(self):
        base_path = os.path.dirname(__file__)
        css_path = os.path.join(base_path, 'Assets', 'style.css')
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    def main(self):
        st.title("Shroomer Shill Station 300k")
        st.markdown("### A tool for searching and displaying records for the [Potato.tf](https://potato.tf/speedrun) MvM speedrunning community")
        st.markdown("#### Created by [Chessmaster Hex](https://github.com/Leafyleafy33) and [Mushroom hunting](https://www.youtube.com/@Mushroomhunting1337) for peak laziness")
        # Data fetching and storing in session state
        if 'data_fetched' not in st.session_state:
            missions_df, players_df = DataFetcher.get_data()
            if missions_df.empty or players_df.empty:
                st.error("Failed to fetch data. Please check data sources and network connectivity.")
                return
            st.session_state['missions_df'] = missions_df
            st.session_state['players_df'] = players_df
            st.session_state['data_fetched'] = True

        # Record Search Section
        with st.expander("Record Search"):
            selected_player_name = st.selectbox("Select a player", st.session_state['players_df']['PersonaName'])
            player_missions = UtilityFunctions.filter_by_player(st.session_state['missions_df'], selected_player_name)
            UtilityFunctions.search_and_display_by_date(player_missions)

        # button to show system prompt
        with st.expander("System Prompt"):
            UtilityFunctions.display_system_prompt()


        # Player Dashboard Section
        with st.expander("Player Dashboard"):
            player_count_options = ['10', '20', '50']
            player_count_selection = st.radio("Select the number of top players to display:", player_count_options, index=0)
            top_players_df = UtilityFunctions.update_top_players(st.session_state['players_df'], int(player_count_selection))
            
            # Call the display_cards function with the DataFrame of top players
            UtilityFunctions.display_cards(top_players_df, int(player_count_selection))

if __name__ == '__main__':
    st.set_page_config(page_title="MvM Shill Station", page_icon="🍄", layout="wide", initial_sidebar_state="expanded")
    app = MainApp()