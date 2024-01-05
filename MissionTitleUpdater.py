from MissionClusterer import MissionClusterer
from MissionDataExtractor import MissionDataExtractor, translate_victory_type, get_emoji

class MissionTitleUpdater:
    """
    A class for updating mission titles and generating AI prompts for creating gaming achievement announcements.

    This class adds a new title column to the input DataFrame with formatted mission titles and generates directive prompts
    in Markdown format for AI to create community-focused, engaging gaming achievement announcements.

    Attributes:
        clusterer (MissionClusterer): An instance of the MissionClusterer class used for clustering mission names.
    """
    def __init__(self, clusterer):
        self.clusterer = clusterer

    def generate_prompt_for_self(self, row):
        """
        Generate a directive prompt in Markdown format for AI to create a community-focused, engaging gaming achievement announcement.
        """
        record_status = "World Record" if row['World Record'] else "Personal Best"
        
        seo_keywords = ', '.join(['TF2 MvM', 'speedrun', row['Map'], row['Mission'], 'teamwork', 'gaming excellence', 'cooperative gaming', 'gaming strategies', 'gaming challenges', 'gaming tips and tricks', 'achievement unlocked', 'competitive gaming', 'esports', 'speed running', record_status])
        prompt = (
            f"**Title:** {row['Title']}\n\n"
            "**Data:**\n"
            f"- **Game Mode:** TF2 Mann vs. Machine\n"
            f"- **Map:** {row['Map']}\n"
            f"- **Mission:** {row['Mission']}\n"
            f"- **Time:** {row['Time']}\n"
            f"- **Date:** {row['Date']}\n"
            f"- **Difficulty:** {row['Difficulty']}\n"
            f"- **World Record Status:** {'World Record' if row['World Record'] else 'Personal Best'}\n"
            f"- **Players:** {', '.join(row['Players'])}\n"
            f"- **Total Players:** {row['Total Players']}\n\n"
            f"- **Dual Theme:** Mushroom Hunting + {row['Map']} & {row['Mission']}\n\n"
            f"- **SEO Keywords:** {seo_keywords}\n\n"
)
        return prompt
        
    def add_title_column(self, missions_df):
        missions_df['Title'] = missions_df.apply(
            lambda row: f"{self.clusterer.get_random_cluster_emoji(row['Mission'])} TF2 MvM Speedrun | Potato.tf: {row['Mission'].capitalize()} - {row['Difficulty']} | {get_emoji(row['Rank'])} {translate_victory_type(row['Rank'])[0]} | [{row['Time']}]",
            axis=1
        )
        return missions_df
    
    def add_ai_prompts_column(self, missions_df):
        """
        Adds a new column 'AI_Prompt' to the DataFrame containing generated AI prompts for each record.
        """
        missions_df['AI_Prompt'] = missions_df.apply(self.generate_prompt_for_self, axis=1)
        return missions_df
    