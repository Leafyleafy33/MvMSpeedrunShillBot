from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import random

class MissionClusterer:
    """
    A class for clustering mission names based on Word2Vec embeddings and KMeans clustering.

    This class clusters mission names into groups and assigns random emojis to each cluster.
    
    Attributes:
        mission_names (list): A list of mission names to be clustered.
        n_clusters (int, optional): The number of clusters to create (default is 30).
    """
    def __init__(self, mission_names, n_clusters=30):
        self.mission_names = mission_names
        self.n_clusters = n_clusters
        self.cluster_names_emojis = {
    0: ['Timekeepers', '⏳', '🕰️', '🌟', '🚀', '🌈', '☣️', '🧚', '🏝️', '🎲', '🌌'],           
    1: ['Toxic Wastelands', '💀', '🔍', '🌊', '🌫️', '❓', '🌅', '🌒', '💧', '🎲', '🌠'],       
    2: ['Enchanted Enigmas', '🔮', '🌅', '🌌', '🌠', '🌪️', '🔍', '📜', '🌟', '🍄', '⚙️'],         
    3: ['Coastal Treasures', '🏖️', '🍀', '🌀', '🕯️', '🏔️', '🔭', '🎶', '🌲', '⚙️', '🧨'],       
    4: ['Lucky Foragers', '🍀', '🌀', '🌊', '🌅', '⚡', '🔮', '🏝️', '🌫️', '💧', '🎲'],       
    5: ['Twilight Voyages', '🌌', '⚡', '🌅', '🏝️', '🔮', '📜', '🌪️', '🌊', '🎲', '🧨'],       
    6: ['Whirling Winds', '🌪️', '🌬️', '🔮', '🍀', '🎲', '🏖️', '🌠', '⚙️', '🧨', '🌒'],           
    7: ['Eclipse Enigmas', '🌑', '🌒', '🏔️', '🔍', '🌪️', '🌌', '🕯️', '💀', '💧', '⚙️'],        
    8: ['Lakeside Serenity', '💧', '🏝️', '🌌', '🌀', '🌈', '🔮', '🌅', '🌲', '📜', '💀'],      
    9: ['Cosmic Insights', '🔭', '🌌', '💀', '🌅', '🌪️', '🌊', '🌀', '🍄', '🏖️', '🧚'],   
    10: ['Volcanic Vengeance', '🌋', '💥', '🔥', '🌅', '🚀', '🪐', '🌪️', '💀', '🔍', '🌌'],           
    11: ['Mystical Midnight', '🌌', '🌟', '🚀', '🌠', '🌈', '🌍', '🌒', '💧', '🏝️', '🔥'],       
    12: ['Jungle Journeys', '🌿', '🌲', '🌊', '🌌', '🍄', '🏔️', '🔭', '🌅', '🌪️', '🔮'],         
    13: ['Secret Societies', '🕯️', '🌌', '💧', '🌅', '🌪️', '🔍', '📜', '🌟', '🍄', '⚙️'],       
    14: ['Mechanical Marvels', '🦾', '🔍', '📜', '🌌', '🌲', '🌳', '🌿', '🏝️', '🌋', '🌌'],       
    15: ['Floral Whispers', '🌸', '🌺', '🍃', '🌿', '🦉', '🐦', '🌸', '🌺', '🍂', '🌰'],       
    16: ['Whirlwind Adventures', '🌪️', '🌊', '🍃', '🌿', '🦉', '🐦', '🌸', '🌺', '🍂', '🌰'],           
    17: ['Electric Sparks', '⚡', '🔌', '💥', '⚙️', '🔋', '🪓', '🌩', '🧨', '🚡', '🌪️'],        
    18: ['Gears of Time', '⏳', '⏰', '🔒', '🌅', '📡', '🧭', '🔬', '🌡️', '📚', '📜'],      
    19: ['Foggy Cliffhangers', '🌫️', '🏞️', '🌫️', '🌁', '🏔️', '❄️', '🌨️', '⛅', '🌦️', '🌄'],  
    20: ['Solitary Path', '🚶‍♂️', '🚶', '🌄', '🌅', '🚶‍♀️', '🏞️', '🌲', '🌳', '🌲', '🌳'],     
    21: ['Starry Expeditions', '✨', '🚀', '🌠', '🌟', '🌍', '🌔', '🌓', '🌒', '🌑', '🌌'],   
    22: ['Inferno Forge', '🔥', '🛠️', '🔥', '🌋', '🌌', '🔥', '🪓', '⚔️', '🚀', '🌌'],        
    23: ['Snowy Peaks', '🏔️', '❄️', '🌨️', '🏂', '🏔️', '🏔️', '❄️', '🌨️', '🏂', '🌬️'],         
    24: ['Wasteland Echoes', '☣️', '🔥', '💣', '🔥', '🌋', '🌌', '💀', '🚀', '🌌'],    
    25: ['Bioluminescent Wonders', '🌟', '🌌', '🌿', '🪐', '🌲', '🌳', '🌸', '🌼', '🌻'],  
    26: ['Clockwork Marvels', '⚙️', '🔩', '🕰️', '📡', '🧭', '🔬', '🌡️', '📚', '📜'],  
    27: ['Eclipse Mysteries', '🌘', '🌒', '🌘', '🌚', '🌌', '🌓', '🌑', '🌘', '🌒', '🌌'],     
    28: ['Arcane Contraptions', '🔮', '🛠️', '📦', '⚗️', '🧪', '📚', '🔍', '🌪️', '💀'],  
    29: ['Candlelit Secrets', '🕯️', '🌌', '🔍', '📦', '🔮', '📜', '📖', '🗝️', '🕰️', '📦'],   
}

        self.model = None
        self.clustered_missions = None

    def tokenize_missions(self):
        return [word_tokenize(name.lower()) for name in self.mission_names]

    def train_word2vec(self, tokenized_missions):
        self.model = Word2Vec(
            tokenized_missions,
            vector_size=200,    # Increased vector size
            window=20,          # Fixed or dynamic window size
            min_count=1,
            sg=1,               # Skip-gram model
            hs=1,               # Using hierarchical softmax
            negative=5,         # Negative sampling
            alpha=0.03,         # Initial learning rate
            min_alpha=0.0007,   # Minimum learning rate
            sample=1e-3,        # Higher sample value for more sparseness
            epochs=20           # Increased training epochs
        )

    def get_mission_vector(self, name):
        vector = np.zeros(self.model.vector_size)
        count = 0
        for word in word_tokenize(name.lower()):
            if word in self.model.wv:
                vector += self.model.wv[word]
                count += 1
        return vector / count if count > 0 else vector

    def cluster_missions(self):
        tokenized_missions = self.tokenize_missions()
        self.train_word2vec(tokenized_missions)
        
        # Dimensionality Reduction
        mission_vectors = np.array([self.get_mission_vector(name) for name in self.mission_names])
        pca = PCA(n_components=0.95)  # Retain 95% variance
        reduced_vectors = pca.fit_transform(mission_vectors)

        # Clustering with optimized KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(reduced_vectors)
        self.clustered_missions = {}
        for cluster_id, mission_name in zip(clusters, self.mission_names):
            self.clustered_missions.setdefault(cluster_id, []).append(mission_name)

    def get_random_cluster_emoji(self, mission_name):
        mission_name = mission_name.lower()
        for cluster_id, missions in self.clustered_missions.items():
            if mission_name in [m.lower() for m in missions]:
                cluster_info = self.cluster_names_emojis.get(cluster_id + 1, ("Unknown", ["❓"]))
                emojis = cluster_info[1:]
                return random.choice(emojis) 
        return "❓"