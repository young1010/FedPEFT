"""
preprocess with PLM
"""

import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import re

ml1m_path = '/data/ml-1m/ml-1m/'
device = "cuda" if torch.cuda.is_available() else "cpu"
processed_path = '/data/ml-1m/processed/t5/'
model = SentenceTransformer('/data/llm/sentence-t5-base', device=device)


def pre_process_rating():
    users_inter = {} # {0:[0],1:[]}
    items_inter = {}    # {0:[1],1:[]}
    rating_data = [] # [{'ratingID':1,'asin':0,..},{},{},..]
    rating_data_fl = {} # rating_data split by user {1:[{'ratingID':1,'asin':0,..}]}

    ratings = pd.read_csv(ml1m_path + 'ratings.dat', sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    rating_data = ratings.values # [[1 101 5],[2 102 4],..] array

    grouped = ratings.groupby('UserID')
    rating_data_fl = {user_id -1: group.values -1 for user_id, group in grouped} # rating_data split by user {1:[[1 101 5],[]], 2:}
    users_inter = {user_id -1: group.values[:,1] -1 for user_id, group in grouped} # {1:[101 103], 2:}

    grouped_m = ratings.groupby('MovieID')
    items_inter = {movie_id-1: group.values[:,0]-1 for movie_id, group in grouped_m} # {101:[1 3], 102:}

    torch.save(users_inter, processed_path + 'graph_user.pth')
    torch.save(items_inter, processed_path + 'graph_item.pth')
    torch.save(rating_data, processed_path + 'ratings.pth')
    torch.save(rating_data_fl, processed_path + 'ratings_fl.pth')

    print("finish")

# movies
def pre_process_movies():
    def process_genres(genres):
        split_genres = genres.replace('|', ',')
        return ", belongs to the " + split_genres + " genres."
    
    def process_title(title):
        split_strings = re.split(r'\s*\(([^)]*)\)\s*$', title)
        title = split_strings[0].strip()
        year = split_strings[1].strip(')') if len(split_strings) > 1 else 'unknown'
        return "A Movie \'" + title + "\', realeased in " + year

    movies = pd.read_csv(ml1m_path + 'movies.dat', sep='::', encoding='latin1', header=None, engine='python', names=['MovieID', 'Title', 'Genres'])
    movies_data = [""] * 3952
    for i, movie in movies.iterrows():
        movies_data[int(movie['MovieID']) - 1] = process_title(str(movie['Title'])) + process_genres(str(movie['Genres']))

    embedding = model.encode(movies_data, convert_to_tensor=True)
    print(embedding.shape) # 3952*768
    torch.save(embedding, processed_path + 'items.pth')
    

def pre_process_users():
    gender = ['F', 'M']
    gender_feature = ['female', 'male']

    age = [1, 18, 25, 35, 45, 50, 56]
    age_feature = ['under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

    occupation = ["other or not specified", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service",
                  "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", 
                  "sales/marketing","scientist", "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]

    users = pd.read_csv(ml1m_path + 'users.dat', sep='::', header=None, engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    users_data = [""] * 6040  
    for i, user in users.iterrows():
        users_data[int(user['UserID'])-1] = "A " + gender_feature[gender.index(str(user['Gender']))]+ " user" + ", aged " + age_feature[age.index(int(user['Age']))] + ", with occupation of " + occupation[int(user['Occupation'])] + "."
                                           
    embedding = model.encode(users_data, convert_to_tensor=True)
    print(embedding.shape) # 6040*768
    torch.save(embedding, processed_path + 'users.pth')

pre_process_users()
pre_process_movies()