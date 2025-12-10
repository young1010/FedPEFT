"""
preprocess with PLM
"""

import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

review_path = '/data/Amazon/5-core/'
review_path_clip = '/data/Amazon/5-core_clip/'
graph_path = '/data/Amazon/graph/'

meta_path = '/data/Amazon/meta/meta_'
device = "cuda" if torch.cuda.is_available() else "cpu"
meta_path_clip = '/data/Amazon/meta_processed/t5/'
model = SentenceTransformer('/data/llm/sentence-t5-base', device=device)

categories = [
    "All_Beauty",
    "AMAZON_FASHION",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Digital_Music",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Luxury_Beauty",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Prime_Pantry",
    "Software",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Electronics",
    "Home_and_Kitchen",]

def pre_process_review(field:str):
    meta_data = {} # {0:{},}
    users = [] # ['B#auser','']
    items = [] # ['B#item','']
    users_inter = {} # {0:[0],1:[]}
    items_inter = {}    # {0:[1],1:[]}
    review_data = [] # [{'reviewID':1,'asin':0,..},{},{},..]
    review_data_fl = {} # review_data split by user {1:[{'reviewID':1,'asin':0,..}]}

    for _, review in pd.read_json(review_path + field + "_5.json", orient='records', lines=True).iterrows():
        if review['reviewerID'] not in users:
            users.append(review['reviewerID'])
            users_inter[users.index(review['reviewerID'])] = list()
            review_data_fl[users.index(review['reviewerID'])] = list()
        if review['asin'] not in items:
            items.append(review["asin"])
            items_inter[items.index(review["asin"])] = list()
            meta_data[items.index(review["asin"])] = dict()
        # graph 
        users_inter[users.index(review['reviewerID'])].append(items.index(review["asin"]))
        items_inter[items.index(review["asin"])].append(users.index(review['reviewerID']))

        # review
        review_dict = dict()
        review_dict['reviewerID'] = users.index(review['reviewerID'])
        review_dict['asin'] = items.index(review["asin"])
        # review_dict['reviewText'] = process_txt_clip(review['reviewText'])
        # review_dict['summary'] = process_txt_clip(review['summary'])
        # if 'image' in review:
        #     if isinstance(review['image'][0], str):
        #         review_dict['image'] = process_img_clip(review['image'][0])
        #     else:
        #         review_dict['image'] = process_img_clip(review['image'][0][0])
        
        # review_dict['unixReviewTime'] = review['unixReviewTime']
        review_dict['overall'] = float(review['overall'])

        review_data.append(review_dict)
        review_data_fl[users.index(review['reviewerID'])].append(review_dict)
    torch.save(users_inter, graph_path + field + '_user.pth')
    torch.save(items_inter, graph_path + field + '_item.pth')
    # torch.save(review_data, review_path_clip + field + '.pth')
    torch.save(review_data_fl, review_path_clip + field + '_fl.pth')
    pd.DataFrame({'reviewerID':users}).to_csv(review_path_clip + field + '_user.csv')
    pd.DataFrame({'asin':items}).to_csv(review_path_clip + field + '_item.csv')
    
    print(field + ':' + str(len(users)) + ',' +str(len(items)))
    
    return meta_data, items

# meta
def pre_process_meta(field:str, meta_data, items):
    # meta_data {0:{},}
    meta_input = [""] * len(items)
    for _, asin in pd.read_json(meta_path + field + ".json", orient='records', lines=True).iterrows():
        if asin['asin'] not in items:
            continue
        else:
            meta_prompt = "A product"
            for attr in ['title', 'description', 'categories', 'brand', 'feature']:
                if attr in asin and not isinstance(asin[attr], float) and len(asin[attr]) > 0:
                    meta_prompt += ", with " + attr + ": " + str(asin[attr])
            meta_prompt += "."

            meta_input[items.index(asin["asin"])] = meta_prompt
    embedding = model.encode(meta_input, convert_to_tensor=True)
    print(embedding.shape)
    torch.save(embedding, meta_path_clip + field + '.pth')

for field in ["All_Beauty", "Industrial_and_Scientific","Video_Games", "Digital_Music",  "Software",]:
    meta_data, items = pre_process_review(field)
    pre_process_meta(field, meta_data, items)