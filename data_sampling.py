import argparse
import pandas as pd
import numpy as np
import os
import re
import nltk

'''
python data_sampling.py --file_path data/Electronics.csv --metadata_path data/metadata_df.csv --target_dir data/ --num_items 15000 --min_history 19
'''

def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        help="Path to the ratings file",
                        required=True)

    parser.add_argument("--metadata_path", type=str,
                        help="Path to the metadata file",
                        required=True)

    parser.add_argument("--target_dir", type=str,
                        help="Path to save the sampled file",
                        required=True)

    parser.add_argument("--num_items", type=int,
                        help="Number of items (action space)",
                        required=True)

    parser.add_argument("--min_history", type=int,
                        help="Minimum user interactions",
                        required=True)

    arguments = parser.parse_args()
    return arguments

def clean_text(line):
        line = line.strip() # Removing leading/trailing whitespace
        line = re.sub('\[.*\]', '', line) # Remove character heading
        line = re.sub('[^\w\s]', '', line) # Remove punctuation
        line = re.sub(r'\w*\d\w*', '', line)
        line = line.lower() # convert to lower case
        return line

def product_description_gen(df):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    df.title = df.title.map(lambda x:tokenizer.tokenize(x))
    df.title = df.title.map(lambda x: " ".join(x))
    df['t_desc'] = df['title']
    df['t_desc'] = df['t_desc'].map(lambda x: clean_text(x))
    df.drop(columns = ['title'],inplace=True)
    return df

def clean_df(ratings, metadata, num_items, min_history):
    rate_cols = ['item','user','rating','timestamp']
    ratings.columns = rate_cols

    # Finding users who have given more than 1 rating for same item
    count_rate = ratings.groupby(['user','item'])['rating'].count().reset_index(name="num_ratings")
    wrong_users_df = count_rate.loc[count_rate.num_ratings > 1].reset_index(drop=True)
    merged_df = ratings.merge(wrong_users_df.drop_duplicates(),on='user',how='left',indicator=True)

    # Keeping only clean ratings (non-duplicate)
    new_ratings = merged_df.loc[merged_df['_merge'] == 'left_only'][['item_x','user','rating','timestamp']]
    new_ratings.columns = rate_cols
    print("Cleaned duplicate rows")

    # clean title for embeddings generation
    meta_df = new_ratings.merge(metadata,left_on='item',right_on='asin',how='inner')[['item','user','rating','timestamp','title']]
    cleaned_df = product_description_gen(meta_df.copy())
    print("Cleaned title for embeddings generation")    
    item_df = cleaned_df.groupby(['t_desc'])['item'].unique().reset_index()
    item_df['count'] = item_df['item'].map(lambda x:len(x))
    item_df = item_df.sort_values(by='count',ascending=False).reset_index(drop=True)
    temp = item_df.loc[item_df['count']>1]
    repeated=[]
    for items in temp.item.values:
        for item in items:
            repeated.append(item)
    final_df = cleaned_df[~cleaned_df['item'].isin(repeated)]
    sample_df = final_df.drop('t_desc',axis=1)   

    #Top K items    
    top_df = sample_df.groupby('item').aggregate(np.count_nonzero).sort_values(by='user',ascending=False).head(n=num_items).reset_index()

    print("Kept only " + str(num_items) + " items")

    df = sample_df.merge(top_df.drop_duplicates(),on = 'item', how = 'left').dropna().reset_index(drop=True)[['item','user_x','rating_x','timestamp_x']]
    df.columns = ratings.columns
    #From below take users with more than 18 ratings
    temp = df.groupby('user')['rating'].count().reset_index()

    print("Kept only top users")

    temp_df = df.merge(temp,on='user',how='left').dropna()
    final_df = temp_df.loc[temp_df.rating_y >= min_history].reset_index(drop=True).drop(['rating_y'], axis = 1)

    return final_df


def main():
    args = parse_arguments()

    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    ratings_df = pd.read_csv(args.file_path)

    metadata = pd.read_csv(args.metadata_path,usecols=['asin','title']).dropna(axis = 0)


    number_of_items = args.num_items

    min_history = args.min_history

    final_df = clean_df(ratings_df, metadata, number_of_items, min_history)
    
    final_df.columns = ['item', 'user', 'rating', 'timestamp']

    final_df.to_csv(args.target_dir + 'sample_df.csv', index=False)

    print("Clean dataframe saved to target directory")

if __name__ == '__main__':
    main()