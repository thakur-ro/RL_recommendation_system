import argparse
import pandas as pd
import numpy as np
import os

'''
python data_sampling.py --file_path dataset/Electronics.csv --target_dir dataset/ --num_items 15000 --min_history 19
'''

def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        help="Path to the ratings file",
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

def clean_df(ratings, num_items, min_history):
    rate_cols = ['item','user','rating','timestamp']
    ratings.columns = rate_cols

    # Finding users who have given more than 1 rating for same item
    count_rate = ratings.groupby(['user','item'])['rating'].count().reset_index(name="num_ratings")
    wrong_users_df = count_rate.loc[count_rate.num_ratings > 1].reset_index(drop=True)
    merged_df = ratings.merge(wrong_users_df.drop_duplicates(),on='user',how='left',indicator=True)

    # Keeping only clean ratings (non-duplicate)
    new_ratings = merged_df.loc[merged_df['_merge'] == 'left_only'][['item_x','user','rating','timestamp']]
    print("Cleaned duplicate rows")

    #Top K items    
    top_df = new_ratings.groupby('item_x').aggregate(np.count_nonzero).sort_values(by='user',ascending=False).head(n=num_items).reset_index()

    print("Kept only " + str(num_items) + " items")

    df = new_ratings.merge(top_df.drop_duplicates(),on = 'item_x', how = 'left').dropna().reset_index(drop=True)[['item_x','user_x','rating_x','timestamp_x']]
    df.columns = rate_cols

    #From below take users with more than 18 ratings
    temp = df.groupby('user')['rating'].count().reset_index()

    print("Kept only top users")

    temp_df = df.merge(temp,on='user',how='left').dropna()
    final_df = temp_df.loc[temp_df.rating_y >= min_history].reset_index()

    return final_df


def main():
    args = parse_arguments()

    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    ratings_df = pd.read_csv(args.file_path)

    number_of_items = args.num_items

    min_history = args.min_history

    final_df = clean_df(ratings_df, number_of_items, min_history).drop(['index', 'rating_y'], axis = 1)
    
    final_df.columns = ['item', 'user', 'rating', 'timestamp']

    final_df.to_csv(args.target_dir + 'sample_df.csv', index=False)

    print("Clean dataframe saved to target directory")

if __name__ == '__main__':
    main()