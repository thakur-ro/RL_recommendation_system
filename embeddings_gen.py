import pandas as pd
import numpy as np
import nltk
import re
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

"""
python embeddings_gen.py --file_path data/sample_df.csv --meta_path data/metadata_df.csv --glove_path data/glove.6B.50d.txt --target_dir data/
"""
def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        help="Path to the sampled data file",
                        required=True)
    parser.add_argument("--meta_path", type=str,
                        help="Path to the metadata file",
                        required=True)
    parser.add_argument("--glove_path", type=str,
                        help="Path to the pre-trained glove embeddings file",
                        required=True)
    parser.add_argument("--target_dir", type=str,
                        help="Path to save the embeddings file",
                        required=True)

    arguments = parser.parse_args()
    return arguments
#preparing pre-trained GLOVE embedding dictionary
def load_glove(glove_path):
    embeddings_dict = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    with open(glove_path, 'r') as f: #create unknown token
        for i, line in enumerate(f):
            pass
    n_vec = i + 1
    hidden_dim = len(line.split(' ')) - 1
    vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

    with open(glove_path, 'r') as f:
        for i, line in enumerate(f):
            vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
    average_vec = np.mean(vecs, axis=0)
    embeddings_dict['<unk>']=np.array(average_vec)

    return embeddings_dict

#clean title text
def clean_text(line): 
    line = line.strip() # Removing leading/trailing whitespace
    line = re.sub('\[.*\]', '', line) # Remove character heading
    line = re.sub('[^\w\s]', '', line) # Remove punctuation
    line = re.sub(r'\w*\d\w*', '', line)
    line = line.lower() # convert to lower case
    return line
#clean and tokenize title text
def product_description_gen(df):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    df.title = df.title.map(lambda x:tokenizer.tokenize(x))
    df.title = df.title.map(lambda x: " ".join(x))
    df['t_desc'] = df['title']
    df['t_desc'] = df['t_desc'].map(lambda x: clean_text(x))
    df.drop(columns = ['title'],inplace=True)
    return df


def main():
    args = parse_arguments()

    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)    
    
    #Get title for each product in sample dataframe
    sample_df = pd.read_csv(args.file_path)
    metadata = pd.read_csv(args.meta_path,usecols=['asin','title'])
    meta_df = sample_df.merge(metadata,left_on='item',right_on='asin',how='inner')[['item','user','rating','timestamp','title']]
    cleaned_df = product_description_gen(meta_df.copy())
    print("product title cleaned")
    item_df = cleaned_df.groupby(['t_desc'])['item'].unique().reset_index()
    item_df['count'] = item_df['item'].map(lambda x:len(x))
    item_df = item_df.sort_values(by='count',ascending=False).reset_index(drop=True)
    temp = item_df.loc[item_df['count']>1]

    #Find repeated titles
    repeated=[]
    for items in temp.item.values:
        for item in items:
            repeated.append(item)
    final_df = cleaned_df[~cleaned_df['item'].isin(repeated)]
    print("repeated titles removed")
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    final_df.t_desc = final_df.t_desc.map(lambda x:tokenizer.tokenize(x)) 

    #Get glove embeddings
    embeddings_dict = load_glove(args.glove_path)
    print("glove embeddings loaded")
    #generate embeddings for each item id
    unique_items = final_df.item.unique()
    item_dict = defaultdict(list)

    #embeddings lookup for each item id
    print("Starting embeddings lookup for unique item ids")
    for item in tqdm(unique_items):
        words = final_df[final_df['item']==item]['t_desc'].iat[0]
        item_arr = np.zeros((len(words),50),dtype=np.float32)
        for i,word in enumerate(words):
            if word not in embeddings_dict.keys():
                item_arr[i] = embeddings_dict['<unk>']
            else:
                item_arr[i] = embeddings_dict[word]
        avg_arr = np.mean(item_arr,axis = 0)
        avg_arr = '|'.join(str(x) for x in avg_arr)
        item_dict['item'].append(item)
        item_dict['embedding'].append(avg_arr)
    embeddings_df = pd.DataFrame(item_dict)
    embeddings_df.to_csv(args.target_dir+'embeddings.csv',index=False)
    print("Embeddings successfully saved to target directory!")

if __name__ == '__main__':
    main()