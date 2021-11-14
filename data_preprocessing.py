import argparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict

'''
python data_preprocessing.py --file_path data/sample_df.csv --target_dir data/ --min_state_hist 10 --min_actions 1 --states_ratio 0.7 --samples_per_user 10
'''

def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        help="Path to the sampled file",
                        required=True)

    parser.add_argument("--target_dir", type=str,
                        help="Path to save the preprocessed file",
                        required=True)

    parser.add_argument("--min_state_hist", type=int,
                        help="Minimum number of historic user ratings (state)",
                        required=True)

    parser.add_argument("--min_actions", type=int,
                        help="Number of items to recommend at a time step",
                        required=True)

    parser.add_argument("--states_ratio", type=float,
                        help="ratio from user history for state generation",
                        required=True)

    parser.add_argument("--samples_per_user", type=int,
                        help="number of samples (state, action, next_state) to create per user",
                        required=True)

    arguments = parser.parse_args()
    return arguments

def create_history(df):
    users = df['user'].unique()
    items = df['item'].unique()

    print("Unique users: ", len(users))
    print("Unique items: ", len(items))

    historic_users = []

    for i, user in enumerate(users):
        temp = df[df['user'] == user]
        twmp = temp.sort_values('timestamp').reset_index(drop=True)
        historic_users.append(temp)

    for user in historic_users:
        user['timestamp'] += user.index

    return historic_users

def create_sample_sequence(user_history, min_num_states, min_num_actions, states_ratio, sample_size):
    n = len(user_history)
    sep_ratio = int(states_ratio * n)
    print("Separation ratio: ", sep_ratio)
    print("num state",min_num_states)
    num_states, num_actions = [], []
    num_states = [min(np.random.randint(min_num_states, sep_ratio), sep_ratio) for i in range(sample_size)]
    num_actions = [min_num_actions for i in range(sample_size)]

    states, actions = [], []

    for i in range(len(num_states)):
        sample_states = user_history.iloc[0: sep_ratio].sample(num_states[i])
        sample_state, sample_action = [], []

        for j in range(num_states[i]):
            row = sample_states.iloc[j]
            state = str(row.loc['item']) + '&' + str(row.loc['rating'])
            sample_state.append(state)

        row = user_history.iloc[-(n - sep_ratio):].sample(num_actions[i])
        action = str(row['item'].item()) + '&' + str(row['rating'].item())
        sample_action.append(action)
        states.append(sample_state)
        actions.append(sample_action)

    return states, actions


def main():
    args = parse_arguments()

    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    sampled_df = pd.read_csv(args.file_path)

    # CREATE USER HISTORY
    user_history = create_history(sampled_df)

    print("User history created")

    sequence_dict = defaultdict(list)

    # CREATE SAMPLE HISTORY
    for i, users in enumerate(user_history):
        states, actions = create_sample_sequence(users, args.min_state_hist, args.min_actions, args.states_ratio, args.samples_per_user)

        for j in range(len(states)):
            state_str = '|'.join(states[j])
            action_str = '|'.join(actions[j])
            next_state_str = state_str + "|" + action_str
            sequence_dict['state'].append(state_str)
            sequence_dict['action_reward'].append(action_str)
            sequence_dict['next_state'].append(next_state_str)
        
    sequence_df = pd.DataFrame(sequence_dict)
    print("Sequenced dataframe created")

    sequence_df.to_csv(args.target_dir + 'preprocessed_df.csv', index=False)

if __name__ == '__main__':
    main()