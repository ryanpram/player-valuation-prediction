# this train file is intended to training the best model with best params 

import pickle
import numpy as np 
import pandas as pd 
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb




# Parameters (the parameter is get from model and data exploration on )
version = 1
output_file = f'model_v{version}.bin'
best_eta = 0.5
best_max_depth = 6
best_min_child_weight = 5
xgb_params = {
    'eta': best_eta, 
    'max_depth': best_max_depth,
    'min_child_weight': best_min_child_weight,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

# Metric function (RMSE)
def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)



# Data preparation & cleaning
players_df = pd.read_csv("./data-raw/players.csv")
games_df = pd.read_csv("./data-raw/games.csv")
appearances_df = pd.read_csv("./data-raw/appearances.csv")

players_df = players_df[players_df["last_season"] == 2023]

games_and_apps_df = appearances_df.merge(games_df, on=['game_id'], how='left')

def get_player_stats(player_id, season, df):
    
    df = df[df['player_id'] == player_id]
    df = df[df['season'] == season]
    
    if (df.shape[0] == 0):
        Out = [(np.nan, season,0,0,0,0,0,0,0)]
        out_df = pd.DataFrame(data = Out, columns = ['player_id','season','goals','games',
                                                     'assists','minutes_played','goals_for','goals_against','clean_sheet'])
        return out_df
    
    else:
        
        df["goals_for"] = df.apply(lambda row: row['home_club_goals'] if row['home_club_id'] == row['player_club_id'] 
                      else row['away_club_goals'] if row['away_club_id'] == row['player_club_id'] 
                      else np.nan, axis=1)
        df["goals_against"] = df.apply(lambda row: row['away_club_goals'] if row['home_club_id'] == row['player_club_id'] 
                      else row['home_club_goals'] if row['away_club_id'] == row['player_club_id'] 
                      else np.nan, axis=1)
        df['clean_sheet'] = df.apply(lambda row: 1 if row['goals_against'] == 0
                      else 0 if row['goals_against'] > 0
                      else np.nan, axis=1)
        
        df = df.groupby(['player_id',"season"],as_index=False).agg({'goals': 'sum', 'game_id': 'nunique', 
                                                                      'assists': 'sum', 'minutes_played' : 'sum', 'goals_for' : 'sum',
                                                                      'goals_against' : 'sum', 'clean_sheet' : 'sum'})
        out_df = df.rename(columns={'game_id': 'games'})

        return out_df
    
    
stat_season = 2022
for index in players_df.index:
    id = players_df.loc[index][0]
    name = players_df.loc[index][1]
    stats = get_player_stats(id, stat_season, games_and_apps_df)
    players_df.at[index,'goals_{}'.format(stat_season)]= stats['goals'][0]
    players_df.at[index,'games_{}'.format(stat_season)]= stats['games'][0]
    players_df.at[index,'assists_{}'.format(stat_season)]= stats['assists'][0]
    players_df.at[index,'minutes_played_{}'.format(stat_season)]= stats['minutes_played'][0]
    players_df.at[index,'goals_for_{}'.format(stat_season)]= stats['goals_for'][0]
    players_df.at[index,'goals_against_{}'.format(stat_season)]= stats['goals_against'][0]
    players_df.at[index,'clean_sheet_{}'.format(stat_season)]= stats['clean_sheet'][0]


players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'])


players_df = players_df[players_df['date_of_birth'].isnull() == False]
now = datetime.now()

players_df['age'] = (now - players_df['date_of_birth']).apply(lambda x: x.days) / 365.25
players_df['age'] = players_df['age'].round().astype(int)

players_df = players_df.dropna(subset = ['foot','height_in_cm','market_value_in_eur','highest_market_value_in_eur'])
players_df.isnull().sum()

players_df = players_df.reset_index(drop=True)


#pick relavant feature to train and divide it based the type of data
numerical_features = ['height_in_cm',
                      'goals_2022',
                      'games_2022',
                      'assists_2022',
                      'minutes_played_2022',
                      'goals_for_2022',
                      'goals_against_2022',
                      'clean_sheet_2022',
                      'age'
                     ]
categorical_features = ['position',
                        'sub_position',
                        'foot',
                        'current_club_name',
                        'current_club_domestic_competition_id'
                       ]


# Data Split
df_full_train, df_test = train_test_split(players_df, test_size=0.2, random_state=1)
df_full_train, df_test = train_test_split(players_df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.market_value_in_eur.values)
y_val = np.log1p(df_val.market_value_in_eur.values)
y_test = np.log1p(df_test.market_value_in_eur.values)

del df_train["market_value_in_eur"]
del df_val["market_value_in_eur"]
del df_test["market_value_in_eur"]


# Train model
print('training the final model')

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = np.log1p(df_full_train.market_value_in_eur.values)
del df_full_train['market_value_in_eur']

dv = DictVectorizer(sparse=False)

train_dict = df_full_train[categorical_features + numerical_features].to_dict(orient='records')
X_full_train = dv.fit_transform(train_dict)

val_dict = df_test[categorical_features + numerical_features].to_dict(orient='records')
X_test = dv.transform(val_dict)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=dv.feature_names_)

dtest = xgb.DMatrix(X_test, feature_names=dv.feature_names_)

final_model = xgb.train(xgb_params, 
                  dfulltrain, 
                  num_boost_round=10, 
                 )
y_pred = final_model.predict(dtest)

rmse_final_score = rmse(y_test, y_pred)

print(f'rmse_score={rmse_final_score}')

#Save the model


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)

print(f'the model is saved to {output_file}')
