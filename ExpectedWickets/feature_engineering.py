
def feature_engineering(mode='development'):
    import pandas as pd
    from preprocessing import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    
    cols = [
        'wicket'
        , 'Pitch X Bins'
        , 'Pitch Y Bins'
        , 'Ball Speed'
        , 'At Stumps X Bins'
        , 'At Stumps Y Bins'
        , 'Over'
        , 'Movement In Air'
        , 'Movement Off Pitch'
        , 'MatchInnings'
        # , 'run_rate'
        # , 'wicket_last_six_deliveries'
        # , 'running_wickets'
        # , 'running_total'
        , 'Bounce Angle'
        , 'Match Id'
        # , 'Bounce Angle Delta'
        # , 'Drop Angle'
        # , 'Bowler Style'
        # , 'Bowler Hand'
        # , 'Striker Hand'
        # , 'runs_last_6_balls'
        # , 'runs_last_9_balls'
        # , 'runs_last_12_balls'
        # , 'dot_balls_last_6_balls'
        # , 'dot_balls_last_9_balls'
        # , 'dot_balls_last_12_balls'
        , 'Ball In Over'
        , 'Power Play'
    ]
    
    df = preprocessing(cols, mode)

    # df['Bowler Right Handed'] = 1
    # df.loc[df['Bowler Hand']=='Left','Bowler Right Handed'] = 0
    # df['Striker Right Handed'] = 1
    # df.loc[df['Striker Hand']=='Left','Striker Right Handed'] = 0

    # df_encoded = pd.get_dummies(df['Bowler Style'], dtype=int)
    # df = df.merge(df_encoded,left_index=True,right_index=True)

    # df = df.drop(
    #     columns=[
    #         # 'Bowler Style'
    #         # ,'Bowler Hand'
    #         # ,'Striker Hand'
    #     ]
    # )

#     cols_to_standardize = [
#     'Pitch X'
#     , 'Pitch Y'
#     , 'Ball Speed'
#     , 'At Stumps X'
#     , 'At Stumps Y'
#     , 'Movement In Air'
#     , 'Movement Off Pitch'
#     # , 'Bounce Angle'
#     # , 'Bounce Angle Delta'
#     # , 'Drop Angle'
# ]
    

    # scaler = StandardScaler()
    # df_transformed = scaler.fit_transform(df[cols_to_standardize])
    # df.drop(columns=cols_to_standardize,inplace=True)
    # df[cols_to_standardize] = df_transformed

    # df['Over'] = df['Over'].apply(lambda x: (x-0)/(20-0))

    df.fillna(0,inplace=True)

    train, test = train_test_split(df, train_size=0.8, random_state=123, stratify=df['wicket'])

    # def smote_func(smote, train):
    #     x_train_smote, y_train_smote = smote.fit_resample(train.drop('wicket',axis=1),train['wicket'])
    #     x_train_smote['wicket'] = y_train_smote
    #     train = x_train_smote
    #     return train

    # if sampling == 'SMOTE':
    #     smote = SMOTE(random_state=123)
    #     train = smote_func(smote, train)
    # elif sampling == 'ADASYN':
    #     smote = ADASYN(random_state=123)
    #     train = smote_func(smote, train)
    # elif sampling == 'SMOTETomek':
    #     smote = SMOTETomek(random_state=123)
    #     train = smote_func(smote, train)
    # elif sampling == 'SMOTEENN':
    #     smote = SMOTEENN(random_state=123)
    #     train = smote_func(smote, train)
    # elif sampling == 'Under':
    #     smote = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    #     train = smote_func(smote, train)
    # else:
    #     pass

    if mode == 'development':
        return train, test
    else:
        return df
    