
def feature_engineering(mode='development'):
    import pandas as pd
    from preprocessing import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    cols = [
        'Bat Score'
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
        # , 'Bounce Angle'
        , 'Match Id'
        , 'Bounce Angle Delta'
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
        # , 'deliveries_since_last_wicket'
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

    df = df.drop_duplicates().reset_index(drop=True)
    
    train, test = train_test_split(df, train_size=0.8, random_state=123)

    if mode == 'development':
        return train, test
    else:
        return df