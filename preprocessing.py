# xW Model #
def preprocessing(cols, mode='development'):
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(r'C:\Users\anude\Desktop\Jupyter Notebook Projects\Cricket NSW\Data\T20M BBB - latest.csv')

    df['Series and Season'] = df['Series'] + ' - ' + df['Season'].astype('str')
    
    # List of series that we will be using for the model
    # These are series where the data is reliable and available.
    series_list = ["Aus Domestic T20 M - 2020-21",
    "Aus Domestic T20 M - 2021-22",
    "Aus Domestic T20 M - 2022-23",
    "Aus Domestic T20 M - 2023-24",
    "Indian Premier League T20 M - 2020",
    "Indian Premier League T20 M - 2023",
    "Indian Premier League T20 M - 2024",
    "International T20 M - 2020",
    "International T20 M - 2020-21",
    "International T20 M - 2021",
    "International T20 M - 2021-22",
    "International T20 M - 2022",
    "International T20 M - 2022-23",
    "International T20 M - 2023",
    "International T20 M - 2023-24",
    "International T20 M - 2024",
    "International T20 World Cup M - 2021",
    "International T20 World Cup M - 2022-23",
    "International T20 World Cup M - 2024",
    "UAE Domestic T20 M - 2023",
    "UAE Domestic T20 M - 2024",
    "USA Domestic T20 M - 2024"]
    
    df = df.loc[df['Series and Season'].isin(series_list)].reset_index(drop=True)
    
    # Define wicket column for only bowler induced wickets
    df['wicket'] = 0
    df.loc[df['How Out'].isin(['C','B','LB','S']), 'wicket'] = 1

    df['Total Runs'] = df['Bat Score'] + df['Wides'] + df['Noballs'] + df['Byes'] + df['Legbyes']
    df['dot_ball'] = df['Total Runs'].apply(lambda x: 1 if x==0 else 0)
    df = df.sort_values(by=['Match Id', 'MatchInnings', 'Over', 'Ball In Over'])
    df['running_total'] = df.groupby(['Match Id', 'MatchInnings'])['Total Runs'].cumsum()
    df['running_wickets'] = df.groupby(['Match Id', 'MatchInnings'])['wicket'].cumsum()
    df['legal_delivery'] = df.apply(lambda x: 0 if (x['Wides'] > 0 or x['Noballs'] > 0) else 1, axis=1)
    df['legal_balls_bowled'] = df.groupby(['Match Id', 'MatchInnings'])['legal_delivery'].cumsum()
    
    df['run_rate'] = df['running_total']/(df['legal_balls_bowled'] // 6 + (df['legal_balls_bowled'] % 6) / 6)
    df['run_rate'] = df.apply(
    lambda x: x['running_total'] if np.isinf(x['run_rate']) else x['run_rate'], axis=1
)
    # Rolling calculations for the last 6, 9, and 12 deliveries
    for window in [6, 9, 12]:
        df[f'runs_last_{window}_balls'] = (
            df.groupby(['Match Id', 'MatchInnings'])['Total Runs']
            .shift(1)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(drop=True)
        )

    
    # Rolling calculations for the last 6, 9, and 12 deliveries
    for window in [6, 9, 12]:
        df[f'dot_balls_last_{window}_balls'] = (
            df.groupby(['Match Id', 'MatchInnings'])['dot_ball']
            .shift(1)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(drop=True)
        )

    df['wicket_last_six_deliveries'] = df.groupby(['Match Id', 'MatchInnings'])['wicket']\
                                        .apply(lambda x: x.shift(1))\
                                        .rolling(6,min_periods=1)\
                                        .max()\
                                        .reset_index(drop=True)\

    # Bin and Label Encode coordinate data
    pitch_x_buckets = np.arange(-1550,1650,100)
    pitch_y_buckets = np.arange(-2000,13250,250)
    at_stumps_x_buckets = np.arange(-1750,2250,100)
    at_stumps_y_buckets = np.arange(0,2100,100)
    df['Pitch X Bins'] = pd.cut(df['Pitch X'], pitch_x_buckets, labels=range(len(pitch_x_buckets)-1))
    df['Pitch Y Bins'] = pd.cut(df['Pitch Y'], pitch_y_buckets, labels=range(len(pitch_y_buckets)-1))
    df['At Stumps X Bins'] = pd.cut(df['At Stumps X'], at_stumps_x_buckets, labels=range(len(at_stumps_x_buckets)-1))
    df['At Stumps Y Bins'] = pd.cut(df['At Stumps Y'], at_stumps_y_buckets, labels=range(len(at_stumps_y_buckets)-1))

    print('at the end of the preprocessing stage the shape is ', df.shape,' ', mode)
    
    if mode == 'development':
        df = df.loc[
                (df['Ball Speed'] >= 80)
                & (df['Ball Speed'] <= 160)
                & (df['Movement In Air'] <= 5)
                & (df['Movement In Air'] >= -5)
                & (df['Movement Off Pitch'] <= 5)
                & (df['Movement Off Pitch'] >= -5)
                & (df['Bounce Angle']<=15)
                & (df['Bounce Angle Delta']<=8)
                & (df['Bounce Angle Delta']!=80)
                & (df['Drop Angle']!=0)
                & (df['Drop Angle']<=-8)
                & (df['Drop Angle']>=-20)
                & (df['Bowler Style']!='Unknown')
                & (df['Pitch X'] <=1525)
                & (df['Pitch X'] >=-1525)
                & (df['Pitch Y'] <= 13000)
                & (df['Pitch Y'] >= -2000)
                & (df['At Stumps Y']>=0)
                & (df['At Stumps Y']<=2000)
                & (df['MatchInnings']<=2)
                    ]\
                    .reset_index(drop=True)
    else:
        pass

    print('at the end of the preprocessing stage the shape is ', df.shape)

    for col in ['Pitch X Bins', 'Pitch Y Bins', 'At Stumps X Bins', 'At Stumps Y Bins']:
        try:
            df[col] = df[col].astype(float)
        except:
            print(col)
    
    df = df[cols]

    print(df.isnull().sum())

    df = df.dropna().reset_index(drop=True)
    
    return df
    