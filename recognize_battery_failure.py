import pandas as pd
#from sklearn.clusters import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

def read_in_data(file, labels):
    df = pd.read_csv('data/' + file)
    df = df.drop_duplicates()
    df['ds'] = pd.to_datetime(df['ReadDateTime'])
    df = df.groupby(['BolusId', 'ds']).agg({'Voltage': 'mean', 'BattRaw': 'mean', 'Battery': 'max'}).reset_index()
    fails = pd.read_csv('data/' + labels)
    df.loc[df['BolusId'].isin(fails['BolusId'].unique()), "failed"] = 1
    df.loc[~df['BolusId'].isin(fails['BolusId'].unique()), "failed"] = 0
    df = df[['BolusId', 'Voltage', 'ds', 'failed', 'Battery']]
    # addition failures from nick
    df2 = pd.read_csv('data/SaftBatteryFailures.csv')
    df2['ds'] = [x.date() for x in pd.to_datetime(df2['ReadDateTime'])]
    df2 = df2.groupby(['BolusId', 'ds']).agg({'Voltage': 'mean', 'BattRaw': 'mean'}).reset_index()
    df2['failed'] = 1
    df2['Battery'] = 'Saft'
    df2 = df2[['BolusId', 'Voltage', 'ds', 'failed', 'Battery']]
    df2['ds'] = df['ds'].apply(lambda x: np.datetime64(x))
    df = pd.concat([df, df2])
    return df

def get_segment(data, days_in=10, through=20):
    """
    Returns a subset of the input data containing the last 'days' worth of records for each 'BolusId'.

    Args:
        data (DataFrame): The input data containing records for each 'BolusId'.
        days (int, optional): The number of days to consider for each 'BolusId'. Defaults to 60.

    Returns:
        DataFrame: A subset of the input data containing the last 'days' worth of records for each 'BolusId'.
    """
    # Sort the data by 'BolusId' and 'ds' in ascending and descending order respectively
    tmp = data.sort_values(['BolusId', 'ds'], ascending=[True, True])

    # Select the 10th to 20th rows for each 'BolusId'
    tmp = tmp.groupby('BolusId').apply(lambda x: x.iloc[days_in: through]).reset_index(drop=True)

    # Add a new column 'time_step' which represents the cumulative count of records for each 'BolusId'
    tmp['time_step'] = tmp.groupby('BolusId').cumcount(ascending=True) + 1


    # Select the last 'days' records for each 'BolusId'
    #tmp = tmp.groupby('BolusId').head(days)

    return tmp

def get_begining_of_life(data, days_in=100):
    tmp = data.sort_values(['BolusId', 'ds'], ascending=[True, True])
    tmp = tmp.groupby('BolusId').apply(lambda x: x.iloc[0: days_in +1]).reset_index(drop=True)
    tmp['time_step'] = tmp.groupby('BolusId').cumcount(ascending=True) + 1
    return tmp


def pre_processing(data):
    X = data.pivot(columns='time_step', index=['BolusId', 'failed', 'Battery'], values='Voltage').reset_index()
    y = X[['failed', 'Battery', 'BolusId']]
    X_train, X_test, y_train, y_test = train_test_split(X, y['failed'], test_size=0.30, random_state=42, stratify=y['failed'], shuffle=True)
    X_test_ids = X_test[['BolusId', 'Battery']]
    X_train = X_train.drop(['failed', 'Battery', 'BolusId'], axis=1)
    X_test = X_test.drop(['failed', 'Battery', 'BolusId'], axis=1)
    
    return X_train, X_test, y_train, y_test, X_test_ids 

def augment_data():
    # chop up the failures to create more of them
    pass

def cluster_model(X, y):
    km = KMeans(n_clusters=3,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=0)
    y_km = km.fit_predict(X)

def my_cm(y_true, y_pred):
    cm = pd.DataFrame({'true':y_true, 'pred':y_pred})
    cm['tp'] = cm.apply(lambda x: 1 if x['true'] == 1 and x['pred'] == 1 else 0, axis=1)
    cm['tn'] = cm.apply(lambda x: 1 if x['true'] == 0 and x['pred'] == 0 else 0, axis=1)
    cm['fp'] = cm.apply(lambda x: 1 if x['true'] == 0 and x['pred'] == 1 else 0, axis=1)
    cm['fn'] = cm.apply(lambda x: 1 if x['true'] == 1 and x['pred'] == 0 else 0, axis=1)
    
    return cm.sum()[['tp', 'tn', 'fp', 'fn']]

def model(data):
    X_train, X_test, y_train, y_test, X_test_ids = pre_processing(data)
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='error')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    confusion = my_cm(y_test, y_pred)
    print(f"Confusion Matrix: \n { confusion}")
    output = X_test
    output['BolusId'] = X_test_ids['BolusId']
    output['Battery'] = X_test_ids['Battery']
    output['pred'] = y_pred
    output['true'] = y_test

    return output


def run(file, label_file):
    df = read_in_data(file, label_file)
    df = get_end_of_life(df, days=60) 
    X, y = pre_processing(df)
    return X, y

    

if __name__ == '__main__':
    df_raw = read_in_data('TadiranvsSaft_3-28-24_DailyAVG.csv', "Failed_units_Forest_view.csv")
    df =  get_begining_of_life(df_raw, days_in=150)
    output = model(df)
    
    
    print(output['true'].value_counts())

    # at 60 days before failure it can predict failure vs not failure very well. i need to take snapshot back in time and see how early we can recofnize failure.
    # 
