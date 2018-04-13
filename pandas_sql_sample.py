import sys
import psycopg2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pickle

# 引数に指定されたＳＱＬファイルを実行する
def execute_sql_from_file(sqlfile):
    try:
        cnn = psycopg2.connect("dbname=keiba_ai host=localhost user=postgres password=postgres")
        cur = cnn.cursor()

        target_sql_file = open(sqlfile)
        sql_data = target_sql_file.read()  # ファイル終端まで全て読んだデータを返す
        target_sql_file.close()

        cnn.commit()
        cur.execute(sql_data)
        rows = cur.fetchall()

        cur.close()
        cnn.close()

        return rows
    except (psycopg2.OperationalError) as e:
        print (e)
    finally:
        cur.close()
        cnn.close()


# テーブルをDataFrameへロード
race_result_df = pd.DataFrame(execute_sql_from_file('./sql/race_top3_test.sql'))

race_result_df.columns =['date','venue','raceNumber','raceName','distance','finishOrder','top3flg','headsCount','postPosition','Sex','horseAge','time3F','loadWeight','horseWeight','dhorseWeight','oddsOrder','odds']
# DataFrameをランダムまぜまぜ
race_result_df['distance'] = race_result_df['distance'].fillna(0.0).astype(int)
race_result_df['postPosition'] = race_result_df['postPosition'].fillna(0.0).astype(int)
race_result_df['headsCount'] = race_result_df['headsCount'].fillna(0.0).astype(int)
race_result_df['time3F'] = race_result_df['time3F'].fillna(0.0).astype(float)
race_result_df['loadWeight'] = race_result_df['loadWeight'].fillna(0.0).astype(float)
race_result_df['horseWeight'] = race_result_df['horseWeight'].fillna(0.0).astype(float)
race_result_df['dhorseWeight'] = race_result_df['dhorseWeight'].fillna(0.0).astype(float)
race_result_df['oddsOrder'] = race_result_df['oddsOrder'].fillna(0.0).astype(int)
race_result_df['odds'] = race_result_df['odds'].fillna(0.0).astype(float)
race_result_df['Sex'] = race_result_df['Sex'].fillna(0.0).astype(int)
race_result_df['horseAge'] = race_result_df['horseAge'].fillna(0.0).astype(int)
race_result_df.reindex(np.random.permutation(race_result_df.index)).reset_index(drop=True)

X_train = race_result_df[['distance','postPosition','headsCount','time3F','loadWeight','horseWeight','dhorseWeight','oddsOrder','odds','Sex','horseAge']]
Y_train = race_result_df['top3flg']

race_result_df
