import os.path

import numpy as np
import pandas as pd
import datetime
import pickle
from pmdarima.preprocessing import BoxCoxEndogTransformer
from arimaclust.models.centroid import ConsistentARMAClustering
from arimaclust.utils import get_equal_time_periods


def prepare_data(df, bc=False, ws=None):
    X = []
    for col in df.columns:
        if ws:
            x = df[col].rolling(ws).mean().dropna().values
        else:
            x = df[col].dropna().values
        if bc:
            x, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(x)
        X.append(x)
    return X


def preprocess_data(df):
    # drop symbols with unknown behaviour (piecewise constants)
    to_drop = ['ZMK', 'BYR', 'LVL', 'TMT', 'CLF', 'ZWL', 'JEP', 'CUP', 'ERN', 'CUC', 'USD', 'BMD', 'BBD', 'BSD', 'SDG',
               'KPW', 'PAB']
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # symbols with to mooch nans
    to_much_nans = ['BYN', 'IMP', 'GGP', 'ZMW', 'KYD']
    df.drop(columns=to_much_nans, inplace=True, errors='ignore')

    # metals
    metals = ['XAU', 'XAG']
    df.drop(columns=metals, inplace=True, errors='ignore')

    # coins
    coins = ['BTC']
    df.drop(columns=coins, inplace=True, errors='ignore')
    df = df.interpolate()
    return df


def main(df, delta, process_args, metric_args, results_path, k, ws, bc):
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    df = preprocess_data(df=df)

    start, end = datetime.datetime(year=2010, day=1, month=1), datetime.datetime(year=2020, day=1, month=1)
    periods = get_equal_time_periods(start_date=start, end_date=end, num_periods=delta)

    for current_start, current_end in periods:
        print(f'Clustering {current_start}:{current_end} time period')
        df_part = df[current_start:current_end].copy()

        # drop zero columns
        df_part.drop(columns=[col for col in df_part.columns if df_part[col].nunique() == 1], inplace=True)

        cols = df_part.columns.tolist()
        df_part = np.log1p(df_part.pct_change())

        X = prepare_data(df=df_part, bc=bc, ws=ws)
        model_part = ConsistentARMAClustering(k=k, metric='piccolo',
                                              metric_args=metric_args, process_args=process_args)
        model_part.fit(X)

        np.save(f'{results_path}/labels_{current_start}.npy', model_part.labels_)
        np.save(f'{results_path}/matrix_{current_start}.npy', model_part.dist_matrix)
        with open(f'{results_path}/cols_{current_start}', 'wb') as fp:
            pickle.dump(cols, fp)


if __name__ == '__main__':
    df_path = 'data/final_usd_2010_2020.csv'

    df_gdp = pd.read_html('https://ceoworld.biz/2022/03/31/economy-rankings-largest-countries-by-gdp-2022/')[0]
    df_gdp['Country'] = df_gdp['Country'].apply(lambda x: x.lower())
    # df_codes = pd.read_html('https://www.iban.com/currency-codes')[0]
    # df_codes['Country'] = df_codes['Country'].apply(lambda x: x.lower())
    df_codes = pd.read_csv('codes.csv')
    df_merged = pd.merge(left=df_codes, right=df_gdp, on='Country').sort_values('Rank')
    top_codes = df_merged[df_merged['Rank'] <= 100]['Code'].unique()
    top_codes = [tp for tp in top_codes if tp not in ['MXV', 'CHE', 'CHW', 'COU', 'UYI', 'VED', 'BOV']]
    df_usd = pd.read_csv(df_path, index_col='Unnamed: 0')

    df_usd = df_usd[top_codes]

    process_args = {'max_p': 3,
                    'max_q': 3,
                    'trace': False,
                    'd': 0
                    }

    metric_args = {'tau': 30}
    k = 5
    ws = None
    bc = False
    results_root = 'results_6_months'
    months_delta = 6  # months
    main(df=df_usd, delta=months_delta, process_args=process_args, metric_args=metric_args,
         results_path=results_root,
         k=k, ws=ws, bc=bc)
