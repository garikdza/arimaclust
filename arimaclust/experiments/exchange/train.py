import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from pmdarima.preprocessing import BoxCoxEndogTransformer
from arimaclust.models.centroid import ConsistentARMAClustering
from arimaclust.models.neighbours import NearestNeighbourModel


def prepare_data(df, bc=True, ws=None):
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


df_usd = pd.read_csv('data/final_usd_2010_2020.csv', index_col='Unnamed: 0')

# drop symbols with unknown behaviour (piecewise constants)
to_drop = ['ZMK', 'BYR', 'LVL', 'TMT', 'CLF', 'ZWL', 'JEP', 'CUP', 'ERN', 'CUC', 'USD', 'BMD', 'BBD', 'BSD', 'SDG',
           'KPW', 'PAB']
df_usd.drop(columns=to_drop, inplace=True)

# symbols with to mooch nans
to_much_nans = ['BYN', 'IMP', 'GGP', 'ZMW', 'KYD']
df_usd.drop(columns=to_much_nans, inplace=True)

# metals
metals = ['XAU', 'XAG']
df_usd.drop(columns=metals, inplace=True)

# coins
coins = ['BTC']
df_usd.drop(columns=coins, inplace=True)

nans = df_usd.isna().sum().sort_values(ascending=False)
nan_symb = nans.index[nans != 0]

for ns in nan_symb:
    df_usd[ns].plot()

    plt.title(f'{ns}:{nans[ns]}')
    plt.show()

for ns in nan_symb:
    l, u = df_usd[df_usd[ns].isna()].index.min(), df_usd[df_usd[ns].isna()].index.max()
    inter = df_usd[ns].interpolate()
    df_usd[l:u][ns].plot(style='r-', label='True')
    inter[l:u].plot(style='g--', label='Interpolated')
    plt.title(f'{ns}:{nans[ns]}')
    plt.legend()
    plt.show()

df_usd = df_usd.interpolate()

start, end = datetime(year=2010, day=1, month=1), datetime(year=2020, day=1, month=1)
current_start = start
n_years = end.year - start.year

process_args = {'max_p': 3,
                'max_q': 3,
                'trace': False,
                'd': 0
                }

metric_args = {'tau': 30}
k = 5
ws = None
results_root = 'results_out_sample/'

for i in range(n_years):
    current_end = datetime(year=current_start.year + 1, day=1, month=1)
    print(current_start, current_end)

    df_part = df_usd[str(current_start):str(current_end)].copy()

    # drop zero columns
    # df_part.drop(columns=[col for col in df_part.columns if df_part[col].nunique() == 1], inplace=True)

    cols = df_part.columns.tolist()
    df_part = np.log1p(df_part.pct_change())

    X = prepare_data(df=df_part, bc=False, ws=ws)
    model_part = ConsistentARMAClustering(k=k, metric='piccolo',
                                          metric_args=metric_args, process_args=process_args)
    model_part.fit(X)

    np.save(f'{results_root}/labels_{current_start.year}.npy', model_part.labels_)
    np.save(f'{results_root}/matrix_{current_start.year}.npy', model_part.dist_matrix)
    with open(f'{results_root}/cols_{current_start.year}', 'wb') as fp:
        pickle.dump(cols, fp)
    current_start = current_end
