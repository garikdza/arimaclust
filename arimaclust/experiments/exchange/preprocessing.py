import os

import pandas as pd


def get_key(d, key):
    try:
        return d[key]
    except Exception as e:
        return None


def get_currency_series(df, symb):
    x = df['rates'].apply(lambda x: get_key(x, symb))
    return pd.Series(index=x.index, data=x.values, name=symb)


def concat_datas(jsons_root):
    # concat downloaded json files
    paths = sorted([os.path.join(jsons_root,path) for path in os.listdir(jsons_root)],reverse=True)

    df_ = pd.read_json(paths[0])
    currencies = df_['rates'].apply(lambda x: list(x.keys()))
    unique_currencies = []
    for row in currencies:
        unique_currencies.extend(row)
    unique_currencies = list(set(unique_currencies))
    currencies_data = get_currency_series(df_, symb=unique_currencies[0])

    for currency in unique_currencies[1:]:
        curr_data_current = get_currency_series(df_, symb=currency)
        currencies_data = pd.concat([currencies_data, curr_data_current], axis=1)

    final_data = currencies_data

    for file_path in paths[1:]:
        df_ = pd.read_json(file_path)
        currencies = df_['rates'].apply(lambda x: list(x.keys()))
        unique_currencies = []
        for row in currencies:
            unique_currencies.extend(row)
        unique_currencies = list(set(unique_currencies))
        currencies_data = get_currency_series(df_, symb=unique_currencies[0])

        for currency in unique_currencies[1:]:
            curr_data_current = get_currency_series(df_, symb=currency)
            currencies_data = pd.concat([currencies_data, curr_data_current], axis=1)
        final_data = pd.concat([currencies_data, final_data], axis=0)

    return final_data


if __name__ == '__main__':
    roots = 'data/usd'
    saving_path = 'data/final_usd.csv'
    final_data = concat_datas(jsons_root=roots)
    final_data.to_csv(saving_path)
