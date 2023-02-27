import requests
import json
import datetime
import os
import dotenv


def get_data(start, end, base, api_key, saving_root='data'):
    # TODO check start and and to be starts of the years

    if not os.path.isdir(saving_root):
        os.makedirs(saving_root)
    payload = {}
    headers = {
        "apikey": api_key
    }

    current_end = end
    n_years = end.year - start.year

    for i in range(n_years):
        current_start = datetime.datetime(year=current_end.year - 1, day=current_end.day, month=current_end.month)
        print(f'start download from {current_start} to {current_end}')
        url = f"https://api.apilayer.com/exchangerates_data/timeseries?" \
              f"start_date={current_start.date()}&end_date={current_end.date()}&base={base}"
        response = requests.request("GET", url, headers=headers, data=payload)
        status_code = response.status_code
        if status_code == 200:
            result = response.json()
            full_path = os.path.abspath(f'{saving_root}/{current_start.date()}_{current_end.date()}.json')
            print(f'saving at:{full_path}')
            with open(full_path, 'w') as f:
                json.dump(result, f)
        else:
            print(f'Cannot download data from {current_start} to {current_end} for {base}')

        current_end = current_start


if __name__ == '__main__':
    dotenv.load_dotenv('.env')
    end = datetime.datetime(year=2023, day=1, month=1)
    start = datetime.datetime(year=2020, day=1, month=1)
    base = 'USD'
    saving_path = 'data/usd/'
    get_data(start=start, end=end, saving_root=saving_path, base=base, api_key=os.getenv('API_KEY'))
