import requests
import json
import datetime
import os


END = datetime.datetime(year=2023, day=1, month=1)
START = datetime.datetime(year=2000, day=1, month=1)
BASE = 'AMD'
payload = {}
headers = {
    "apikey": "***REMOVED***"
}

current_end = END
n_years = END.year - START.year
for i in range(n_years):
    current_start = datetime.datetime(year=current_end.year - 1, day=current_end.day, month=current_end.month)
    print(current_start, current_end)
    url = f"https://api.apilayer.com/exchangerates_data/timeseries?start_date={current_start.date()}&end_date={current_end.date()}&base={BASE}"
    response = requests.request("GET", url, headers=headers, data=payload)
    status_code = response.status_code
    result = response.json()
    full_path = os.path.abspath(f'data/{current_start.date()}_{current_end.date()}.json')
    print(full_path)
    with open(full_path, 'w') as f:
        json.dump(result, f)

    current_end = current_start

#
#

#
#
