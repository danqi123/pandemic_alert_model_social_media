import datetime
import pandas as pd


def get_dates(file: str, up_or_down: int):
    gs = pd.read_csv(file)
    gs_up_dates = list(gs[gs['up/down trend'] == up_or_down]['date'])
    return gs_up_dates

def transfer_date(date):
    date_int = [int(number) for number in date.split("-")]
    return date_int

def filter_dates(start_date: str, end_date: str, input_dates_list: list):
    new_list = []
    y1, m1, d1 = transfer_date(start_date)
    d_start = datetime.datetime(y1, m1, d1)

    y2, m2, d2 = transfer_date(end_date)
    d_end = datetime.datetime(y2, m2, d2)

    for d in input_dates_list:
        y, m, d = transfer_date(d)
        d_test = datetime.datetime(y, m, d)
        if d_start <= d_test <= d_end:
            new_list.append(d_test)
    return new_list

def filter_dates_trend_analysis(date_split: str, input_dates_list: list):
    new_list = []
    y1, m1, d1 = transfer_date(date_split)
    d_split = datetime.datetime(y1, m1, d1)

    for d in input_dates_list:
        y, m, d = transfer_date(d)
        d_test = datetime.datetime(y, m, d)
        if d_test < d_split:
            str_time = d_test.strftime("%Y-%m-%d")
            new_list.append(str_time)
    return new_list

def get_date_interval(gold_standard_dates_list: list, proxy_dates_list: list):
    test_date_hit = []
    time_diff = []
    #proxy_dates_list = [date + datetime.timedelta(days=13) for date in proxy_dates_list]
    for d_standard in gold_standard_dates_list:
        for d_test in proxy_dates_list:
            diff = (d_test - d_standard).days
            if -30 <= diff < 0:
                test_date_hit.append(d_standard)
                time_diff.append(diff)
                break

    hit_percentage = len(test_date_hit)/len(gold_standard_dates_list)
    print(test_date_hit)
    print(hit_percentage)
    time_diff = [t+13 for t in time_diff]
    return time_diff