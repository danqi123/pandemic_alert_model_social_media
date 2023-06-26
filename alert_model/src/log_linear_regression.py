"""Source functions for run log-linear regression model."""

from plotly.subplots import make_subplots
import pandas as pd
import datetime
from math import log
from statsmodels.tsa.seasonal import STL
from date import filter_dates_trend_analysis
import logging
from tqdm import tqdm
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
from startup import PROCESSED_DATA, TREND_VISUALIZATION_FIG, GOLD_STANDARD_TREND, GOOGLE_TREND, TWITTER_TREND, COMBINED_TREND, \
    GOOGLE_TREND_METRICS, TWITTER_TREND_METRICS, COMBINED_TREND_METRICS, COVID_KG_DIR, PAIRWISE_EVENT_FIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_csv(file:str) -> pd.DataFrame:
    """
    Read .csv file.
    """
    g_trends = pd.read_csv(file, encoding='utf_8_sig')
    return g_trends

def STL_decomposition(dataframe: pd.DataFrame, sym: str, period:int = 7) -> pd.DataFrame:
    """
    this function will use STL to get residual component.
    """
    date_list = list(dataframe['date'])
    dataframe = dataframe.set_index("date")
    trend_df = dataframe.copy()

    stl = STL(dataframe[sym], period=period, robust=True)
    result = stl.fit()

    trend = result.trend.values.tolist()
    trend_df[sym] = trend
    trend_df['date'] = date_list
    return trend_df


def linear_model(dataframe: pd.DataFrame, symptom: str, time_window: int, stride: int, alpha: float, proxy: str) -> pd.DataFrame:
    """
    perform linear model on individual symptom or surveillance data_folder,
    and generate up and down trend .csv file.
    """
    data = list(dataframe[symptom])
    order = list(dataframe["order"])
    result_df = pd.DataFrame()

    start_list = []
    end_list = []
    date_list = []
    slope_list = []
    p_list = []
    for i in range(0, len(data)-time_window+1, stride):
        start_list.append(i+1)
        end_list.append(i+time_window)
        date_list.append(list(dataframe.date)[i])
        y = data[i:i+time_window]
        log_y = [log(number) if number > 0 else 0 for number in y]
        x = order[i:i+time_window]
        slope, _, r, p, _ = linregress(x, log_y)
        slope_list.append(slope)
        p_list.append(p)

    result_df['start'] = start_list
    result_df['end'] = end_list
    result_df['date'] = date_list
    result_df['slope'] = slope_list
    result_df['p_value'] = p_list

    _, p_val_corrected, _, _ = multipletests(result_df['p_value'], alpha=alpha, method='holm', returnsorted=False)
    result_df[f'p_value Holm (alpha={alpha})'] = p_val_corrected

    row, column = result_df.shape
    up_down_trend = []
    for r in range(row):
        if result_df.iloc[r].loc['slope'] > 0 and result_df.iloc[r].loc[f'p_value Holm (alpha={alpha})'] < alpha:
            up_down_trend.append(1)
        elif result_df.iloc[r].loc['slope'] < 0 and result_df.iloc[r].loc[f'p_value Holm (alpha={alpha})'] < alpha:
            up_down_trend.append(-1)
        else:
            up_down_trend.append(0)
    result_df['up/down trend'] = up_down_trend

    # save dataframe to csv files
    if proxy == "gold_standard":
        result_df.to_csv(f'{GOLD_STANDARD_TREND}/{symptom}_trend_label.csv')
    elif proxy == "Google_Trends":
        result_df.to_csv(f'{GOOGLE_TREND}/Google_Trends_{symptom}_trend_label.csv')
    elif proxy == "Twitter":
        result_df.to_csv(f'{TWITTER_TREND}/Twitter_{symptom}_trend_label.csv')
    elif proxy == "combined":
        result_df.to_csv(f'{COMBINED_TREND}/Combined_{symptom}_trend_label.csv')

    return result_df

def transfer_date(date) -> list:
    """
    transfer the date into a list which contains numbers.
    """
    date_int = [int(number) for number in date.split("-")]
    return date_int

def date_within_timeperiod(date: str, time_period: tuple) -> bool:
    """
    check if a given date is located within one time period.
    """
    date_check = datetime.datetime.strptime(date, "%Y-%m-%d")
    start = datetime.datetime.strptime(time_period[0][1:-1], "%Y-%m-%d")
    end = datetime.datetime.strptime(time_period[1][2:-1], "%Y-%m-%d")
    if start <= date_check <= end:
        return True
    else:
        return False


def get_date_diff(date1, date2):
    """
    calculate the time difference between two dates.
    """
    y1, m1, d1 = transfer_date(date1)
    d1 = datetime.datetime(y1, m1, d1)
    y2, m2, d2 = transfer_date(date2)
    d2 = datetime.datetime(y2, m2, d2)
    interval = d2-d1
    return interval.days

def get_date_hit(anomalies_list: list, trend_data: list) -> list:
    """
    calculate the anomalies dates within 30 days window ahead of up_trend or down_trend list.
    """
    date_hit = []
    for d1 in anomalies_list:
        for d2 in trend_data:
            if 0 < get_date_diff(d1, d2) <= 30 and d1 not in date_hit:
                date_hit.append(d1)
    return date_hit


def convert_list_to_str(list_: list) -> str:
    """
    combine the number related to a date into a string.
    """
    to_str = ', '.join(e for e in list_)
    return to_str


def filter_date(Case_up, Case_down) -> list:
    """
    method used to filter the follow-up trend.
    """

    i = 0
    while i <len(Case_up)-1:
        date1 = Case_up[i]
        date2 = Case_up[i+1]
        y1, m1, d1 = transfer_date(date1)
        d1 = datetime.datetime(y1, m1, d1)
        y2, m2, d2 = transfer_date(date2)
        d2 = datetime.datetime(y2, m2, d2)
        exist_list = []
        for date3 in Case_down:
            y3, m3, d3 = transfer_date(date3)
            d3 = datetime.datetime(y3, m3, d3)
            if d1 < d3 < d2:
                exist_list.append(d3)
        if exist_list:
            i += 1
            continue
        else:
            Case_up.remove(Case_up[i+1])
    return Case_up

def get_up_down_from_linear_model(linear_df: pd.DataFrame, filter_dates: False) -> tuple:
    """
    this function will get up/down/no trend dates list, and return a tuple.
    """
    trend_list = list(linear_df['up/down trend'])
    date_list = list(linear_df['date'])
    up_trend = []
    down_trend= []
    no_trend = []
    if trend_list[0] == 1:
        up_trend.append(date_list[0])
    for index in range(1, len(trend_list)-1):
        if trend_list[index] == 1 and trend_list[index-1] != 1 and trend_list[index+1] == 1:
            up_trend.append(date_list[index])
        elif trend_list[index] == -1 and trend_list[index-1] != -1 and trend_list[index+1] == -1:
            down_trend.append(date_list[index])
        elif trend_list[index] == 0 and trend_list[index-1] != 0 and trend_list[index+1] == 0:
            no_trend.append(date_list[index])

    # used to filter follow-up trends
    if filter_dates:
        new_up_trend = filter_date(up_trend, down_trend)
        new_down_trend = filter_date(down_trend, new_up_trend)
        return new_up_trend, new_down_trend
    else:
        return up_trend, down_trend

def get_date_hit_sensitivity(proxy_dates: list, gold_standard_dates: list) -> list:
    """
    calculate the anomalies dates within 30 days window ahead of up_trend or down_trend list.
    """

    gold_date_hit = []
    for d1 in gold_standard_dates:
        for d2 in proxy_dates:
            if 0 < get_date_diff(d2, d1) <= 30 and d1 not in gold_date_hit:
                gold_date_hit.append(d1)
                break

    return gold_date_hit


def get_date_hit_precision(proxy_dates: list, gold_standard_dates: list) -> list:
    """
    calculate the anomalies dates within 30 days window ahead of up_trend or down_trend list.
    """

    proxy_date_hit = []
    for d1 in proxy_dates:
        for d2 in gold_standard_dates:
            if 0 < get_date_diff(d1, d2) <= 30 and d1 not in proxy_date_hit:
                proxy_date_hit.append(d1)
                break

    return proxy_date_hit

def get_TP(flag:str, gold_standard: str, proxy_trend_file: str, date_split: str) -> tuple:
    """
    this method is used to calculate the metrics for all cases.
    """
    gs = read_csv(gold_standard)
    proxy = read_csv(proxy_trend_file)
    proxy = proxy.set_index('date')
    proxy = proxy.loc['2020-02-02':, :]

    up_gs, down_gs = get_up_down_from_linear_model(gs, filter_dates=True)
    y1, m1, d1 = transfer_date(up_gs[0])
    d_up = datetime.datetime(y1, m1, d1)

    y2, m2, d2 = transfer_date(down_gs[0])
    d_down = datetime.datetime(y2, m2, d2)
    if d_down <= d_up:
        down_gs.remove(down_gs[0])
    
    proxy['date'] = list(proxy.index)
    up_proxy, down_proxy = get_up_down_from_linear_model(proxy, filter_dates=True)
    y1, m1, d1 = transfer_date(up_proxy[0])
    d_up = datetime.datetime(y1, m1, d1)

    y2, m2, d2 = transfer_date(down_proxy[0])
    d_down = datetime.datetime(y2, m2, d2)
    if d_down <= d_up:
        down_proxy.remove(down_proxy[0])

    # get dates for training (up-trend)
    up_gs = filter_dates_trend_analysis(date_split, up_gs)
    up_proxy = filter_dates_trend_analysis(date_split, up_proxy)

    TP_up_sensitivity = get_date_hit_sensitivity(up_proxy, up_gs)
    TP_up_precision = get_date_hit_precision(up_proxy, up_gs)

    # get dates for training (down-trend)
    down_gs = filter_dates_trend_analysis(date_split, down_gs)
    down_proxy = filter_dates_trend_analysis(date_split, down_proxy)

    TP_down_sensitivity = get_date_hit_sensitivity(down_proxy, down_gs)
    TP_down_precision = get_date_hit_precision(down_proxy, down_gs)

    sensitivity_up = 0
    precision_up = 0
    sensitivity_down = 0
    precision_down = 0

    if len(up_gs) >= len(TP_up_sensitivity) != 0:
        sensitivity_up = len(TP_up_sensitivity) / len(up_gs)
    elif 0 < len(up_gs) < len(TP_up_sensitivity):
        sensitivity_up = 1
    elif not len(up_gs):
        sensitivity_up = 0

    if len(up_proxy) >= len(TP_up_precision) != 0:
        precision_up = len(TP_up_precision) / len(up_proxy)
    elif 0 < len(up_proxy) < len(TP_up_precision):
        precision_up = 1
    elif not len(up_proxy):
        precision_up = 0

    if sensitivity_up + precision_up:
        F1_up = 2 * sensitivity_up * precision_up / (sensitivity_up + precision_up)
    else:
        F1_up = 0

    if len(down_gs) >= len(TP_down_sensitivity) != 0:
        sensitivity_down = len(TP_down_sensitivity) / len(down_gs)
    elif 0 < len(down_gs) < len(TP_down_sensitivity):
        sensitivity_down = 1
    elif not len(down_gs):
        sensitivity_down = 0

    if len(down_proxy) >= len(TP_down_precision) != 0:
        precision_down = len(TP_down_precision) / len(down_proxy)
    elif 0 < len(down_proxy) < len(TP_down_precision):
        precision_down = 1
    elif not len(down_proxy):
        precision_down = 0

    if sensitivity_down + precision_down:
        F1_down = 2 * sensitivity_down * precision_down / (sensitivity_down + precision_down)
    else:
        F1_down = 0

    return sensitivity_up, precision_up, F1_up, sensitivity_down, precision_down, F1_down


def visualization_trend(case_trend:pd.DataFrame, death_trend:pd.DataFrame, hos_trend: pd.DataFrame,
                            case_up_trend_dates: list, case_down_trend_dates: list,
                            death_up_trend_dates: list, death_down_trend_dates: list,
                            hos_up_trend_dates: list, hos_down_trend_dates: list,
                            combined_google_up_dates: list, combined_google_down_dates: list,
                            combined_proxy_up_dates: list, combined_proxy_down_dates: list):#combined_twitter_up_dates: list, combined_twitter_down_dates: list,
    """
    this function is used to plot the up and down trends from linear model.
    """
    case_up_trend_value = []
    for d in case_up_trend_dates:
        case_up_trend_value.extend(case_trend[case_trend["date"]==d]['Case'].tolist())

    case_down_trend_value = []
    for d in case_down_trend_dates:
        case_down_trend_value.extend(case_trend[case_trend["date"]==d]['Case'].tolist())

    death_up_trend_value = []
    for d in death_up_trend_dates:
        death_up_trend_value.extend(death_trend[death_trend["date"]==d]['Death'].tolist())

    death_down_trend_value = []
    for d in death_down_trend_dates:
        death_down_trend_value.extend(death_trend[death_trend["date"]==d]['Death'].tolist())

    hos_up_trend_value = []
    for d in hos_up_trend_dates:
        hos_up_trend_value.extend(hos_trend[hos_trend["date"] == d]['Hosp'].tolist())

    hos_down_trend_value = []
    for d in hos_down_trend_dates:
        hos_down_trend_value.extend(hos_trend[hos_trend["date"] == d]['Hosp'].tolist())

    fig = make_subplots(rows=6, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=case_trend['date'], y=case_trend['Case']), row=1, col=1)
    fig.add_trace(go.Scatter(x=case_up_trend_dates, y=case_up_trend_value, mode="markers", marker=dict(color="red", symbol="205", size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=case_down_trend_dates, y=case_down_trend_value, mode="markers", marker=dict(color="green", symbol="206", size=8)), row=1, col=1)


    fig.add_trace(go.Scatter(x=death_trend['date'], y=death_trend['Death']), row=2, col=1)
    fig.add_trace(go.Scatter(x=death_up_trend_dates, y=death_up_trend_value, mode="markers", marker=dict(color="red", symbol="205", size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=death_down_trend_dates, y=death_down_trend_value, mode="markers",marker=dict(color="green", symbol="206", size=8)), row=2, col=1)

    fig.add_trace(go.Scatter(x=hos_trend['date'], y=hos_trend['Hosp']), row=3,
                  col=1)
    fig.add_trace(go.Scatter(x=hos_up_trend_dates, y=hos_up_trend_value, mode="markers",
                             marker=dict(color="red", symbol="205", size=8)), row=3, col=1)
    fig.add_trace(go.Scatter(x=hos_down_trend_dates, y=hos_down_trend_value, mode="markers",
                             marker=dict(color="green", symbol="206", size=8)), row=3, col=1)

    up_google = len(combined_google_up_dates)
    up_google_y = list(range(up_google))*5
    up_google_count_y = [10]*up_google
    down_google = len(combined_google_down_dates)
    down_google_count_y = [10]*down_google

    fig.add_trace(go.Scatter(x=combined_google_up_dates, y=up_google_count_y, mode="markers",
                             marker=dict(color="red", symbol="205", size=8)), row=4, col=1)
    fig.add_trace(go.Scatter(x=combined_google_down_dates, y=down_google_count_y, mode="markers",
                             marker=dict(color="green", symbol="206", size=8)), row=4, col=1)

    up_combined = len(combined_proxy_up_dates)
    up_combined_y = list(range(up_combined)) * 5
    up_combined_count_y = [10] * up_combined
    down_combined = len(combined_proxy_down_dates)
    down_combined_count_y = [10] * down_combined

    fig.add_trace(go.Scatter(x=combined_proxy_up_dates, y=up_combined_count_y, mode="markers",
                             marker=dict(color="red", symbol="205", size=8)), row=5, col=1)
    fig.add_trace(go.Scatter(x=combined_proxy_down_dates, y=down_combined_count_y, mode="markers",
                             marker=dict(color="green", symbol="206", size=8)), row=5, col=1)

    fig.update_layout(showlegend = True)
    fig.write_image(f"{TREND_VISUALIZATION_FIG}/visualization_trend_and_up_down_events.png", scale=5)
    return

def plot_up_down(df: pd.DataFrame, linear_df: pd.DataFrame, sym: str, STL_period: int, up_trend_dates: list, down_trend_dates: list, linear_window: int, proxy: str, alpha:float = 0.05):
    """
    this function is used to plot the up and down trends from linear model.
    """
    up_trend_value = []
    up_p_value = []
    for d in up_trend_dates:
        up_trend_value.extend(df[df["date"]==d][sym].tolist())
        up_p_value.extend(linear_df[linear_df['date']==d][f'p_value Holm (alpha={alpha})'])
    up_p_value = [elem*10 for elem in up_p_value]

    down_trend_value = []
    down_p_value = []
    for d in down_trend_dates:
        down_trend_value.extend(df[df["date"]==d][sym].tolist())
        down_p_value.extend(linear_df[linear_df['date']==d][f'p_value Holm (alpha={alpha})'])
    down_p_value = [elem*10 for elem in down_p_value]

    fig = px.area(df, x="date", y=sym)
    fig.add_trace(go.Scatter(x=up_trend_dates, y=up_trend_value, mode="markers", name='Up Trends', marker=dict(color="red", symbol="205", size=8)))
    fig.add_trace(go.Scatter(x=down_trend_dates, y=down_trend_value, mode="markers", name='Down Trends', marker=dict(color="green", symbol="206", size=8)))

    fig.update_layout(legend_orientation='h')
    if proxy == "Google_Trends":
        fig.write_image(f"{GOOGLE_LINEAR_MODEL_FIG}/STL_{STL_period}_window{linear_window}_{sym}.png")
    else:
        fig.write_image(f"{TWITTER_LINEAR_MODEL_FIG}/STL_{STL_period}_window{linear_window}_{sym}.png")
    return

def symptom_get_up_down(input_file: str, window: int, sym: str, proxy: str, period:int = 30):
    """
    this function is a complied version of all mini_functions which is used to generate STL decomposition and get up-and down-trends of individual symptom.
    """
    daily_data = f'{PROCESSED_DATA}/{input_file}'
    daily_df = read_csv(daily_data)

    sym_df = daily_df[['date', sym]]
    sym_df.index = sym_df['date']
    
    # perform STL decomposition.
    trend_df = STL_decomposition(sym_df, sym, period)
    trend_df['order'] = list(range(trend_df.shape[0]))
    trend_df['date'] = trend_df.index

    # slice data_folder for performing trend analysis.
    trend_df = trend_df.loc['2020-02-02':, :]
    linear_df = linear_model(trend_df, sym, time_window=window, stride=1, alpha=0.05, proxy=proxy)

    up_dates, down_dates = get_up_down_from_linear_model(linear_df, filter_dates=True)
    d_start = ''
    d_end = ''

    if up_dates:
        y1, m1, d1 = transfer_date(up_dates[0])
        d_start = datetime.datetime(y1, m1, d1)
    if down_dates:
        y2, m2, d2 = transfer_date(down_dates[0])
        d_end = datetime.datetime(y2, m2, d2)
    if d_start and d_end:
        if d_end <= d_start:
            down_dates.remove(down_dates[0])
    return up_dates, down_dates


def get_metrics_from_files(gold_standard_flag: str, proxy: str, symptom_list: list, gold_standard_file: str, date_split: str):

    result = []
    proxy_file = ''
    for sym in tqdm(symptom_list, desc='iterating symptoms for getting the metrics'):
        try:
            if proxy == "Google_Trends":
                proxy_file = f'{GOOGLE_TREND}/Google_Trends_{sym}_trend_label.csv'
            elif proxy == "Twitter":
                proxy_file = f'{TWITTER_TREND}/Twitter_{sym}_trend_label.csv'
            recall_up, precision_up, F1_up, recall_down, precision_down, F1_down = get_TP(flag=gold_standard_flag, gold_standard=gold_standard_file, proxy_trend_file=proxy_file, date_split=date_split)
            result.append([sym, recall_up, precision_up, F1_up, recall_down, precision_down, F1_down])
        except:
            continue
    df = pd.DataFrame(result, columns=['German_symptom', 'Up_recall', 'Up_precision', 'Up_F1_score', 'Down_recall',
                                       'Down_precision', 'Down_F1_score'])
    if proxy == 'Google_Trends':
        df.to_csv(f'{GOOGLE_TREND_METRICS}/{gold_standard_flag}_metrics.csv')
    elif proxy == 'Twitter':
        df.to_csv(f'{TWITTER_TREND_METRICS}/{gold_standard_flag}_metrics.csv')
    return df


def combined(proxy: str, symptoms: list, alpha: float, report_csv: bool, filter_dates: bool) -> tuple:
    """
    This funcition is used to get combined p_value and up/down trend.
    """
    from collections import Counter
    # combine all up/down trends of all symptoms into one dataframe
    if proxy == "Google_Trends":
        df_1 = pd.read_csv(f"{GOOGLE_TREND}/Google_Trends_{symptoms[0]}_trend_label.csv")
    elif proxy == "Twitter":
        df_1 = pd.read_csv(f"{TWITTER_TREND}/Twitter_{symptoms[0]}_trend_label.csv")
    else:
        logger.error('Please check you proxy name!')

    date = df_1['date']
    up_down_list = []
    for sym in symptoms:
        if proxy == "Google_Trends":
            df = pd.read_csv(f'{GOOGLE_TREND}/Google_Trends_{sym}_trend_label.csv')
        elif proxy == "Twitter":
            df = pd.read_csv(f'{TWITTER_TREND}/Twitter_{sym}_trend_label.csv')
        up_down_list.append(df['up/down trend'])

    trend_df = pd.DataFrame(up_down_list).transpose()
    trend_df.columns = symptoms
    trend_df.index = date


    # get ensemble trend per row
    ensemble_trend_list = []
    for i in range(len(trend_df)):
        trends = list(trend_df.iloc[i, :])
        trend_count = Counter(trends)
        trend_dict = dict(trend_count)
        print(trend_dict)
        max_value = max(list(trend_dict.values()))
        if list(trend_dict.values()).count(max_value) != 1:
            ensemble_trend_list.append(0)
        else:
            for key, value in trend_dict.items():
                if value == max_value:
                    ensemble_trend_list.append(key)
    trend_df['ensemble_trend'] = ensemble_trend_list

    # calculate harmonic p_value (Based on the implementation of Kogan et al.)
    harmonic_p = []
    for i in list(trend_df.index):
        p_value_list = []
        if trend_df.loc[i, 'ensemble_trend'] != 0:
            for sym in symptoms:
                try:
                    if trend_df.loc[i, sym] == trend_df.loc[i, 'ensemble_trend']:
                        if proxy == "Google_Trends":
                            sym_df = pd.read_csv(f'{GOOGLE_TREND}/Google_Trends_{sym}_trend_label.csv')
                        elif proxy == "Twitter":
                            sym_df = pd.read_csv(f'{TWITTER_TREND}/Twitter_{sym}_trend_label.csv')
                        p_value_list.append(float(sym_df[sym_df['date'] == i][f'p_value Holm (alpha={alpha})']))
                except:
                    print('you need to generate all trend files first.')
                    logger.error("you need to generate all trend files first.")

            w = len(p_value_list)  # equally weights
            demoninator = sum([1 / f_p for f_p in p_value_list])
            harmonic_p.append(w / demoninator)
        elif trend_df.loc[i, 'ensemble_trend'] == 0:
            harmonic_p.append(100)

    trend_df['harmonic_p'] = harmonic_p
    _, p_val_corrected, _, _ = multipletests(trend_df['harmonic_p'], alpha=alpha, method='holm', returnsorted=False)
    trend_df[f'Holm_pval(alpha={alpha})'] = p_val_corrected

    # get combined up/down trends
    row, column = trend_df.shape
    up_down_trend = []
    for r in range(row):
        if trend_df.iloc[r].loc['ensemble_trend'] == 1 and trend_df.iloc[r].loc[f'Holm_pval(alpha={alpha})'] < alpha:
            up_down_trend.append(1)
        elif trend_df.iloc[r].loc['ensemble_trend'] == -1 and trend_df.iloc[r].loc[f'Holm_pval(alpha={alpha})'] < alpha:
            up_down_trend.append(-1)
        else:
            up_down_trend.append(0)
    trend_df['up/down trend'] = up_down_trend
    trend_df = trend_df.loc["2020-02-02":,:]

    # transfer data_folder to .csv file
    if report_csv:
        if proxy == "Google_Trends":
            output_file = f'{GOOGLE_TREND_METRICS}/Combined_Google_Trends_trend_file.csv'
        elif proxy == "Twitter":
            output_file = f'{TWITTER_TREND_METRICS}/Combined_Twitter_trend_file.csv'
        trend_df.to_csv(output_file)
    trend_df['date'] = trend_df.index

    up, down = get_up_down_from_linear_model(trend_df, filter_dates=filter_dates)

    # ensure up-trends start first.
    y1, m1, d1 = transfer_date(up[0])
    d_up = datetime.datetime(y1, m1, d1)

    y2, m2, d2 = transfer_date(down[0])
    d_down = datetime.datetime(y2, m2, d2)
    if d_down <= d_up:
        down.remove(down[0])
    
    return up, down

def get_combined_p(google_file: str, twitter_file: str, alpha: float, return_up_down_dates: bool, filter_dates: bool) -> tuple:
    """
    Generate combine_p trend file and return up/down events.
    """
    trend_list = [pd.read_csv(google_file)['up/down trend'], pd.read_csv(twitter_file)['up/down trend']]
    google_p = list(pd.read_csv(google_file)['Holm_pval(alpha=0.05)'])

    twitter_p = list(pd.read_csv(twitter_file)['Holm_pval(alpha=0.05)'])
    trend_df = pd.DataFrame(trend_list).transpose()
    trend_df.columns = ['google_trends', 'twitter']
    trend_df.index = pd.read_csv(google_file)['date']

    ensemble_trend_list = []
    for i in range(len(trend_df)):
        trends = list(trend_df.iloc[i, :])
        if trends.count(1) == 2:
            ensemble_trend_list.append(1)
        elif trends.count(-1) == 2:
            ensemble_trend_list.append(-1)
        else:
            ensemble_trend_list.append(0)
    trend_df['ensemble_trend'] = ensemble_trend_list

    # calculate harmonic p_value
    harmonic_p = []
    for i in range(len(trend_df.index)):
        if trend_df.iloc[i].loc['ensemble_trend'] != 0:
            p_value_list = [google_p[i], twitter_p[i]]
            w = 2
            demoninator = sum([1 / f_p for f_p in p_value_list])
            harmonic_p.append(w / demoninator)
        else:
            harmonic_p.append(100)
    trend_df['harmonic_p'] = harmonic_p
    _, p_val_corrected, _, _ = multipletests(trend_df['harmonic_p'], alpha=alpha, method='holm', returnsorted=False)
    trend_df[f'Holm_pval(alpha={alpha})'] = p_val_corrected

    # get combined up/down trends
    row, column = trend_df.shape
    up_down_trend = []
    for r in range(row):
        if trend_df.iloc[r].loc['ensemble_trend'] == 1 and trend_df.iloc[r].loc[f'Holm_pval(alpha={alpha})'] < alpha:
            up_down_trend.append(1)
        elif trend_df.iloc[r].loc['ensemble_trend'] == -1 and trend_df.iloc[r].loc[f'Holm_pval(alpha={alpha})'] < alpha:
            up_down_trend.append(-1)
        else:
            up_down_trend.append(0)
    trend_df['up/down trend'] = up_down_trend
    trend_df['date'] = trend_df.index

    # save the trend file into .csv
    trend_df.to_csv(f'{COMBINED_TREND_METRICS}/Combined_trend.csv')

    # return the up/down events if needed
    if return_up_down_dates:
        up, down = get_up_down_from_linear_model(trend_df, filter_dates=filter_dates)

        # ensure uptrends comes first.
        y1, m1, d1 = transfer_date(up[0])
        d_up = datetime.datetime(y1, m1, d1)

        y2, m2, d2 = transfer_date(down[0])
        d_down = datetime.datetime(y2, m2, d2)
        if d_down <= d_up:
            down.remove(down[0])
        
        return up, down
    else:
        return

def flatten_top_symptom_get_translation(threshold: int, proxy: str, filter_synonym: bool) -> list:
    """
    This function is used to get top sympotoms based on the result of hypergeometics test the the number of co-occurances of docs,
    flatten the symptom lists and get top german symptoms.
    """

    covid_top = pd.read_csv(f"{COVID_KG_DIR}/COVID_sort_pvalue_occurances.csv")
    scaiview_translation_df = pd.read_csv(f"{PROCESSED_DATA}/symptom_translations.csv", index_col='english')
    top_symptoms = list(covid_top['symptom'])[:threshold]
    print(f'Top {threshold} English symptoms:{top_symptoms}')
    google_raw = pd.read_csv(f"{PROCESSED_DATA}/daily_{proxy}_german.csv", index_col='date')
    flatten_list = []

    for sym in top_symptoms:
        if ',' in sym:
            if filter_synonym:
                sum_syn = 0
                flag = ''
                synonyms = sym.split(",")
                for syn in synonyms:
                    if syn in list(scaiview_translation_df.index):
                        syn_german = scaiview_translation_df.loc[syn, 'german']
                        try:
                            
                            if ',' in syn_german:
                                syn_german = syn_german.split(',')
                                german_value = 0
                                flag_german = ''
                                for s in syn_german:
                                    syn_value = google_raw.loc['2020-02-02':'2022-03-01', s].sum()
                                    if syn_value > german_value:
                                        german_value = syn_value
                                        flag_german = s
                                if german_value > sum_syn:
                                    flag = flag_german
                                    sum_syn = german_value
                            else:
                                syn_value = google_raw.loc['2020-02-02':'2022-03-01', syn_german].sum()
                                if syn_value > sum_syn:
                                    flag = syn
                                    sum_syn = syn_value
                        except:
                            continue
                flatten_list.append(flag)
            else:
                flatten_list.extend(sym.split(","))
        else:
            flatten_list.append(sym)
    print(f'Flatten top {threshold}: {flatten_list}')

    scaiview_translation_df = pd.read_csv(f"{PROCESSED_DATA}/symptom_translations.csv", index_col='english')
    english_list = list(scaiview_translation_df.index)
    german_list = list(scaiview_translation_df['german'])
    german_trans = []
    for english_sym in flatten_list:
        if english_sym in english_list:
            if ',' in german_list[english_list.index(english_sym)]:
                syn_sym = []
                flag_sym = []
                for i in german_list[english_list.index(english_sym)].split(','):
                    try:
                        syn_sym.append(google_raw.loc[:, i].sum())
                        flag_sym.append(i)
                    except:
                        continue

                flag_ = flag_sym[syn_sym.index(max(syn_sym))]
                german_trans.append(flag_)
            else:
                german_trans.append(german_list[english_list.index(english_sym)])
    german_top_unique_symptoms = list(set(german_trans))
    print(f'Flatten top {threshold} symptoms with German translation: {german_top_unique_symptoms}')

    return german_top_unique_symptoms

def plot_pairwise(google_dates, combined_dates_lag, gold_standard, trend, year):
    """
    Function used to plot pairwise events.
    """
    fig = go.Figure()
    fig.add_trace(go.Box(x=google_dates, name='Google Trends', orientation='h'), )
    fig.add_trace(go.Box(x=combined_dates_lag, name='Combined', orientation='h'), )

    fig.update_layout(xaxis_range=[-20,15])
    fig.update_layout(
        font_family="Averta",
        hoverlabel_font_family="Averta",
        title_text=trend,
        title_x=0.3,
        xaxis_title_text="Information availabillity (days)",
        xaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_title_font_size=20,
        yaxis_tickfont_size=20,
        hoverlabel_font_size=16,
        height=600,
        width=600,
        showlegend=True,)

    fig.write_image(f'{PAIRWISE_EVENT_FIG}/{year}_pairwise_{trend}_{gold_standard}_google_combined.png', height=480,
                    width=640, scale=6)
    return
