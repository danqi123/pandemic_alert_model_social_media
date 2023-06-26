from datetime import date, timedelta
from functools import partial
from time import sleep
from calendar import monthrange

import pandas as pd

from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq


def get_last_date_of_month(year: int, month: int) -> date:
    """Given a year and a month returns an instance of the date class
    containing the last day of the corresponding month.

    Source: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
    """
    return date(year, month, monthrange(year, month)[1])


def convert_dates_to_timeframe(start: date, stop: date) -> str:
    """Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data_folder for a specific time frame
    from Google_Trends.
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """Attempts to fecth data_folder and retries in case of a ResponseError."""
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    return pytrends.interest_over_time()


def get_daily_data(word: str,
                 start_year: int,
                 start_mon: int,
                 stop_year: int,
                 stop_mon: int,
                 language: str="de-DE",
                 geo: str = 'DE',
                 verbose: bool = True,
                 wait_time: float = 5.0) -> pd.DataFrame:
    """Given a word, fetches daily search volume data_folder from Google_Trends and
    returns results in a pandas DataFrame.

    Details: Due to the way Google_Trends scales and returns data_folder, special
    care needs to be taken to make the daily data_folder comparable over different
    months. To do that, we download daily data_folder on a month by month basis,
    and also monthly data_folder. The monthly data_folder is downloaded in one go, so that
    the monthly values are comparable amongst themselves and can be used to
    scale the daily data_folder. The daily data_folder is scaled by multiplying the daily
    value by the monthly search volume divided by 100.
    For a more detailed explanation see http://bit.ly/trendsscaling

    Args:
        word (str): Word to fetch daily data_folder for.
        start_year (int): the start year
        start_mon (int): start 1st day of the month
        stop_year (int): the end year
        stop_mon (int): end at the last day of the month
        geo (str): geolocation
        verbose (bool): If True, then prints the word and current time frame
            we are fecthing the data_folder for.

    Returns:
        complete (pd.DataFrame): Contains 4 columns.
            The column named after the word argument contains the daily search
            volume already scaled and comparable through time.
            The column f'{word}_unscaled' is the original daily data_folder fetched
            month by month, and it is not comparable across different months
            (but is comparable within a month).
            The column f'{word}_monthly' contains the original monthly data_folder
            fetched at once. The values in this column have been backfilled
            so that there are no NaN present.
            The column 'scale' contains the scale used to obtain the scaled
            daily data_folder.
    """

    # Set up start and stop dates
    start_date = date(start_year, start_mon, 1)
    stop_date = get_last_date_of_month(stop_year, stop_mon)

    # Start pytrends for region, germany or france.
    pytrends = TrendReq(hl=language, tz=60)
    # Initialize build_payload with the word we need data_folder for
    build_payload = partial(pytrends.build_payload,
                            kw_list=[word], cat=0, geo=geo, gprop='')

    # Obtain monthly data_folder for all months in years [start_year, stop_year]
    monthly = _fetch_data(pytrends, build_payload,
                         convert_dates_to_timeframe(start_date, stop_date))

    # Get daily data_folder, month by month
    results = {}
    # if a timeout or too many requests error occur we need to adjust wait time
    current = start_date
    while current < stop_date:
        last_date_of_month = get_last_date_of_month(current.year, current.month)
        timeframe = convert_dates_to_timeframe(current, last_date_of_month)
        if verbose:
            print(f'{word}:{timeframe}')
        results[current] = _fetch_data(pytrends, build_payload, timeframe)
        current = last_date_of_month + timedelta(days=1)
        sleep(wait_time)  # don't go too fast or Google will send 429s

    daily = pd.concat(results.values()).drop(columns=['isPartial'])
    complete = daily.join(monthly, lsuffix='_unscaled', rsuffix='_monthly')

    # Scale daily data_folder by monthly weights so the data_folder is comparable
    complete[f'{word}_monthly'].ffill(inplace=True)  # fill NaN values
    complete['scale'] = complete[f'{word}_monthly'] / 100
    complete[word] = complete[f'{word}_unscaled'] * complete.scale

    return complete
