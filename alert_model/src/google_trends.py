"""Module used to get google trends Data."""

from google_trends_daily import get_daily_data
import pandas as pd
from startup import PROCESSED_DATA, COVID_KG_DIR
import logging
import time
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_csv(file:str) -> pd.DataFrame:
    """read csv file"""
    g_trends = pd.read_csv(file, encoding='utf_8_sig')
    return g_trends

def get_sym_syn_list(file: str, flag:str) -> list:
    """ get symptom or synonym list."""
    csv_dataframe = pd.read_csv(file, encoding='utf_8_sig')
    synonyms_list = []
    for synonyms in csv_dataframe[flag]:
        try:
            synonyms_list.extend(synonyms.split(","))
        except:
            synonyms_list.append(synonyms)
    sym_syn_list = list(set(synonyms_list))
    return sym_syn_list

def get_google_trends_daily(key_word_list: list,
                            start_year: int,
                            start_month: int,
                            end_year: int,
                            end_month: int,
                            geo: str,
                            start: int,
                            end: int) -> None:

    """Generate DAILY Google_Trends data."""
    frames = []
    for keyword in tqdm(key_word_list):
        try:
            df = get_daily_data(keyword, start_year, start_month, end_year, end_month, geo=geo, verbose=False, wait_time=5)
            df = df.iloc[:, [4]].fillna(0)
            frames.append(df)
            logger.info(f"{keyword} DAILY Google_Trends is stored.")
        except:
            logger.error(f"No entry for {keyword} in google trends.")
        time.sleep(60)

    result = pd.concat(frames, axis=1)

    sym_file_path = f"{PROCESSED_DATA}/daily_google_german_{start}_{end}_{geo}.csv"
    result.to_csv(sym_file_path, encoding='utf_8_sig')

    logger.info(f"{geo}_{key_word_list} DAILY google trends Data is generated.")
    return result

def compile_google_trends(file: str, start_year: int, start_month: int, end_year: int, end_month: int, geo: str):

    f = open(f'{COVID_KG_DIR}/{file}')
    data = json.load(f)
    get_google_trends_daily(data, start_year, start_month, end_year, end_month, geo)


if __name__ == "__main__":
    file_name = 'COVID_symptoms_from_hypergeometrictest.json'
    # Note that it is better to slice the synonym list and let Google API retrieve the data one by one.
    compile_google_trends(file_name, 2020, 2, 2022, 6, "DE")
    

