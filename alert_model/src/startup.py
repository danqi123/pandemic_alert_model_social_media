""" Set up file names and folders """

import os
import logging

logger = logging.getLogger(__name__)

# Data file
SYMP_ONTOLOGY = "../data_folder/raw/symp.json"
RKI = "../data_folder/raw/RKI_covid_cases.csv"
RKI_death = "../data_folder/raw/RKI_deaths.csv"
RKI_hospitalization = "../data_folder/raw/RKI_hospitalization.csv"
SCAIVIEW_SYMPTOM = "../data_folder/processed/symptom_translations.csv"

# default directory paths
PROCESSED_DATA = "../data_folder/processed"
RAW_DATA = "../data_folder/raw"
KG_DATA = "../data_folder/Knowledge_graph"
GOLD_STANDARD_DATA = "../data_folder/Gold_standard"
GOOGLE_TRENDS_DATA = "../data_folder/Google_Trends"
TWITTER_DATA = "../data_folder/Twitter"
COMBINED_DATA = "../data_folder/Combined"

# processed data_folder of Google_Trends and Twitter (input data_folder of trend analysis)
GOOGLE_TRENDS_DAILY = "../data_folder/processed/daily_google_german.csv"
TWITTER_DAILY = "../data_folder/processed/daily_twitter_german.csv"

# folder for knowledge graph
COVID_KG_DIR = os.path.join(KG_DATA, "COVID")

# results of log-linear regression model
GOLD_STANDARD_TREND = os.path.join(GOLD_STANDARD_DATA, "Trends")
GOOGLE_TREND = os.path.join(GOOGLE_TRENDS_DATA, "Trends")
TWITTER_TREND = os.path.join(TWITTER_DATA, "Trends")
COMBINED_TREND = os.path.join(COMBINED_DATA, "Trends")

# up-and down-trends of trend analysis
GOLD_STANDARD_EVENT = os.path.join(GOLD_STANDARD_DATA, "Up_and_Down_events")
GOOGLE_EVENT = os.path.join(GOOGLE_TRENDS_DATA, "Up_and_Down_events")
TWITTER_EVENT = os.path.join(TWITTER_DATA, "Up_and_Down_events")
COMBINED_EVENT = os.path.join(COMBINED_DATA, "Up_and_Down_events")

# evaluation metrics folder of trend analysis
GOOGLE_TREND_METRICS = os.path.join(GOOGLE_TRENDS_DATA, "metrics")
TWITTER_TREND_METRICS = os.path.join(TWITTER_DATA, "metrics")
COMBINED_TREND_METRICS = os.path.join(COMBINED_DATA, "metrics")

# folder for trend_forecasting
GOOGLE_FORECASTING_DATA = os.path.join(GOOGLE_TRENDS_DATA, "Trend_forecasting")
COMBINED_FORECASTING_DATA = os.path.join(COMBINED_DATA, "Trend_forecasting")
GOOGLE_LSTM = os.path.join(GOOGLE_FORECASTING_DATA, 'lstm_data')
COMBINED_LSTM = os.path.join(COMBINED_FORECASTING_DATA, 'lstm_data')

# report data_folder folder
LOG_DIR = "../reports/scripts_logs"
REPORT_FIG_DIR = "../reports/figures"
REPORT_DATA_DIR = "../reports/data"

PAIRWISE_EVENT_FIG = os.path.join(REPORT_FIG_DIR, "pairwise_events")
TREND_VISUALIZATION_FIG = os.path.join(REPORT_FIG_DIR, "trend_visualization")

GOOGLE_FORECASTING_REPORT_RF = os.path.join(REPORT_DATA_DIR, "Google_trends", "Trend_forecasting", "Random Forest")
COMBINED_FORECASTING_REPORT_RF = os.path.join(REPORT_DATA_DIR, "Combined", "Trend_forecasting", "Random Forest")

GOOGLE_FORECASTING_REPORT_LSTM = os.path.join(REPORT_DATA_DIR, "Google_trends", "Trend_forecasting", "LSTM")
COMBINED_FORECASTING_REPORT_LSTM = os.path.join(REPORT_DATA_DIR, "Combined", "Trend_forecasting", "LSTM")

# generate dirs
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DATA_DIR, exist_ok=True)
os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(GOLD_STANDARD_EVENT, exist_ok=True)
os.makedirs(GOOGLE_EVENT, exist_ok=True)
os.makedirs(TWITTER_EVENT, exist_ok=True)
os.makedirs(COMBINED_EVENT, exist_ok=True)
os.makedirs(GOLD_STANDARD_TREND, exist_ok=True)
os.makedirs(GOOGLE_TREND, exist_ok=True)
os.makedirs(TWITTER_TREND, exist_ok=True)
os.makedirs(COMBINED_TREND, exist_ok=True)
os.makedirs(GOOGLE_TREND_METRICS, exist_ok=True)
os.makedirs(TWITTER_TREND_METRICS, exist_ok=True)
os.makedirs(COMBINED_TREND_METRICS, exist_ok=True)
os.makedirs(GOOGLE_FORECASTING_DATA, exist_ok=True)
os.makedirs(GOOGLE_LSTM, exist_ok=True)
os.makedirs(COMBINED_FORECASTING_DATA, exist_ok=True)
os.makedirs(COMBINED_LSTM, exist_ok=True)
os.makedirs(TREND_VISUALIZATION_FIG, exist_ok=True)
os.makedirs(PAIRWISE_EVENT_FIG, exist_ok=True)
os.makedirs(COVID_KG_DIR, exist_ok=True)
os.makedirs(GOOGLE_FORECASTING_DATA, exist_ok=True)
os.makedirs(GOOGLE_FORECASTING_REPORT_RF, exist_ok=True)
os.makedirs(COMBINED_FORECASTING_REPORT_RF, exist_ok=True)
os.makedirs(GOOGLE_FORECASTING_REPORT_LSTM, exist_ok=True)
os.makedirs(COMBINED_FORECASTING_REPORT_LSTM, exist_ok=True)

# logging configuration
LOG_FILE_PATH = os.path.join(LOG_DIR, "scripts.log")
logging.basicConfig(filename = LOG_FILE_PATH, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


