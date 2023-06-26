"""CLI trend analysis module"""

import click
import logging
import pandas as pd
import datetime
import json
import os
from tqdm import tqdm
import numpy as np
from statsmodels.tsa.seasonal import STL
from startup import RKI, RKI_death, RKI_hospitalization, PROCESSED_DATA, GOOGLE_TRENDS_DAILY, TWITTER_DAILY, GOOGLE_TREND_METRICS, TWITTER_TREND_METRICS, COMBINED_TREND_METRICS, GOLD_STANDARD_TREND, GOOGLE_EVENT, TWITTER_EVENT, COMBINED_EVENT, \
	GOLD_STANDARD_EVENT, GOOGLE_TREND, TWITTER_TREND, GOOGLE_TRENDS_DATA, TWITTER_DATA
from log_linear_regression import read_csv, linear_model, visualization_trend, filter_date, STL_decomposition, \
	symptom_get_up_down, get_metrics_from_files, get_up_down_from_linear_model, flatten_top_symptom_get_translation, \
	combined, get_TP, get_combined_p, plot_pairwise
from date import filter_dates, get_date_interval, transfer_date, filter_dates_trend_analysis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@click.group(help = f'The Command Line Utilities of implementing trend analysis.')
def trend_analysis():
	"""Entry method."""
	pass

# get trend from gold standard and store trend from log linear regression model
@trend_analysis.command(name = "generate_gold_standard_trend")
@click.argument('flag')
@click.argument('window')
@click.argument('stl_number')
def generate_gold_standard_trend(flag: str, window: str, stl_number: str) -> pd.DataFrame:

	"""
	Generate trend analysis on gold standards and get onset dates of up- and down-trends.
	"""

	# Normalize the data
	if flag == "RKI_case":
		df = read_csv(RKI)
		df.index = df.pop('date')
		df["normalized cases"] = round(df["Case"] / 830, 10)
	elif flag == "RKI_death":
		df = read_csv(RKI_death)
		df.index = df.pop('date')
		df["normalized cases"] = round(df["Death"] / 830, 10)
	elif flag == "RKI_hospitalization":
		df = read_csv(RKI_hospitalization)
		df.index = df.pop('date')
		df["normalized cases"] = round(df["Hosp"] / 830, 10)
	order_list = []
	for i, d in enumerate(df.index, 1):
		order_list.append(i)
	df['order'] = order_list
	df['date'] = df.index

	# perform STL on RKI data
	stl = STL(df["normalized cases"], period=int(stl_number), robust=True)
	result = stl.fit()
	trend = result.trend.values.tolist()
	df[flag] = trend

	# call log linear regression model function and return the result of a dataframe
	linear_df = linear_model(df, flag, int(window), 1, 0.05, proxy='gold_standard')
	logger.info(f'Trend file of {flag} is stored.')
	return linear_df

@trend_analysis.command(name = "get_trends_from_gold_standard")
@click.argument('gold_standard')
@click.option('-f', '--filter_dates', default=False, is_flag=True, help="When used, will filter the following-up events")
def get_trends_from_gold_standard(gold_standard: str, filter_dates: bool) -> None:
	"""
    Get up and down events from a input trend file.
	"""
	df = read_csv(f'{GOLD_STANDARD_TREND}/{gold_standard}_trend_label.csv')
	up, down = get_up_down_from_linear_model(df, filter_dates=filter_dates)
	y1, m1, d1 = transfer_date(up[0])
	d_up = datetime.datetime(y1, m1, d1)

	y2, m2, d2 = transfer_date(down[0])
	d_down = datetime.datetime(y2, m2, d2)
	if d_down <= d_up:
		down.remove(down[0])
	result_dict = {'Up_trends': up, 'Down_trends': down}
	output_path = f'{GOLD_STANDARD_EVENT}/{gold_standard}_up_and_down.json'
	with open(output_path, 'w') as f:
		json.dump(result_dict, f)
		logger.info(f'{gold_standard}_up_and_down_events have been retrieved and saved in folder {output_path}.')
	print(up, down)


@trend_analysis.command(name = "generate_proxy_trend")
@click.argument('proxy')
@click.argument('input_file')
@click.argument('window')
@click.argument('period')
def generate_proxy_trend(proxy: str, input_file: str, window: str, period: str) -> None:
	"""
	Get up- and down-trend of individual symptom.
	"""
	df = read_csv(f'{PROCESSED_DATA}/{input_file}')
	
	# get symptom list.
	german_symptoms = list(df.columns)[1:]
	for sym in tqdm(german_symptoms, desc='iterating symptoms...'):
		symptom_get_up_down(input_file=input_file, window=int(window), sym=sym, period=int(period), proxy=proxy)
	logger.info(f'{proxy}_trend_files are stored.')
	return


@trend_analysis.command(name = "generate_evaluation_metrics")
@click.argument('gold_standard')
@click.argument('proxy')
@click.argument('split_date')
def generate_evaluation_metrics(gold_standard: str, proxy: str, split_date: str) -> pd.DataFrame:
	"""
	Perform evaluation metrics for all symptoms from Google_Trends OR Twitter.
	"""
	df = pd.DataFrame()
	if proxy == "Google_Trends":
		df = read_csv(GOOGLE_TRENDS_DAILY)
	elif proxy == "Twitter":
		df = read_csv(TWITTER_DAILY)
	german_symptoms = list(df.columns)[1:]
	df = get_metrics_from_files(gold_standard, proxy, german_symptoms, f'{GOLD_STANDARD_TREND}/{gold_standard}_trend_label.csv', split_date)
	logger.info(f'{gold_standard} ----{proxy} evaluation metrics file is stored.')
	return df

@trend_analysis.command(name = "get_symptoms")
@click.argument('scaiview_threshold')
@click.argument('proxy')
@click.option('-f', '--filter_synonym', default=False, is_flag=True, help="When used, will filter the synonyms.")
def get_symptoms(scaiview_threshold: str, proxy: str, filter_synonym: bool) -> list:
	"""when used, will get top symptoms from hypergeometric test.

	Args:
		scaiview_threshold (str): the threshold of top symptoms (hypergeometric test)
		filter_synonym (bool): check if needed to filter synonyms out.

	Returns:
		list: the final symptom list for trend analysis and trend forecasting.
	"""
	symptom_list = flatten_top_symptom_get_translation(int(scaiview_threshold), proxy, filter_synonym)
	final_symptom_list = []
	for s in symptom_list:
		if proxy == 'google':
			path = f'{GOOGLE_TREND}/Google_Trends_{s}_trend_label.csv'
		elif proxy == 'twitter':
			path = f'{TWITTER_TREND}/Twitter_{s}_trend_label.csv'
		if os.path.isfile(path):
			final_symptom_list.append(s)
	symptom_dict = {'symptom_synonyms': final_symptom_list}

	output_path = ''
	if proxy == 'google':
		output_path = f'{GOOGLE_TRENDS_DATA}/{proxy}_top_{scaiview_threshold}_symptom_and_synonyms.json'
	elif proxy == 'twitter':
		output_path = f'{TWITTER_DATA}/{proxy}_top_{scaiview_threshold}_symptom_and_synonyms.json'
	with open(output_path, 'w') as f:
		json.dump(symptom_dict, f)
	return symptom_list

@trend_analysis.command(name = "combined_proxy")
@click.argument('scaiview_threshold')
@click.argument('proxy')
@click.argument('alpha')
@click.argument('date_split')
@click.option('-r', '--report_csv', default=False, is_flag=True, help="When used, will save the trend into .csv file.")
@click.option('-f', '--filter_dates', default=False, is_flag=True, help="When used, will filter the following-up events")
def combined_proxy(scaiview_threshold: str, proxy: str, alpha: float, date_split: str, report_csv: bool, filter_dates: bool) -> tuple:
	"""
	Generate the symptom list which is used for trend analysis (calculate the evaluation metrics)
	e.g. top 20 synonyms from SCAIView based hypergeometric test.
	"""
	if proxy == "Google_Trends":
		f = open(f'{GOOGLE_TRENDS_DATA}/google_top_{scaiview_threshold}_symptom_and_synonyms.json')
	elif proxy == "Twitter":
		f = open(f'{TWITTER_DATA}/twitter_top_{scaiview_threshold}_symptom_and_synonyms.json')
	scaiview_top_list = json.load(f)['symptom_synonyms']

	up, down = combined(proxy=proxy, symptoms=scaiview_top_list, alpha=float(alpha), report_csv=report_csv, filter_dates=filter_dates)
	y1, m1, d1 = transfer_date(up[0])
	d_up = datetime.datetime(y1, m1, d1)

	y2, m2, d2 = transfer_date(down[0])
	d_down = datetime.datetime(y2, m2, d2)
	if d_down <= d_up:
		down.remove(down[0])

	# get dates for training 
	up = filter_dates_trend_analysis(date_split, up)
	down = filter_dates_trend_analysis(date_split, down)
	result_dict = {'Up_trends': up, 'Down_trends': down}

	print(f'up trends:{up}')
	print(f'down trends:{down}')
	if proxy == 'Google_Trends':
		output_path = f'{GOOGLE_EVENT}/{proxy}_up_and_down.json'
	elif proxy == 'Twitter':
		output_path = f'{TWITTER_EVENT}/{proxy}_up_and_down.json'
	else:
		raise IndexError('Please given correct proxy names: Google_Trends or Twitter.')
	with open(output_path, 'w') as f:
		json.dump(result_dict, f)
		logger.info(f'{proxy} up and downtrends events are stored in {output_path}.')
	return up, down




@trend_analysis.command(name = "get_combined_P_trends")
@click.argument('alpha')
@click.argument('date_split')
@click.option('-f', '--filter_dates', default=False, is_flag=True, help="When used, will filter the following-up events")
def get_combined_P_trends(alpha: str, date_split: str, filter_dates: bool) -> tuple:
	"""
	With the information of Google Trends and Twitter selected symptoms and their slope/trend information, generate the Combined P proxy.
	Parameters
	----------
	alpha: str

	Returns: tuple
	-------
	up and down json file.
	and return the up and down trend dates in STDOUT.
	"""
	try:
		google_combined_file = f'{GOOGLE_TREND_METRICS}/Combined_Google_Trends_trend_file.csv'
		twitter_combined_file = f'{TWITTER_TREND_METRICS}/Combined_Twitter_trend_file.csv'
		up, down = get_combined_p(google_combined_file, twitter_combined_file, alpha=float(alpha), return_up_down_dates=True, filter_dates=filter_dates)
		y1, m1, d1 = transfer_date(up[0])
		d_up = datetime.datetime(y1, m1, d1)

		y2, m2, d2 = transfer_date(down[0])
		d_down = datetime.datetime(y2, m2, d2)
		if d_down <= d_up:
			down.remove(down[0])
		
		# get dates for training
		up = filter_dates_trend_analysis(date_split, up)
		down = filter_dates_trend_analysis(date_split, down)
		
		print(f'Up_trends: {up}')
		print(f'Down_trends: {down}')
	
		result_dict = {'Up_trends': up, 'Down_trends': down}
		output_path = f'{COMBINED_EVENT}/Combined_up_and_down.json'
		with open(output_path, 'w') as f:
			json.dump(result_dict, f)
			logger.info(f'Combined up and downtrends events are stored in {output_path}.')
		return up, down
	except:
		raise IndexError('cannot generate combined trend.')
	
@trend_analysis.command(name = "generate_metrics_for_combined_proxy_or_combinedP")
@click.argument('proxy')
@click.argument('gold_standard')
@click.argument('date_split')
def generate_metrics_for_combined_proxy_or_combinedP(proxy: str, gold_standard: str, date_split: str) -> None:
	"""
	This function is used to get metrics for combined_proxy of combined_p with respect to certain gold standard file.
	"""
	if proxy == "Google_Trends":
		folder = GOOGLE_TREND_METRICS
		proxy_file = f'{folder}/Combined_Google_Trends_trend_file.csv'
	elif proxy == "Twitter":
		folder = TWITTER_TREND_METRICS
		proxy_file = f'{folder}/Combined_Twitter_trend_file.csv'
	elif proxy == "Combined":
		folder = COMBINED_TREND_METRICS
		proxy_file = f'{folder}/Combined_trend.csv'
	metric_dict = {}
	recall_up, precision_up, F1_up, recall_down, precision_down, F1_down = get_TP(flag=gold_standard, gold_standard=f'{GOLD_STANDARD_TREND}/{gold_standard}_trend_label.csv', proxy_trend_file=proxy_file, date_split=date_split)
	metric_dict['up_events_sensitivity'] = recall_up
	metric_dict['up_events_precision'] = precision_up
	metric_dict['up_events_F1'] = F1_up
	metric_dict['down_events_sensitivity'] = recall_down
	metric_dict['down_events_precision'] = precision_down
	metric_dict['down_events_F1'] = F1_down
	df_report = pd.DataFrame(metric_dict, index=['value'])
	df_report.to_csv(f'{folder}/{proxy}_{gold_standard}_combined_symptoms_metrics.csv')
	
	print('----------------------------------------------------------------------------------------------------')
	print(f'Recall_up_events:{recall_up}, Precision_up_events:{precision_up}, F1 Score_up_events:{F1_up}')
	print(f'Recall_down_events:{recall_down}, Precision_down_events:{precision_down}, F1 Score_down_events:{F1_down}')

@trend_analysis.command(name = "visualize_trend")
def visualize_trend() -> None:
	"""
	Make a visualization plot of gold standard and digital trace up- and down-trends with their onsets.
	"""
	# read gold standard data
	RKI_case_data = read_csv(RKI)
	RKI_death_data = read_csv(RKI_death)
	RKI_hospitalization_data = read_csv(RKI_hospitalization)

	# use STL to get the trend data from RKI case and RKI death.
	case_trend = STL_decomposition(RKI_case_data, "Case", 7)

	# get the trainig period (slicing data)
	case_trend = case_trend.iloc[:-105, :] 
	death_trend = STL_decomposition(RKI_death_data, "Death", 7)
	death_trend = death_trend.iloc[:-105, :] 
	hosp_trend = STL_decomposition(RKI_hospitalization_data, "Hosp", 7)
	hosp_trend = hosp_trend.iloc[:-105, :] 

	# filter the follow-up trend events.
	case_file = open(f'{GOLD_STANDARD_EVENT}/RKI_case_up_and_down.json')
	case_dict = json.load(case_file)
	RKI_Case_Up_trend_list = case_dict["Up_trends"]
	RKI_Case_Down_trend_list = case_dict["Down_trends"]

	# get training dates for cases
	RKI_Case_Up_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Case_Up_trend_list)
	RKI_Case_Down_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Case_Down_trend_list)

	case_up_trend_date = filter_date(RKI_Case_Up_trend_list, RKI_Case_Down_trend_list)
	case_down_trend_date = filter_date(RKI_Case_Down_trend_list, case_up_trend_date)
    
	death_file = open(f'{GOLD_STANDARD_EVENT}/RKI_death_up_and_down.json')
	death_dict = json.load(death_file)
	RKI_Death_Up_trend_list = death_dict["Up_trends"]
	RKI_Death_Down_trend_list = death_dict["Down_trends"]

	# get training dates for deaths
	RKI_Death_Up_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Death_Up_trend_list)
	RKI_Death_Down_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Death_Down_trend_list)

	death_up_trend_date = filter_date(RKI_Death_Up_trend_list, RKI_Death_Down_trend_list)
	death_down_trend_date = filter_date(RKI_Death_Down_trend_list, death_up_trend_date)
    
	hos_file = open(f'{GOLD_STANDARD_EVENT}/RKI_hospitalization_up_and_down.json')
	hos_dict = json.load(hos_file)
	RKI_Hos_Up_trend_list = hos_dict["Up_trends"]
	RKI_Hos_Down_trend_list = hos_dict["Down_trends"]

	# get training dates for hospitalization
	RKI_Hos_Up_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Hos_Up_trend_list)
	RKI_Hos_Down_trend_list = filter_dates_trend_analysis("2022-03-01", RKI_Hos_Down_trend_list)
	
	hos_up_trend_date = filter_date(RKI_Hos_Up_trend_list, RKI_Hos_Down_trend_list)
	hos_down_trend_date = filter_date(RKI_Hos_Down_trend_list, hos_up_trend_date)
    
	google_file = open(f'{GOOGLE_EVENT}/Google_Trends_up_and_down.json')
	google_dict = json.load(google_file)
	google_up_trends = google_dict["Up_trends"]
	google_down_trends = google_dict["Down_trends"]
	google_up_trend_date = filter_date(google_up_trends, google_down_trends)
	google_down_trend_date = filter_date(google_down_trends, google_up_trend_date)

	combined_file = open(f'{COMBINED_EVENT}/Combined_up_and_down.json')
	combined_dict = json.load(combined_file)
	combined_up_trends = combined_dict["Up_trends"]
	combined_down_trends = combined_dict["Down_trends"]
	combined_up_trend_date = filter_date(combined_up_trends, combined_down_trends)
	combined_down_trend_date = filter_date(combined_down_trends, combined_up_trend_date)

	# call visualization function.
	visualization_trend(case_trend= case_trend, death_trend= death_trend, hos_trend=hosp_trend,
						case_up_trend_dates=case_up_trend_date, case_down_trend_dates=case_down_trend_date,
						death_up_trend_dates=death_up_trend_date, death_down_trend_dates=death_down_trend_date,
						hos_up_trend_dates=hos_up_trend_date, hos_down_trend_dates=hos_down_trend_date,
						combined_google_up_dates=google_up_trend_date, combined_google_down_dates=google_down_trend_date,
						combined_proxy_up_dates=combined_up_trend_date, combined_proxy_down_dates=combined_down_trend_date)#combined_twitter_up_dates=twitter_up_trend_date, combined_twitter_down_dates=twitter_down_trend_date,
	logger.info('Trend visualization file is stored.')

@trend_analysis.command(name = "plot_pairwise_trend_event")
@click.argument('gold_standard')
@click.argument('trend')
@click.argument('flag')
@click.argument('date_split')
def plot_pairwise_trend_event(gold_standard: str, trend: str, flag: str, date_split: str) -> None:
	"""pairwise event visualization and get time lags between gold standard and Google Trends/Combined trace.

	Args:
		gold_standard (str): RKI_case/RKI_death/RKI_hospitalization
		trend (str): uptrend or downtrend
		flag (str): the time period: 2020_2022

	"""
	# get up/down trend list for gold standard.
	gold_standard_trends_dates_file = f'{GOLD_STANDARD_EVENT}/{gold_standard}_up_and_down.json'
	f1 = open(gold_standard_trends_dates_file)
	gold_trends = json.load(f1)
	gold_trend_date = gold_trends[trend]

	# filter out dates for test
	gold_trend_date = filter_dates_trend_analysis(date_split, gold_trend_date)

	google_trends_dates_file = f'{GOOGLE_EVENT}/Google_Trends_up_and_down.json'
	f2 = open(google_trends_dates_file)
	google_trends = json.load(f2)
	google_trend_date = google_trends[trend]

	if flag == '2020_2022':
		start = '2020-02-02'
		end = '2022-06-15'
	else:
		raise IndexError("check correct time range.")

	# filter out dates for test
	google_trend_date = filter_dates_trend_analysis(date_split, google_trend_date)

	gold_standard_filter_dates = filter_dates(start, end, gold_trend_date)

	google_filter_dates = filter_dates(start, end, google_trend_date)

	google_gold_standard = get_date_interval(gold_standard_dates_list = gold_standard_filter_dates,
											 proxy_dates_list = google_filter_dates)
	print(google_gold_standard)
	print(f'google_{trend}_{gold_standard}:', np.median(google_gold_standard))


	combined_dates_file = f'{COMBINED_EVENT}/Combined_up_and_down.json'
	f3 = open(combined_dates_file)
	combined = json.load(f3)
	combined_date = combined[trend]
    
	combined_date = filter_dates_trend_analysis(date_split, combined_date)
	combined_filter_dates = filter_dates(start, end, combined_date)

	combined_gold_standard = get_date_interval(gold_standard_dates_list=gold_standard_filter_dates,
											 proxy_dates_list=combined_filter_dates)
	print(combined_gold_standard)
	print(f'combined_{trend}_{gold_standard}:', np.median(combined_gold_standard))

	plot_pairwise(google_gold_standard, combined_gold_standard, gold_standard, trend, flag)#twitter_gold_standard, combined_gold_standard, 
	return


if __name__ == "__main__":
	trend_analysis()
