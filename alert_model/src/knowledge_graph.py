"""CLI module for generating symptom corpus."""

import click
import logging
import json
import pandas as pd
from startup import SYMP_ONTOLOGY, COVID_KG_DIR, SCAIVIEW_SYMPTOM, REPORT_FIG_DIR
from disease_network import get_symptoms_synonyms, get_request, get_count_request, hypergeometric, get_top_symptoms_, get_sym_syn_list
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@click.group(help = f'The Command Line Utilities of performing knowledge graph for the identification of COVID-19 related symptoms.')
def KG():
	"""Entry method."""
	pass

@KG.command(name = "get_symptom_disease_IDs")
@click.argument('disease_flag')
def get_symptom_disease_IDs(disease_flag: str) -> dict:

    """
    This function is used to request the SCAIView knowledge software with the input of disease name AND symptom terms. The output is the Unique IDs of iterature documents which are related to the disease and symptoms.
    Parameters
    ----------
    disease_flag: str
    Returns
    -------
    A dictionary: key is the symptom list, value is the IDs of documents in SCAIView.
    """
    assert disease_flag == 'COVID'
    
    symptoms = get_symptoms_synonyms(SYMP_ONTOLOGY)
    symptom_list = list(symptoms.values())
    
    ID_dict = {}
    for sym_synonym in tqdm(symptom_list, desc="Iterating through symptoms"):
        key = ','.join(sym_synonym)
        ID_dict[key] = get_request(sym_synonym, ["COVID", "Coronavirus disease", "COVID 19"])

    with open(f'{COVID_KG_DIR}/COVID_sym_IDs.json', 'w') as fp:
       json.dump(ID_dict, fp)
    
    return ID_dict
        
    
@KG.command(name='get_disease_count')
def get_disease_count() -> None:
    """get the number of documents related to "COVID-19".
    """
    disease_count_dict = {}
    count = get_count_request(["COVID", "Coronavirus disease", "COVID 19"])
    disease_count_dict["COVID"] = count
    with open(f'{COVID_KG_DIR}/disease_count.json', 'w') as fp:
        json.dump(disease_count_dict, fp)
    return


@KG.command(name='get_symptoms_count')
def get_symptoms_count() -> None:
    """
	This function is used to get the count of specific terms in SCAIView, and save the .csv and .json file.
	"""
    symptoms = get_symptoms_synonyms(SYMP_ONTOLOGY)
    symptom_list = list(symptoms.values())
    symptoms_count_dict = {}
    sym_str = ''

    for sym in tqdm(symptom_list, desc='Iterating symptoms'):
        print(f'{sym} is searched...')
        if len(sym) > 1:
            sym_str = ','.join(sym)
        elif len(sym) == 1:
            sym_str = sym[0]
        count = get_count_request(sym)
        if count:
            symptoms_count_dict[sym_str] = count
            print(f'{sym_str}:{count}')
            logger.info(f'The number of {sym_str} related documents is counted.')
        else:
            logger.warning(f'{sym_str}: No documents found!')

    symptoms_count_df = pd.DataFrame.from_dict(symptoms_count_dict, orient='index', columns=['Count'])
    symptoms_count_df.to_csv(f'{COVID_KG_DIR}/symptoms_count.csv')

    with open(f'{COVID_KG_DIR}/symptoms_count.json', 'w') as fp:
        json.dump(symptoms_count_dict, fp)


@KG.command(name='get_covid_symptoms')
def get_covid_symptoms():
    symptom_lists = get_sym_syn_list(SCAIVIEW_SYMPTOM, "german")
    with open(f'{COVID_KG_DIR}/COVID_symptoms_from_hypergeometrictest.json', 'w') as fp:
        json.dump(symptom_lists, fp)

@KG.command(name='perform_disease_hypergeo_test')
@click.argument('disease')
@click.argument('alpha')
@click.option('-v', '--verbose', default=False, is_flag=True, help="When used, will save the json file.")
def perform_disease_hypergeo_test(disease: str, verbose: bool, alpha: str) -> dict:

    """This function is used to perform hypergeometric test.
    Args:
    disease (str): COVID
    verbose (bool): True
    print_top_symptoms (bool): True (print top 50 symptoms.)
    alpha (str): 0.05

    Returns:
    dict: The dictionary of the result of hypergeometric test (sorted based on the p_values.)
    """
    disease_symptom_pair_file = open(f'{COVID_KG_DIR}/COVID_sym_IDs.json')
    symptoms_dict = json.load(disease_symptom_pair_file)
    symptoms = list(symptoms_dict.keys())
    disease_p_value_dict = {}
    for sym in symptoms:
        if not symptoms_dict.get(sym):
            del symptoms_dict[sym]
        else:
            disease_symptoms_docs = len(symptoms_dict[sym])
            try:
                  p_value = hypergeometric(disease, sym, disease_symptoms_docs)
                  disease_p_value_dict[sym] = p_value
            except:
                  continue
    
    p_value_list = list(disease_p_value_dict.values())
    _, p_val_corrected, _, _ = multipletests(p_value_list, alpha=float(alpha), method='holm', returnsorted=False)
    multiple_p_dict = dict(zip(list(disease_p_value_dict.keys()), p_val_corrected))
    
    for ele in multiple_p_dict.copy():
        if multiple_p_dict[ele] > float(alpha):
            del multiple_p_dict[ele]
    
    sort_dict = dict(sorted(multiple_p_dict.items(), key=lambda item: item[1]))
    
    if verbose:
        with open(f'{COVID_KG_DIR}/COVID_hypergeometric_{alpha}.json', 'w') as fp:
            json.dump(sort_dict, fp)
    
    return sort_dict

@KG.command(name='get_top_relevant_symptoms')
@click.argument("alpha")
@click.argument("threshold")
def get_top_relevant_symptoms(alpha: str, threshold: str) -> None:
    """
    get the top symptoms with lowest p_values and highest co-occurance, and save in .csv file.
    """
    top_sym = get_top_symptoms_(float(alpha), int(threshold))
    print(top_sym)

@KG.command(name='show_plot')
@click.argument('file')
@click.argument('threshold')
def show_plot(file: str, threshold: str) -> None:
    """
    plot of the top symptom list into a graph with descending co-occurrences with COVID-19.
    """
    df = pd.read_csv(f'{COVID_KG_DIR}/{file}')
    df = df.iloc[:int(threshold), :]
    
    # The x-axis list is given by the current top symptom list.
    x_axis = ['pneumonia', 'fever', 'cough', 'inflammation', 'dyspnea', 'respiratory failure', 'discharge', 'fatigue', 'diarrhea', 'anxiety', 'shock', 'headache', 'muscle pain', 'lymphopenia', 'vomiting', 'dry cough', 'sputum', 'throat pain', 'nausea', 'hypoxemia', 'severe pneumonia', 'septic shock', 'coagulopathy', 'confusion', 'stuffy nose']
    
    fig = px.bar(df, x = x_axis, y = 'co-occurances', color='co-occurances', labels=dict(x="Symptoms", y="Co-occurances with COVID-19 related terms in PubMed/PMC"),)
    fig.add_trace(go.Scatter(x = x_axis, y = df['p_value'], mode='markers', name='adjusted P value', showlegend=False,))
    fig.update_layout(width=5000, height=500, bargap=0.3, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.01), xaxis_tickangle=-45)
    fig.write_image(f'{REPORT_FIG_DIR}/co-occurances-symptoms-covid.png', height=480, width=1200, scale=4)
    


if __name__ == "__main__":
    KG()
