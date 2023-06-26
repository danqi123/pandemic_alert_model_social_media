"""Module used to get lists of symptoms and diseases."""
# ["COVID", "Coronavirus disease", "COVID 19"]


import requests
import logging
import json
import pandas as pd
from typing import Dict, Union, List
from scipy.stats import hypergeom
from startup import COVID_KG_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_symptoms_synonyms(file: str)-> dict:
    """
    input: json file from symptom ontology
    output: A dictionary with the symptom ID as the key, and symptom with its synonyms are value.

    Parameters
    ----------
    file: str
    json file of symptom ontology.

    Returns
    -------
    symptom and synonym dictionary: dict

    """
    f = open(file)
    data = json.load(f)
    sym_dict = {}
    sym_info = data["graphs"][0]['nodes']
    for info in sym_info:
        names = [info["lbl"]]
        # check if synonym exists:
        if "meta" in info and "synonyms" in info["meta"]:
            names.extend([x["val"] for x in info["meta"]["synonyms"]])
        sym_dict[info['id']] = names
    return sym_dict

def get_diseases(file: str)-> dict:
    """
    get diseases dict: {id: disease}
    """
    f = open(file)
    data = json.load(f)
    di_dict = {}
    for di_info in data["graphs"][0]['nodes']:
        try:
            di_dict[di_info['id']] = di_info['lbl']
        except:
            continue
    return di_dict


def generate_freetext_term(search_term:str) -> Dict:
    return {"FREETEXT":{"searchTerm": search_term}}


def generate_or_element(left: Union[str, Dict], right) -> Dict:
    return {
        "OR": {
            "lhs": generate_freetext_term(left) if isinstance(left, str) else left,
            "rhs": generate_freetext_term(right)
        }
    }

def generate_search(symptom: List[str], disease: List[str]):
    """
    generate symptom and disease search.
    """
    assert isinstance(symptom, list) and len(symptom) >= 1
    assert isinstance(disease, list) and len(disease) >= 1
    symptom_part = generate_freetext_term(symptom.pop())

    if len(symptom) > 0:
        for sym in symptom:
            symptom_part = generate_or_element(symptom_part, sym)
    disease_part = generate_freetext_term(disease.pop())
    if len(disease) > 0:
        for dis in disease:
            disease_part = generate_or_element(disease_part, dis)
    return {"AND":{
        "lhs": disease_part,
        "rhs": symptom_part
    }}

def get_request(symptom: list, disease: list)-> dict:
    """generate document content with specific symptom and disease.
    key: document id-uuid.
    value: paragraphs."""
    s = generate_search(symptom, disease)
    page = 0

    doc_ids = []
    while True:
        resp = requests.post(f"https://api.academia.scaiview.com/api/v6/search/identifier?page={page}&size=1000", json=s)
        try:
            obj = json.loads(resp.text)
            try:
                total = obj["totalPages"]
                total_elements = obj['totalElements']
                if total_elements > 1000:
                    logger.warning("More elements than expected")
                logger.info(f"Found {total_elements}")
                if total == 0:
                    return doc_ids
            except:
                return doc_ids

            doc_ids.extend([x["id"] for x in obj["content"]])

            if (page + 1) == total:
                break
            page += 1
            doc_ids = list(set(doc_ids))   
        
        except:
            doc_ids = []
     
    return doc_ids

def generate_general_search(terms: List[str]):
    """
    generate term search.
    """
    assert isinstance(terms, list) and len(terms) >= 1

    term_part = generate_freetext_term(terms.pop())
    if len(terms) > 0:
        for term in terms:
            term_part = generate_or_element(term_part, term)
    return term_part

def get_count_request(terms: list)-> int:
    """
    generate document content with specific terms.
    """
    s = generate_general_search(terms)
    resp = requests.post(f"https://api.academia.scaiview.com/api/v6/search/identifier?page=0&size=20", json=s)
    try:
        obj = json.loads(resp.text)
        try:
            total_elements = obj['totalElements']
            if total_elements:
                logger.info(f"Found {total_elements}")
                return total_elements
            else:
                return 0
        except:
            return 0
    except:
        return 0


def get_docs(doc_id: str) -> str:
    """
    get documents for specific IDs.
    """

    headers = {'Accept': 'text/html'}
    URL = f"https://api.academia.scaiview.com/api/v6/documents/{doc_id}"
    r = requests.get(URL, headers=headers)
    doc = ''
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'lxml')
        # print(soup.prettify())
        for node in soup.findAll('u'):
            doc += ''.join(node.findAll(text = True))
        if doc:
            logger.info(f'Document for {doc_id} is downloaded.')
        else:
            logger.error(f'No document for {doc_id} in SCAIView.')
        return doc

    else:
        logger.error(f'No document for {doc_id} in SCAIView.')
        return ''


def hypergeometric(disease: str, symptom: str, disease_symptom_docs: int) -> float:
    """
    perform hypergeometric test given disease and symptom.
    """
    # The number of documents in SCAIView (till June 15, 2022).
    total_docs = 36364597

    disease_file = open(f'{COVID_KG_DIR}/disease_count.json')
    disease_data = json.load(disease_file)
    disease_docs = disease_data[disease]

    symptom_file = open(f'{COVID_KG_DIR}/symptoms_count.json')
    symptom_data = json.load(symptom_file)
    symptom_docs = symptom_data[symptom]

    p = hypergeom.sf(disease_symptom_docs, total_docs, disease_docs, symptom_docs)
    return p

 
def get_top_symptoms_(alpha: float, threshold: int) -> list:
    """
    get the symptoms with lowest p_values and larger co-occurances, and return the top symptom list with certain threshold.
    """
    f1 = open(f'{COVID_KG_DIR}/COVID_hypergeometric_{alpha}.json')
    data= json.load(f1)
    f_filter_file = open(f'{COVID_KG_DIR}/COVID_sym_IDs.json')
    data_IDs = json.load(f_filter_file)

    # remove irrelevant terms
    remove_list = ['symptom', 'definition', 'han']
    for elem in remove_list:
        if elem in data:
            del data[elem]
        if elem in data_IDs:
            del data_IDs[elem]

    symptom_co_occurances = []

    for sym, p_value in list(data.items()):
        symptom_co_occurances.append({'symptom': sym, 'p_value': p_value, 'co-occurances': len(data_IDs[sym])})
    new_df = pd.DataFrame(symptom_co_occurances, columns=['symptom', 'p_value', 'co-occurances'])
    
    new_df = new_df.sort_values(['p_value', 'co-occurances'], ascending=[True, False])

    top_symptoms = new_df['symptom'][:threshold]
    print(f'The top 50 symptoms with lowest p_value are: \n{top_symptoms}')
    print('\n')
    print('All symptom and synonyms with ascending p_values are saved in the csv file under folder: /data/Knowledge_graph/COVID.')
    
    # save .csv file.
    new_df.to_csv(f'{COVID_KG_DIR}/COVID_sort_pvalue_occurances.csv')
    return top_symptoms

def get_sym_syn_list(file: str, flag:str) -> list:
    """ 
    This function will retrieve symptom or synonym German list with flag: German.
    """
    csv_dataframe = pd.read_csv(file, encoding='utf_8_sig')
    print(csv_dataframe.shape)
    synonyms_list = []
    for synonyms in csv_dataframe[flag]:
        try:
            synonyms_list.extend(synonyms.split(","))
        except:
            synonyms_list.append(synonyms)
    sym_syn_list = list(set(synonyms_list))

    print(f'There are {len(sym_syn_list)} {flag} synonyms in the symptom corpus and The list has been saved in /Knowledge_graph/COVID/.')
    return sym_syn_list
