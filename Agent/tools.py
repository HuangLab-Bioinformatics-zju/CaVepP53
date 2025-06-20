from dotenv import load_dotenv
from langchain_core.tools import tool
#from serpapi import GoogleSearch    
import os
from textwrap import dedent
from langchain_deepseek import ChatDeepSeek  
from langgraph.prebuilt import InjectedState
from typing import Annotated, Dict
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests
from datetime import datetime
import streamlit as st
import functools
import logging
import multiprocessing
import json
import re
import sys
from io import StringIO
from typing import Dict, Optional
from pydantic import BaseModel, Field
import pandas as pd
from PubMed_RAG import pubmed_rag_agent
logger = logging.getLogger(__name__)
load_dotenv()


from Bio import Entrez
import time
from Bio.PDB import MMCIFParser
from Bio.PDB.Superimposer import Superimposer
import numpy as np

Entrez.email = os.getenv("ENTREZ_EMAIL")
# Search for literature
@tool
def search_pubmed(query, retmax=5)-> str:
    """
    Search PubMed for literature based on a given query.

    Args:
        query (str): The search query.
        retmax (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        str: A string containing the search results.
    """
    try:

        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(1) 
        pmid_list = record["IdList"]
        
        handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()
        cleaned_abstracts = []
        for abstract in abstracts.split("\n\n"):
            if abstract.startswith("Author information"):
                continue
            cleaned_abstracts.append(abstract.strip())

        return "\n\n".join(cleaned_abstracts)
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return "Results could not be returned due to network issues."

@tool
def get_uniprot_info(variant:str) -> str:
    """
    Download and search for relevant information on specified variants from the UniProt database.
    
    """
    import requests
    import re
    base_url = "https://rest.uniprot.org/uniprotkb/"
    uniprot_id="P04637"
    url = f"{base_url}{uniprot_id}.json"
    response = requests.get(url)
    pos = re.findall(r'\d+', variant)
    if pos: 
        pos = int(pos[0]) 
    else:
        pos = None  
    pos = str(pos)
    if response.status_code == 200:
        data = response.json()
    else:
        data=f"Failed to retrieve data for {uniprot_id}"

    functions = []
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            functions.extend([text.get("value") for text in comment.get("texts", [])])
    if functions:
        functions_sum= ";".join(functions)  if functions else "No functional annotations found."   

    ptm_values = []
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "PTM":
            for text in comment.get("texts", []):
                value = text.get("value")
                if value:
                    ptm_values.append(value)
    if ptm_values:
        ptm_sum= "; ".join(ptm_values)  if ptm_values else "No ptm annotations found."

    disease_info = []
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "DISEASE":
            disease = comment.get("disease", {})
            disease_id = disease.get("diseaseId")
            description = disease.get("description")
            diseases = [{
                "diseaseId": disease_id,
                "description": description
            }]
            disease_info.extend(diseases)
    disease_info_strings = []
    for disease in disease_info:
        disease_str = f"Disease ID: {disease['diseaseId']}, Description: {disease['description']}"
        disease_info_strings.append(disease_str)
    disease_sum="\n".join(disease_info_strings) if disease_info_strings else "No disease information found."

    domain_info = []
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "DOMAIN":
            for text in comment.get("texts", []):
                value = text.get("value")
                if value:
                    domain_info.append(value)
    domain_sum="; ".join(domain_info) if domain_info else "No domain information found."


    features = data.get("features", [])
    natural_variants = []
    mutagenesis = []
    other_types = []
    for feature in features:
        feature_type = feature.get("type")
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")
        description = feature.get("description", "")
        #feature_id = feature.get("featureId", "")
        extracted_feature = {
            "type": feature_type,
            "start": start,
            "end": end,
            "description": description,
        }
        if feature_type == "Natural variant":
            if str(start) == str(pos):
                natural_variants.append(extracted_feature)
        elif feature_type == "Mutagenesis":
            if str(start) == str(pos):
                mutagenesis.append(extracted_feature)
        else:
            other_types.append(extracted_feature)
    categorized_features_string = []
    if natural_variants:
        categorized_features_string.append("Natural:")
        for variant in natural_variants:
            variant_str = (f"Start: {variant['start']}, "
                           f"End: {variant['end']}, Description: {variant['description']}")
            categorized_features_string.append(variant_str)
    if mutagenesis:
        categorized_features_string.append("Mutagenesis:")
        for mutation in mutagenesis:
            mutation_str = (f"Start: {mutation['start']}, "
                            f"End: {mutation['end']}, Description: {mutation['description']}")
            categorized_features_string.append(mutation_str)
    if other_types:
        categorized_features_string.append("Other types:")
        for other in other_types:
            if int(pos) >= int(other['start']) and int(pos) <= int(other['end']):
                other_str = (f"Type: {other['type']}, Start: {other['start']}, "
                            f"End: {other['end']},  {other['description']}")
                categorized_features_string.append(other_str)
    features_sum="\n".join(categorized_features_string) if categorized_features_string else "No categorized features found."

    subunits = []
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "SUBUNIT":
            for text in comment.get("texts", []):
                value = text.get("value")
                if value and str(pos) in value:
                    subunits.append(value)
    subunits_sum="; ".join(subunits) if subunits else "No subunit information found."

    cofactor_info = []
    comments = data.get("comments", [])    
    for comment in comments:
        if comment.get("commentType") == "COFACTOR":
            cofactors = comment.get("cofactors", [])
            for cofactor in cofactors:
                name = cofactor.get("name")
                description = cofactor.get("description", "")
                
                cofactor_details = {
                    "name": name,
                    "description": description,
                }
                cofactor_info.append(cofactor_details)
    cofactor_info_strings = []
    for cofactor in cofactor_info:
        cofactor_str = f"Name: {cofactor['name']}, Description: {cofactor['description']}"
        cofactor_info_strings.append(cofactor_str)
    cofactor_sum="\n".join(cofactor_info_strings)
    summary = f""" UniProt Summary for {uniprot_id}:
        ▶ Functional Annotations:{functions_sum}
        ▶ Post-Translational Modifications (PTMs):{ptm_sum}
        ▶ Disease Associations:{disease_sum}
        ▶ Protein Domains:{domain_sum}
        ▶ Subunit Structure:{subunits_sum}
        ▶ Cofactors:{cofactor_sum}
        ▶ Annotated Features:{features_sum}"""
    return summary


@tool
def get_tp53_info(variant:str)-> str:
    """Searches TP53_database for relevation information."""
    import requests
    from bs4 import BeautifulSoup
    base_url = 'https://tp53.cancer.gov'
    url = 'https://tp53.cancer.gov/get_tp53data#get_som_mut'
    downloaded_data = {} 
    response = requests.get(url,timeout=60)
    response.encoding = 'utf-8'
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            logging.error("No tables found.")
            return "No tables found."
        else:
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  
                    cols = row.find_all('td')
                    if len(cols) > 2:
                        name = cols[0].text.strip()
                     
                        download_link = base_url + cols[2].find('a')['href']
                    # download file
                        download_response = requests.get(download_link)
                        if download_response.status_code == 200:
                            csv_data = StringIO(download_response.text)
                            df = pd.read_csv(csv_data)
                            file_name = name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                            downloaded_data[file_name] = df
                            logging.info(f"Downloaded {file_name}")
                        else:
                             logging.error(f"Failed to download {name}: HTTP {download_response.status_code}")
            #search mutation from data
            relevant_data=[]
            for name, df in downloaded_data.items():
                if 'ProtDescription' in df.columns:
                    df['ProtDescription'] = df['ProtDescription'].fillna('')
                    relevant_rows = df[df['ProtDescription'].str.contains(variant, case=False, na=False)]
                    if not relevant_rows.empty:
                        description = f"\nInformation about {name}:"
                        for column in relevant_rows.columns:
                            if column != 'ProtDescription':  
                                values = relevant_rows[column].dropna().unique()
                                if len(values) > 0:
                                    values_str = ', '.join(map(str, values))
                                    description+= f"""{column}: {values_str}""" 
        return description
    else:
        logging.error('Failed to retrieve the webpage')
        return 'Failed to retrieve the webpage'


@tool
def get_MSA_info(pos:int)-> str:
    """Searches MSA file for the query."""
    try:
        data_path = './data/entropy_conservation_scores.csv'
        df = pd.read_csv(data_path)    
        record = df[df['pos'] == pos]
        if record.empty:
            return (f"Error: No record found at position {pos} in file {data_path}.")
        return (f"MSA-related information: The{record['pos'].values[0]}-th amino acid in the sequence has an entropy score of {record['entropy_score'].values[0]},and a conservation score of {record['conservation_score'].values[0]}.The average value of the Entropy score is: 0.6055573599652174.The average value of the Conservation score is: 84.32178508514384")    
    except FileNotFoundError:
        return (f"Error: File {data_path} not found.")
    except Exception as e:
        return (f"An error occurred:{e}")
@tool   
def get_virtual_perturbation_info(variant:str)-> str:
    """Searches virtual perturbation file for the query."""
    data_path = './data/protein_virtual_perturbation.csv'
    df = pd.read_csv(data_path)
    record = df[df['Mutation'] == variant]
    if record.empty:
        return (f"Error: No record found for variant {variant} in file {data_path}.")
    description = f"The variant '{record['Mutation']}' has the following properties(The primary results are based on the CaVepP53 label):\n"
    description += f"- Number of labels: {record['labels']};Label 0 indicates benign, Label 1 indicates pathogenic,Label 2 indicates uncertainty.\n"
    description += f"- CaVepP53 label: {record['CaVepP53_label']};Label 0 indicates benign, Label 1 indicates pathogenic.\n"
    description += f"- CaVepP53 score: {record['CaVepP53_score']}\n"
    description += f"- Confidence score: {record['confidence_score']}\n"
    description += f"- Alphamissense score: {record['AM_score']}\n, These are derived using the following thresholds:likely benign if alphamissense pathogenicity < 0.34; likely pathogenic if alphamissense pathogenicity > 0.564; and ambiguous otherwise.)"
    description += f"- Alphamissense class: {record['AM_class']}"
    return description


@tool
def compare_structure(cif_file: str)-> str:
    """
    Compare two protein structures from Alphafold3 CIF files.

    Args:
        cif_file (str): The content of the Mutation CIF file.

    Returns:
        str: A string containing the comparison results.
    """

    parser = MMCIFParser()
    cif_WT = r'./data/fold_tp53_wild_tpye_model_0.cif'

    structure1 = parser.get_structure("structure1", cif_WT)
    structure2 = parser.get_structure("structure2", cif_file)
    
    model1 = structure1[0]
    model2 = structure2[0]
    
    backbone_atoms1 = [atom for atom in model1.get_atoms() if atom.name in ['N', 'CA', 'C', 'O']]
    backbone_atoms2 = [atom for atom in model2.get_atoms() if atom.name in ['N', 'CA', 'C', 'O']]
    

    if len(backbone_atoms1) != len(backbone_atoms2):
        raise ValueError("The two structures have a different number of main-chain atoms and cannot be aligned.")
    
   
    superimposer = Superimposer()
    superimposer.set_atoms(backbone_atoms1, backbone_atoms2)
    superimposer.apply(model2.get_atoms())  
    
    rmsd = superimposer.rms
    
    ca_atoms1 = [atom for atom in backbone_atoms1 if atom.name == 'CA']
    ca_atoms2 = [atom for atom in backbone_atoms2 if atom.name == 'CA']
    
    coords1 = np.array([atom.get_coord() for atom in ca_atoms1])
    coords2 = np.array([atom.get_coord() for atom in ca_atoms2])
    
    thresholds = [1, 2, 4, 8]
    if len(coords1) != len(coords2):   
       raise ValueError("The two structures have a different number of Cα atoms.")
    distances = np.linalg.norm(coords1 - coords2, axis=1)
    gdt_scores = []
    for threshold in thresholds:
        score = np.sum(distances <= threshold) / len(coords1)
        gdt_scores.append(score)
        gdt_ts = np.mean(gdt_scores)    
    result_description = (
        f"The RMSD between the two protein structures is {rmsd:.3f} Å, indicating the average deviation of their main-chain atoms in spatial position."
        f"The GDT-TS score is {gdt_ts:.4f}, reflecting the global similarity between the two structures."
        f"The GDT scores at different distance thresholds are as follows: 1 Å, {gdt_scores[0]:.4f}; 2 Å, {gdt_scores[1]:.4f}; "
        f"4 Å, {gdt_scores[2]:.4f}; 8 Å, {gdt_scores[3]:.4f}."
    )
    return result_description


@tool
def report_tool(state: Annotated[Dict, InjectedState], query: str) -> str:
    """Generates a comprehensive scientific report based on the conversation history.
    
    This tool takes the entire conversation history and generates a well-structured scientific report
    in academic paper format, covering  TP53-related literature and data through systematic workflows.
    The report includes sections like Abstract, Introduction, Methods, Results,Discussion, Conclusion, 
    and References to elucidate the potential effects of specific variants, and it will also provide 
    detailed experimental planning and guidance to facilitate the validation of these findings in a 
    laboratory setting.
    
    The tool saves the report as a PDF file.
    
    Args:
        state: The current conversation state containing message history
        query: Additional context or specific requirements for the report (optional)
        
    Returns:
        str: Confirmation message with the path to the saved PDF file
    """

    # Extract the chat history from the injected state
    chat_history = state["messages"]



    report_prompt = """
    # Scientific Analysis Report

    <objective>
    Generate a comprehensive scientific report (minimum 1000 words) focused on predicting the effects of TP53 variants. The report should be specific and avoid general statements. All analysis should be based on data presented in the conversation, focusing on TP53 variants.
    </objective>

    <report_structure>
    ## 1. Objective
    - Clear statement of the research goals(The focus is primarily on a specific research variant of TP53).
    - Overview of what the report aims to address regarding TP53 variant effects

    ## 2. Study Overview
    - Purpose of the study.
    - Key research questions being investigated regarding TP53 variants.

    ## 3. Literature Review
    - Discuss the research background pertaining to the variant in question, and investigate if any relevant prior studies have already been conducted regarding it.
    - Summarize the potential functional impacts that alterations may have on P53 as identified in the literature found.
    - Summarize the literature if available; otherwise, this section can be skipped.

    ## 4. Key Findings
    - Data sources and variants Analysis.
    - Detailed results from each analysis of TP53 variants.
    - Identify the impacts of the variant across various levels, including molecular pathways, protein structures, and cellular processes.

    ## 5. Biological Implications
    - Interpretation of the biological significance of the findings related to TP53 variants.
    - Discussion of broader impacts and relevance

    ## 6. Experimental Plan Based on Potential Compilation Results
    - Define experimental goals to validate TP53 variant predictions based on previously analyzed potential functional impacts.
    - Detail the step-by-step methods, and ensure high feasibility

    ## 7. Conclusion
    - Summary of major discoveries regarding TP53 variant effects.
    - Future research directions in TP53 variant analysis.

    ## 8. References
    - Relevant citations from literature searches
    - Format: Title only (NO author names or years or URL)
    </report_structure>

    <important_instructions>
    1. OUTPUT ONLY THE REPORT CONTENT, NO OTHER TEXT
    2. Use specific data-driven insights rather than general statements
    3. Maintain scientific tone throughout
    4. Do not assume conclusions not supported by the data
    5. Be consistent with the user's input language. you are a multi-lingual assistant.
    FORMAT: THE REPORT SHOULD BE IN MARKDOWN FORMAT.
    </important_instructions>
    """    
    # Generate the report
    ins = chat_history[:-1] + [HumanMessage(content=report_prompt, name="report_tool")]
    st.write(ins)
    llm = ChatDeepSeek(model="deepseek-chat",max_tokens=8000)
    report = llm.invoke(ins)
    try:
        # Save as markdown file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('output_report', exist_ok=True)
        md_filename = f'./output_report/report_{timestamp}.md'
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(report.content)
        return f"Report has been saved as markdown file: {md_filename}\n\nHere is the report content:\n\n{report.content}"
        
    except Exception as e:
        return f"Error saving markdown file: {str(e)}"


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")

class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")  # type: ignore[arg-type]
    locals: Optional[Dict] = None  # type: ignore[arg-type]

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, queue)
        # get the result from the worker function
        return queue.get()




