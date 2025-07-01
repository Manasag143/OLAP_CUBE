import logging
import time
import json
import pandas as pd
import csv
from typing import Dict, List, Tuple ,Optional
import os
import requests
from pathlib import Path
from datetime import date
from colorama import Fore, Style ,init
from langchain_core.documents import Document
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import nltk 
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
history_file = os.path.join(BASE_DIR, "conversation_history.json")
vector_db_path = os.path.join(BASE_DIR, "vector_db")
config_file = os.path.join(BASE_DIR, "config.json")

# Add this class after your imports
class BM25FileNotFoundError(Exception):
    """Raised when BM25 pickle file is not found"""
    pass

def load_documents_from_json(cube_id: str, doc_type: str, base_dir: str) -> List[Document]:
    """Load documents from JSON file"""
    try:
        file_path = os.path.join(base_dir, cube_id, f"{cube_id}_{doc_type}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cube data doesn't exits")
            
        with open(file_path) as f:
            data = json.load(f)
            
        # Convert to Document objects
        documents = []
        for doc in data:
            content = f"Group Name:{doc['Group Name']}--Level Name:{doc['Level Name']}--Description:{doc['Description']}"
            documents.append(Document(page_content=content))
        return documents
            
    except Exception as e:
        logging.error(f"Error loading {doc_type} documents: {str(e)}")
        raise


# Add this helper method to DimensionMeasure class
def load_bm25_documents(self, cube_id: str) -> List[Document]:
    """Load saved BM25 documents"""
    try:
        bm25_file = os.path.join(vector_db_path, cube_id, f"{cube_id}_measures.pkl")
        
        if not os.path.exists(bm25_file):
            raise FileNotFoundError(f"BM25 file not found at {bm25_file}")
            
        with open(bm25_file, 'rb') as f:
            try:
                documents = pickle.load(f)
                return documents
            except Exception as e:
                logging.error(f"Error loading pickle file: {str(e)}")
                raise ValueError(f"Failed to load BM25 documents: {str(e)}")
                
    except Exception as e:
        logging.error(f"Error in load_bm25_documents: {str(e)}")
        raise
    
def setup_logging():
    """
    function responsible for storing errors in log folder datewise and also stores token consumptions.
    """
    today = date.today()
    log_folder = './log'
  
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    logging.basicConfig(filename = f"{log_folder}/{today}.log",level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')


class LLMConfigure:
    """
    Class responsible for loading and configuring LLM and embedding models from a config file.
    """

    def __init__(self, config_path: json = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            print(config_path)
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise

    def initialize_llm(self):
        """Initializes and returns the LLM model."""
        try:
            # LLM initialization using the config
            self.llm = AzureChatOpenAI(openai_api_key= self.config['llm']['OPENAI_API_KEY'],
                                      model=self.config['llm']['model'],
                                      temperature=self.config['llm']['temperature'],
                                      api_version= self.config['llm']["OPENAI_API_VERSION"],
                                      azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      seed=self.config['llm']["seed"]
            )
            return self.llm
        except KeyError as e:
            logging.error(f"Missing LLM configuration in config file: {e}")
            raise

    def initialize_embedding(self):
        """Initializes and returns the Embedding model."""
        try:
            # embedding initialization using the config
            self.embedding = AzureOpenAIEmbeddings(deployment = self.config['embedding']['deployment'],
                                      azure_endpoint = self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      openai_api_key = self.config['llm']['OPENAI_API_KEY'],
                                      show_progress_bar = self.config['embedding']['show_progress_bar'],
                                      disallowed_special = (),
                                      openai_api_type = self.config['llm']['OPENAI_API_TYPE']
                        )
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise


class DimensionMeasure:
    """
    Class responsible for extracting dimensions and measures from the natural language query.
    """

    def __init__(self,llm: str,embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore
        self.vector_embedding= vectorstore
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 5

    def load_bm25_documents(self, cube_id: str) -> List[Document]:
        bm25_file = os.path.join(vector_db_path, cube_id, f"{cube_id}_measures.pkl")
        try:
            with open(bm25_file, 'rb') as f:
                documents = pickle.load(f)
                return documents
        except FileNotFoundError:
            raise BM25FileNotFoundError(
                f"BM25 document file not found for cube {cube_id}"
            )

    def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts dimensions from the query."""
        try:

            with get_openai_callback() as dim_cb:
                
                query_dim = """ 
                As an SQL CUBE query expert, analyze the user's question and identify all relevant cube dimensions from the dimensions delimited by ####.
                
                <instructions>
                - Select relevant dimension group, level, description according to user query from dimensions list delimited by ####
                - format of one dimension: Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description> 
                - Include all dimensions relevant to the question in the response
                - Group Name and Level Name can never be same, extract corresponding group name for a selected level name according to the user query and vice versa.
                - If relevant dimensions group and level are not present in the dimensions list, please return "Not Found"
                - If the query mentions date, year, month ranges, include corresponding dimensions in the response  
                </instructions>
                
                
                Response format:
                Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>

                Review:
                - ensure dimensions are only selected from dimensions list delimited by ####
                - Group Name and Level Name can never be same, extract corresponding group name, description for a selected level name according to the user query and vice versa.
                - Kindly ensure that the retrieved dimensions group name and level name is present otherwise return "Not found".

                User Query: {question}
                ####
                {context}
                ####

                """


                print(Fore.RED + '    Identifying Dimensions group name and level name......................\n')
                
                # NOTE: ---------------- uncomment below code only once if want to store vector embeddings of dimension table -------------------
                
                # vectordb = Chroma.from_documents(
                # documents=text_list_dim , # chunks
                # embedding=self.embedding, # instantiated embedding model
                # persist_directory=persist_directory # directory to save the data
                # )

                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
                
                # Initialize BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)
                    
                    # Set up vector store directory
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                persist_directory_dim = cube_dim
                
                load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=self.embedding)
                vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                prev_q = prev_conv["query"]
                prev_d = prev_conv["dimensions"]
                prev_cube_q = prev_conv["response"]
                data = {"prev_query":prev_q,"prev_measures":prev_d}

                # Create QA chain with prompt template
                QA_CHAIN_PROMPT = PromptTemplate(
                    template=query_dim,
                    input_variables=["query", "context"]
                )

                # Create ensemble retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )
                # Initialize and run QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=ensemble_retriever,
                    return_source_documents=True,
                    verbose=True,
                    chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=query_dim,
                        input_variables=["query", "context"]
                    ),
                    "verbose": True
                }
                )

                # Get results
                result = qa_chain({"query": query, "context": ensemble_retriever})
                dim = result['result']
                print(Fore.GREEN + '    Identified Group and level name :        ' + str(dim))
                logging.info(f"Extracted dimensions :\n {dim}")
                return dim

        except Exception as e:
            logging.error(f"Error extracting dimensions : {e}")
            raise

    def get_dimensions_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
        """Extract dimensions with error correction."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim_error_inj = f"""
                As a user query to cube query conversion expert, analyze the error message from applying the cube query. 
                Your goal is to correct the identification of dimensions if they are incorrect.
                
                Below details using which the error occurred:-
                User Query: {query}
                Current Dimensions: {prev_conv["dimensions"]}
                Current Cube Query Response: {prev_conv["response"]}
                Error Message: {error}

                <error_analysis_context>
                - Analyze the error message to identify incorrect dimension references
                - Check for syntax errors in dimension names/hierarchy levels
                - Verify dimension existence in cube structure
                - Identify missing or invalid dimension combinations
                - Consider temporal context for date/time dimensions
                </error_analysis_context>

                <correction_guidelines>
                - Strictly Give only dimension only
                - Fix incorrect dimension names
                - Correct hierarchy references
                - Add required missing dimensions
                - Remove invalid combinations
                - Preserve valid dimensions
                - Maintain temporal relationships
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["corrected_group1", "corrected_group2"],
                    "dimension_level_names": ["corrected_level1", "corrected_level2"],
                    "corrections": ["Fixed dimension X", "Added missing dimension Y"],
                    "reasoning": "Explanation of corrections made"
                }}
                '''
                </correction_guidelines>

                <examples>
                1. Error: "Unknown dimension [Time].[Date]"
                Correction: Change to [Time].[Month] since Date level doesn't exist

                2. Error: "Invalid hierarchy reference [Geography].[City]"
                Correction: Change to [Geography].[Region].[City] to include parent

                3. Error: "Missing required dimension [Product]"
                Correction: Add [Product].[Category] based on context

                4. Error: "Incompatible dimensions [Customer] and [Item]"
                Correction: Remove [Item] dimension as it's incompatible
                </examples>
                """
                
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                load_embedding_dim = Chroma(
                    persist_directory=cube_dim,
                    embedding_function=self.embedding
                )
                retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 15})
                chain_dim = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_dim,
                    verbose=True,
                    return_source_documents=True
                )
                result_dim = chain_dim.invoke({"query": query_dim_error_inj})
                return result_dim.get('result')

        except Exception as e:
            logging.error(f"Error in dimension correction: {e}")
            raise

    def get_dimensions_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str) -> str:
        """Extracts dimensions with user feedback consideration."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim_feedback = f"""
                As a user query to cube query conversion expert, analyze the user feedback for the cube query. 
                Your goal is to refine the identification of dimensions based on user feedback.
                
                Below details using which the feedback was received:-
                User Query: {query}
                Current Dimensions: {prev_conv["dimensions"]}
                Current Cube Query Response: {prev_conv["response"]}
                Feedback Type: {feedback_type}

                <feedback_analysis_context>
                - For accepted feedback: reinforce successful dimension mappings
                - For rejected feedback: identify potential dimension mismatches
                - Utilize the user feedback to generate more accurate response
                - Consider previous successful queries as reference
                - Analyze dimension hierarchy appropriateness
                - Check for implicit dimension references
                </feedback_analysis_context>

                <refinement_guidelines>
                - Strictly provide dimensions only
                - For accepted feedback: maintain successful mappings
                - For rejected feedback: suggest alternative dimensions
                - Consider dimension hierarchies
                - Maintain temporal relationships
                
                Response format:
                '''json
                {{
                    "dimension_group_names": ["refined_group1", "refined_group2"],
                    "dimension_level_names": ["refined_level1", "refined_level2"],
                    "refinements": ["Maintained dimension X", "Changed dimension Y"],
                    "reasoning": "Explanation of refinements made"
                }}
                '''
                </refinement_guidelines>
                """
                
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                load_embedding_dim = Chroma(
                    persist_directory=cube_dim,
                    embedding_function=self.embedding
                )
                retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 15})
                chain_dim = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_dim,
                    verbose=True,
                    return_source_documents=True
                )
                result_dim = chain_dim.invoke({"query": query_dim_feedback})
                return result_dim.get('result')

        except Exception as e:
            logging.error(f"Error in dimension feedback processing: {e}")
            raise

    def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts measures from the query."""
        try:
            
            with get_openai_callback() as msr_cb:
                
                 query_msr = """ 
                As an SQL CUBE query expert, analyze the user's question and identify all relevant cube measures from the measures delimited by ####.
                
                <instructions>
                - Select relevant measure group, level, description according to user query from measures list delimited by ####
                - format of one measure: Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description> 
                - Include all measures relevant to the question in the response
                - Group Name and Level Name can never be same, extract corresponding group name for a selected level name according to the user query and vice versa.
                - If relevant measures are not present in the measures list, please return "Not Found" 
                </instructions>
                
                <examples>
                Query: Remove City
                Response: Not Found

                Query: What is the rank by Balance amount for Mutual Fund
                Response: Group Name:Business Drivers--Level Name:Balance Amount Average--Description:Average balance amount of the customer/borrower
                Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity--Description:Mutual Fund Quantity
                Group Name:Fund Investment Details--Level Name:Mutual Fund QoQ EOP--Description:Mutual Fund QoQ EOP
                Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity QoQ--Description:Mutual Fund Quantity QoQ
                Group Name:Fund Investment Details--Level Name:Mutual Fund MoM EOP--Description:Mutual Fund MoM EOP
                
                </examples>

                Response format:
                Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>

                Review:
                - ensure measures are only selected from measures list delimited by ####
                - Group Name and Level Name can never be same, extract corresponding group name, description for a selected level name according to the user query and vice versa.
                - Kindly ensure that the retrieved measures group name and level name is present otherwise return "Not found".


                User Query: {question}

                ####
                {context}
                ####

                
                """


            print(Fore.RED + '    Identifying Measure group name and level name......................\n')
                
                # NOTE: uncomment below lines only if want to store vector embedding of measure table for first time  ------
                
                # vectordb = Chroma.from_documents(
                # documents=text_list_msr , # chunks
                # embedding=self.embedding, # instantiated embedding model
                # persist_directory=persist_directory_msr # directory to save the data
                # )
            # Load BM25 documents from pickle file
            """bm25_file = os.path.join(vector_db_path, cube_id, f"{cube_id}_measures.pkl") 
            try: 
                with open(bm25_file, 'rb') as f: 
                    documents = pickle.load(f) 
                logging.info(f"Successfully loaded {len(documents)} BM25 documents for cube {cube_id}") 
            except FileNotFoundError: 
                logging.error(f"BM25 file not found at {bm25_file}") 
                raise FileNotFoundError(f"BM25 document file not found for cube {cube_id}") 
            except Exception as e: 
                logging.error(f"Error loading BM25 file: {str(e)}") 
                raise ValueError(f"Failed to load BM25 documents: {str(e)}")
            """
            # Load documents from JSON
            documents = load_documents_from_json(cube_id, "measures", vector_db_path)
            
            bm25_retriever = BM25Retriever.from_documents(
                        documents,k=10)

            cube_msr = os.path.join(vector_db_path, cube_id)
            cube_msr = os.path.join(cube_msr, "measures")
            
            persist_directory_msr = cube_msr
            load_embedding_msr = Chroma(persist_directory=persist_directory_msr, embedding_function=self.embedding)

            """#new
            results = load_embedding_msr.similarity_search_with_score(
                        query_msr, k=45)
            print("*"*50)
            print(results)
            for res, score in results:
                print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
            #end of new"""

            vector_retriever = load_embedding_msr.as_retriever(search_type="similarity",search_kwargs={"k": 20})
            prev_q = prev_conv["query"]
            prev_m = prev_conv["measures"]
            prev_cube_q = prev_conv["response"]
            data = {"prev_query":prev_q,"prev_measures":prev_m}
            QA_CHAIN_PROMPT = PromptTemplate(template=query_msr, input_variables=["query","context"])
                                                #partial_variables=data)
            
            ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )

            # Run QA chain with ensemble retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=ensemble_retriever,
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=query_msr,
                        input_variables=["query", "context"]
                    ),
                    "verbose": True
                }
            )
            
            result = qa_chain({"query": query, "context": ensemble_retriever})
            msr = result['result']

            print(Fore.GREEN + '    Measures result :        ' + str(result)) 
            logging.info(f"Extracted measures :\n {msr}")  
            return msr
        
        except Exception as e:
            print("Error:{}".format(e))
            logging.error(f"Error Extracting Measure")
            raise

    def get_measures_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
        """Extracts measures with error correction logic"""
        try:
            with get_openai_callback() as msr_cb:
                query_msr_error_inj = f"""
                As a user query to cube query conversion expert, analyze the error message from applying the cube query. 
                Your goal is to correct the identification of measures if they are incorrect.
                
                Below details using which the error occurred:-
                User Query: {query}
                Current Measures: {prev_conv["measures"]}
                Current Cube Query Response: {prev_conv["response"]}
                Error Message: {error}

                <error_analysis_context>
                - Analyze error message for specific measure-related issues
                - Check for syntax errors in measure references
                - Verify measures exist in the cube structure
                - Look for aggregation function errors
                - Identify calculation or formula errors
                - Check for measure compatibility issues
                </error_analysis_context>

                <correction_guidelines>
                - Strictly Give only measure only
                - Fix measure name mismatches
                - Correct aggregation functions
                - Fix calculation formulas
                - Add missing required measures
                - Remove invalid measure combinations
                - Preserve valid measure selections
                
                Response format:
                '''json
                {{
                    "measure_group_names": ["corrected_group1", "corrected_group2"],
                    "measure_names": ["corrected_measure1", "corrected_measure2"],
                    "corrections": ["Fixed measure X", "Updated aggregation Y"],
                    "reasoning": "Explanation of corrections made"
                }}
                '''
                </correction_guidelines>

                <examples>
                1. Error: "Unknown measure [Sales].[Amount]"
                Correction: Change to [Sales].[Sales Amount] as the correct measure name

                2. Error: "Invalid aggregation function SUM for [Average Price]"
                Correction: Change to AVG([Price].[Unit Price]) for correct aggregation

                3. Error: "Incompatible measures [Profit] and [Margin %]"
                Correction: Use [Sales].[Profit Amount] instead of incompatible combination

                4. Error: "Calculation error in [Growth].[YoY]"
                Correction: Add proper year-over-year calculation formula

                5. Error: "Missing required base measure for [Running Total]"
                Correction: Add base measure [Sales].[Amount] for running total calculation
                </examples>
                """

                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_msr = os.path.join(cube_dir, "measures")
                
                load_embedding_msr = Chroma(
                    persist_directory=cube_msr,
                    embedding_function=self.embedding
                )
                retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 15})
                
                chain_msr = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_msr,
                    verbose=True,
                    return_source_documents=True
                )
                
                result_msr = chain_msr.invoke({"query": query_msr_error_inj})
                msr = result_msr.get('result')
                
                logging.info(f"Extracted corrected measures:\n {msr}")
                return msr

        except Exception as e:
            logging.error(f"Error in measure correction: {str(e)}")
            raise          

    def get_measures_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str) -> str:
        """Extracts measures with user feedback consideration."""
        try:
            with get_openai_callback() as msr_cb:
                query_msr_feedback = f"""
                As a user query to cube query conversion expert, analyze the user feedback for the cube query. 
                Your goal is to refine the identification of measures based on user feedback.
                
                Below details using which the feedback was received:-
                User Query: {query}
                Current Measures: {prev_conv["measures"]}
                Current Cube Query Response: {prev_conv["response"]}
                Feedback Type: {feedback_type}

                <feedback_analysis_context>
                - For accepted feedback: reinforce successful measure mappings
                - For rejected feedback: identify potential measure mismatches
                - Utilize the user feedback to generate more accurate response
                - Consider aggregation appropriateness
                - Check for calculation errors
                - Analyze measure combinations
                </feedback_analysis_context>

                <refinement_guidelines>
                - Strictly provide measures only
                - For accepted feedback: maintain successful mappings
                - For rejected feedback: suggest alternative measures
                - Consider aggregation functions
                - Maintain calculation integrity
                
                Response format:
                '''json
                {{
                    "measure_group_names": ["refined_group1", "refined_group2"],
                    "measure_names": ["refined_measure1", "refined_measure2"],
                    "refinements": ["Maintained measure X", "Changed measure Y"],
                    "reasoning": "Explanation of refinements made"
                }}
                '''
                </refinement_guidelines>
                """

                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_msr = os.path.join(cube_dir, "measures")
                load_embedding_msr = Chroma(
                    persist_directory=cube_msr,
                    embedding_function=self.embedding
                )
                retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 15})
                chain_msr = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever_msr,
                    verbose=True,
                    return_source_documents=True
                )
                result_msr = chain_msr.invoke({"query": query_msr_feedback})
                return result_msr.get('result')

        except Exception as e:
            logging.error(f"Error in measure feedback processing: {e}")
            raise

class FinalQueryGenerator(LLMConfigure):
    """
    Class responsible for generating the final OLAP query based on dimensions and measures.
    """
    def __init__(self,query, dimensions: None, measures: None,llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 6
        
    def call_gpt(self,final_prompt:str):
      """
      function responsible for generating final query
      """
      API_KEY = self.config['llm']['OPENAI_API_KEY']
      headers = {
          "Content-Type": "application/json",
          "api-key": API_KEY,
      }
      
      # Payload for the request
      payload = {
        "messages": [
          {
            "role": "system",
            "content": [
              {
                "type": "text",
                "text": "You are an AI assistant that writes accurate OLAP cube queries based on given query."
              }
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": final_prompt
              }
            ]
          }
        ],
        "temperature": self.config['llm']['temperature'],
        "top_p": self.config['llm']['top_p'],
        "max_tokens": self.config['llm']['max_tokens']
      }
      
      
      ENDPOINT = self.config['llm']['ENDPOINT']
      
      # Send request
      try:
          response = requests.post(ENDPOINT, headers=headers, json=payload)
          response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
      except requests.RequestException as e:
          raise SystemExit(f"Failed to make the request. Error: {e}")
      
      output = response.json()
      token_details = output['usage']
      output = output["choices"][0]["message"]["content"]
      return output ,token_details


    def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:

        try:
            if not dimensions or not measures:
                raise ValueError("Both dimensions and measures are required to generate a query.")
                
            final_prompt = f"""You are an expert in generating SQL Cube query. You will be provided dimensions delimited by $$$$ and measures delimited by &&&&.
            Your Goal is to generate a precise single line cube query for the user query delimited by ####.


            Instructions:            
            - Generate a single-line Cube query without line breaks
            - Include 'as' aliases for all level names in double quotes. alias are always level names.
            - Choose the most appropriate dimensions group names and level from dimensions delimited by $$$$ according to the query.
            - Choose the most appropriate measures group names and level from measures delimited by &&&& according to the query.
            - check the examples to learn about correct syntax, functions and filters which can be used according to the user query requirement.
            - User Query could be a follow up query in a conversation, you will also be provided previous query, dimensions, measures, cube query. Generate the final query including the contexts from conversation as appropriate.

            Formatting Rules:
            - Dimensions format: [Dimension Group Name].[Dimension Level Name] as "Dimension Level Name"
            - Measures format: [Measure Group Name].[Measure Level Name] as "Measure Level Name"
            - Conditions in WHERE clause must be properly formatted with operators
            - For multiple conditions, use "and" "or" operators
            - All string values in conditions must be in single quotes
            - All numeric values should not have leading zeros
            
            Functions and filters supported for query generation:-
            <functions>
            
            - IF	IF(condition, TRUE Statement, FALSE Statement	IF ( [Measures.Brokerage] = 0 or [Measures.Brokerage] = null, null, [Measures.Total Cost incl rent]  /  [Measures.Brokerage] )	Conditional IF Else calculation on Measures
            ISPRESENT	ISPRESENT(Level Name)	IF ( ISPRESENT ( [Branch.Zone] ) ,  [Measures.PROFIT] ,  0) 	"Used as condition in IF function Checks whether level zone is present in report or not, if present then returns true else false"
            TRENDNUMBER	"TRENDNUMBER ( measure name,  Time group level ,  number of period,  trend type ) 

            - trend type = ('average','value','percentage','sum','delta')	TRENDNUMBER ( [Measures.PROFIT] ,  [Calendar.Year] ,  <number of years>,  'value' ) 	"Trendnumber function is used for performing the lead or lag computations across time dimension.

            Type of trends supported are
             - average = average of previous (lag) or next (lead) period
             - value = value of previous (lag) or next (lead) period 
             - percentage = percentage change  of previous (lag) or next (lead) period 
             - sum = sum  of previous (lag) or next (lead) period 
             - delta = change in value between the current and previous (lag) period or next (lead) period
             - PERIODSTODATE	"PERIODSTODATE ( Time dimension level , measure , period type ) ) 

            - period type = ('avg','sum')"	PERIODSTODATE ( [Calendar.FiYear] , [Measures.PROFIT] , 'avg' ) ) 	"Used for calculation of Year to Date (YTD) Months to Date (MTD) etc.

            - Aggregation supported: sum, avg
            FILTERKPI	FILTERKPI ( kpi , condition) 	FILTERKPI ( [Measures.PROFIT] , [Branch Type.Status] = 'ACTIVE') 	"Conditional Aggregation of Measure.
            Filters data with the condition mentioned and then applies the measure aggregate as defined in cube
            Used for functions like conditional sum, conditional average etc."

            - EOP	EOP ( kpi, time dimension level)	EOP ( [Measures.# of Employees] , [Calendar.Year] ) 	As per time dimension level, it will calculate the End of Period
            ROUND	ROUND( kpi, number)	ROUND ( [Measures.PROFIT] , 3 )	Rounding Function on the measure
            COALESCE	COALESCE( kpi, number)	COALESCE ( [Measures.PROFIT] , 0)	If the value is null, what default value to be taken can be defined using COALESCE function
            MAX / SUM / MIN / COUNT / AVG	SUM ( dimension level, 'specific'/'current'/'all', kpi,  dimension level )	SUM ( [Customer.Customer Code] ,  'specific',  [Measures.Exposure],  [Account.Account Number] ) 	"specific = aggregate exposure measure on customer code level for dataset. (Dataset contains the distinct values of exposure according to account number level).

            This is customised function and is applicable for measures which are defined at a particular level.
            In this example Exposure is always at Customer Code level."

            Use below filters on functions based on query requirements:
            - TimeBetween:- use this function if any date, month, year range is mentioned in query 
            - TrendNumber:- for year-over-year/period comparisons
            - Head/Tail:- for top/bottom N queries
            - percentage:- Percentage cumulative calculations
            - runningsum/percentageofrunningsum:- for cumulative calculations runningsum([<group name>].[<level name>],'sumacrossrows'/'percentagerunningsumacrossrows')
            - rank:- it will provide rank for values rank([<group name>].[<level name>],'rankAcrossRows')
            
            </functions>

            <examples>

            user query:What is the Total Revenue and Total Revenue From Operations Between the year 2012 to 2017?
            Expected Response:-select [Time].[Year] as "Year", [Financial Data].[Total Revenue] as "Total Revenue", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",TimeBetween(20120101,20171231,[Time].[Year], false) from [Cube].[{cube_name}]
	    
	    user query:Mutual Fund Name wise Trade Price & Quantity ?
  	    Expected Response:select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Bulk Deal Trade].[Trade Price] as "Trade Price", [Bulk Deal Trade].[Traded Quantity] as "Traded Quantity" from [Cube].[{cube_name}]

            user query:-Which months has the value of balance average amount between 40,00,00,000 to 2,00,00,00,000 ?           
            Expected Response:-select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount Average] between 400000000.00 and 2000000000.00

            user query:-Top 5 Cities on Average Balance Amount.
            Expected Response:-select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average",Head([Branch Details].[City],[Business Drivers].[Balance Amount Average],5,undefined) from [Cube].[{cube_name}]

            user query:-Please provide Cities and Average Balance Amount where Average Balance Amount is more than 5000 and Count of Customers more than 10.
            Expected Response:-select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Count of Customers] > 10.00 and [Business Drivers].[Balance Amount Average] > 5000.00

            user query:-what is the count of customers based on rating ?
            Expected Response:-select [External Funding].[Rating] as "Rating", [Business Drivers].[Count of Customers] as "Count of Customers" from [Cube].[{cube_name}]

            user query:-Please provide Year and Average Balance Amount and % change from previous year for past 2 years
            Expected Response:-select [Time].[Year] as "Year", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", TrendNumber([Business Drivers].[Balance Amount Average],[Time].[Year],2,'percentage') as "YOY % Change" from [Cube].[{cube_name}]

            user query:-What is the closing price of Industry ?
            Expected Response:-select [Industry Details].[Industry Name] as "Industry Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[{cube_name}]

            user query:-Which are the bottom 4 years having lowest total revenue from operation ?
            Expected Response:-select [Time].[Year] as "Year", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",Tail([Time].[Year],[Financial Data].[Total Revenue From Operations],4,undefined) from [Cube].[{cube_name}]

            user query:-What are the closing price of AXIS, HDFC, ICICI & LIC Mutual Funds ?
            Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[{cube_name}] where [Mutual Fund Investment].[Mutual Fund Name] in ('AXIS','HDFC','ICICI','LIC')

            user query:-What are the Cash Ratio Current Ratio and Quick Ration Based on Month from 2012 to 2017 ? 
            Expected Response:-select [Time].[Month] as "Month", [Financial Ratios].[Cash Ratio] as "Cash Ratio", [Financial Ratios].[Current Ratio] as "Current Ratio", 
            [Financial Ratios].[Quick Ratio] as "Quick Ratio",TimeBetween(20120101,20171231,[Time].[Month], false) from [Cube].[{cube_name}]

            user query:-Provide Current Ratio, Cash Ratio, and Quick ration segregated by industry group?
            Expected Response:-select [Industry Details].[Industry Group Name] as "Industry Group Name", [Financial Ratios].[Current Ratio] as "Current Ratio", [Financial Ratios].[Cash Ratio] as "Cash Ratio", [Financial Ratios].[Quick Ratio] as "Quick Ratio" from [Cube].[{cube_name}]

            user query:-Provide the Year and Total Revenue and Value of Previous Year.
            Expected Response:-select [Time].[Year] as "Year", [Financial Data].[Total Revenue] as "Total Revenue", 
            TrendNumber([Financial Data].[Total Revenue],[Time].[Year],1,'value') as "Trend 1" from [Cube].[{cube_name}]

            user query:-Provide the Quarter wise %Change from Previous year for Total Revenue.
            Expected Response:-select [Time].[Quarter] as "Quarter", [Financial Data].[Total Revenue] as "Total Revenue", TrendNumber([Financial Data].[Total Revenue],[Time].[Quarter],1,'percentage') as "Trend" from [Cube].[{cube_name}]

            user query:-What are the Bottom 4 state based on Total Debit ?
            Expected Response:-select [Branch Details].[State] as "State", [Financial Data].[Total Debit] as "Total Debit",Tail([Branch Details].[State],[Financial Data].[Total Debit],4,undefined) from [Cube].[{cube_name}]

            user query:-Which Mutual Funds have the trade price greater than 272 but less than 276 ?
            Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Bulk Deal Trade].[Trade Price] as "Trade Price" from [Cube].[{cube_name}] where [Bulk Deal Trade].[Trade Price] < 276.00 and [Bulk Deal Trade].[Trade Price] > 272.00

            user query:-What is the Balance Amount for Mutual funds other than HDFC,SBI,Nippon,HSBC,ICICI,IDFC,AXIS ?
            Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[{cube_name}] where [Mutual Fund Investment].[Mutual Fund Name] not in ('SBI','Nippon','HDFC','HSBC','ICICI','IDFC','AXIS')

            user query:-What is the total revenue from operation between March 2015 to September 2018 ?
            Expected Response:-select [Time].[Month] as "Month", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",TimeBetween(20150301,20180930,[Time].[Month], false) from [Cube].[{cube_name}]

            user query:-What is the % of Running Sum of Balance amount with respect to Mutual Fund.
            Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", 
            percentageofrunningsum([Business Drivers].[Balance Amount],'percentagerunningsumacrossrows') as "% of Running sum of Balance Amount" from [Cube].[{cube_name}]
                        
            user query:-Provide the Close price of Index Nifty and Sensex based on Index Name that doesn't contains "Nifty".
            Expected Response:-select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[{cube_name}] where [Benchmark Index Details].[Index Name] not like '%Nifty%'. 

            user query:-Provide the list of months not having the balance average amount between 40,00,00,000 to 2,00,00,00,000.               
            Expected Response:-select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount Average] not between 400000000.00 and 2000000000.00

            user query:-Provide the Month wise Balance Amount with Mutual Fund Name with Balance Amount.
            Expected Response:-select [Time].[Month] as "Month", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[{cube_name}]

            user query:-Provide the Close price of Index Nifty and Sensex based on Index Name that contains "Nifty".
            Expected Response:-select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[{cube_name}] where [Benchmark Index Details].[Index Name] like '%Nifty%' 

            user query:-What is the Month vise Total Revenue and Total Debit between 1st Jan 2014 to 31st Dec 2019 ?
            Expected Response:-select [Time].[Month] as "Month", [Financial Data].[Total Revenue] as "Total Revenue", [Financial Data].[Total Debit] as "Total Debit",TimeBetween(20140101,20191231,[Time].[Month], false) from [Cube].[{cube_name}]

            user query:-Show me the Customer Name and Mutual Fund Name with Balance Amount Greater than 0 and Balance Average Amount with % of Balance Amount.
            Expected Response:-select [Customer Details].[Customer Name] as "Customer Name", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", percentage([Business Drivers].[Balance Amount],'percentColumn') as "% of Balance Amount" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount] > 0.00

            user query:-Kindly provide Region wise Industry wise count of customers with HIGH Risk  for 15 Jan 2025
            Expected Response:-select [Customer Details].[Region] as "Region", [Customer Details].[Industry] as "Industry", [Customer Details].[Count of Customers] as "Count of Customers" from [Cube].[{cube_name}] where [Customer Details].[EWS Tag] = 'HIGH' and [Time].[Day] = '2025-01-15'

            user query:-Provide the Mutual Fund and their Quantity along with Month on Month Quantity and Quarter on Quarter Quantity.
            Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Fund Investment Details].[Mutual Fund Quantity] as "Mutual Fund Quantity", [Fund Investment Details].[Mutual Fund Quantity MoM] as "Mutual Fund Quantity MoM", [Fund Investment Details].[Mutual Fund Quantity QoQ] as "Mutual Fund Quantity QoQ" from [Cube].[{cube_name}]
    
            </examples>

            <final review>
            - ensure if the query has been generated with dimensions and measures extracted only from the current and previous conversation
            - check if functions and filters have been used appropriately in the final cube query, ensure generated query contains filters and functions from given supported functions only
            - review the syntax of the final cube query, refer the examples to help with the review for syntax check, functions and filters usage
            </final review>


            User Query: ####{query}####
            
            $$$$
            Dimensions: {dimensions}
            $$$$

            &&&&
            Measures: {measures}
            &&&&

            Generate a precise single-line Cube query that exactly matches these requirements:"""

            print(Fore.CYAN + '   Generating OLAP cube Query......................\n')
            #output,token_details= self.call_gpt(final_prompt)
            result = self.llm.invoke(final_prompt)
            output = result.content
            token_details = result.response_metadata['token_usage']
            pred_query = self.cleanup_gen_query(output)
            print(f"{pred_query}")
            
            logging.info(f"Generated OLAP Query: {pred_query}")
            return pred_query
        
        except Exception as e:
            logging.error(f"Error generating OLAP query: {e}")
            raise
    
    def cleanup_gen_query(self,pred_query):
        
        pred_query = pred_query.replace("```sql","").replace("\n", "").replace("```","")
        check = pred_query.replace("```","")
        final_query = check.replace("sql","")
        return final_query



class ConversationalQueryGenerator(LLMConfigure):
    def __init__(self, query, dimensions: None, measures: None, llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        self.prev_query = []
        self.prev_dimension = []
        self.prev_measures = []
        self.prev_response = []
        self.max_history = 6

    def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
        """Generate query using conversation context by preserving and extending existing query."""
        try:
            if not dimensions or not measures:
                raise ValueError("Both dimensions and measures are required to generate a query.")
                
            # Get previous conversation elements
            prev_query = prev_conv.get("query", "")
            prev_dims = prev_conv.get("dimensions", "")
            prev_measures = prev_conv.get("measures", "")
            prev_cube_query = prev_conv.get("response", "")

            if not prev_cube_query:  # If no previous query, generate new one
                final_query_generator = FinalQueryGenerator(query, dimensions, measures, self.llm)
                return final_query_generator.generate_query(query, dimensions, measures, prev_conv, cube_name)
                
            final_prompt = f"""You are an SQL Cube query expert tasked with EXTENDING an existing query based on new requirements.

BASE QUERY (This query must be preserved, modified according to new user query): 
{prev_cube_query}

New User query: {query}
New Dimensions: {dimensions}
New Measures: {measures}

CRITICAL INSTRUCTIONS:
1. START with the base query - it must be preserved exactly as is
2. ONLY ADD/Remove corresponding new dimensions/measures that are specifically requested in the new user query
3. DO NOT remove or modify any existing dimensions or measures unless explicitly requested
4. Keep all existing WHERE clauses, filters, and functions from the base query
5. Only add new conditions if specifically mentioned in the new query
6. Maintain the exact same syntax and formatting as the base query
7. Remove all filters when asked, remove all filters like WHERE clause, Head/Tail, IF, Function, percentage, runningsum
8. Use - TrendNumber:- for year-over-year/period comparisons, TimeBetween:- for data between year range, use TrendNumber and TimeBetween both as mentioned in query
9. Check if the Dimensions and measures relate to the new user query, modify the base query for only those new dimensions/measures specifically mentioned in the new user query
10. When asked to remove, only remove what is asked don't add any measures/dimensions.

Functions and filters supported for query generation:-
<functions>

- IF	IF(condition, TRUE Statement, FALSE Statement	IF ( [Measures.Brokerage] = 0 or [Measures.Brokerage] = null, null, [Measures.Total Cost incl rent]  /  [Measures.Brokerage] )	Conditional IF Else calculation on Measures
ISPRESENT	ISPRESENT(Level Name)	IF ( ISPRESENT ( [Branch.Zone] ) ,  [Measures.PROFIT] ,  0) 	"Used as condition in IF function Checks whether level zone is present in report or not, if present then returns true else false"
TRENDNUMBER	"TRENDNUMBER ( measure name,  Time group level ,  number of period,  trend type ) 

- trend type = ('average','value','percentage','sum','delta')	TRENDNUMBER ( [Measures.PROFIT] ,  [Calendar.Year] ,  <number of years>,  'value' ) 	"Trendnumber function is used for performing the lead or lag computations across time dimension.

Type of trends supported are
    - average = average of previous (lag) or next (lead) period
    - value = value of previous (lag) or next (lead) period 
    - percentage = percentage change  of previous (lag) or next (lead) period 
    - sum = sum  of previous (lag) or next (lead) period 
    - delta = change in value between the current and previous (lag) period or next (lead) period
    - PERIODSTODATE	"PERIODSTODATE ( Time dimension level , measure , period type ) ) 

- period type = ('avg','sum')"	PERIODSTODATE ( [Calendar.FiYear] , [Measures.PROFIT] , 'avg' ) ) 	"Used for calculation of Year to Date (YTD) Months to Date (MTD) etc.

- Aggregation supported: sum, avg
FILTERKPI	FILTERKPI ( kpi , condition) 	FILTERKPI ( [Measures.PROFIT] , [Branch Type.Status] = 'ACTIVE') 	"Conditional Aggregation of Measure.
Filters data with the condition mentioned and then applies the measure aggregate as defined in cube
Used for functions like conditional sum, conditional average etc."

- EOP	EOP ( kpi, time dimension level)	EOP ( [Measures.# of Employees] , [Calendar.Year] ) 	As per time dimension level, it will calculate the End of Period
ROUND	ROUND( kpi, number)	ROUND ( [Measures.PROFIT] , 3 )	Rounding Function on the measure
COALESCE	COALESCE( kpi, number)	COALESCE ( [Measures.PROFIT] , 0)	If the value is null, what default value to be taken can be defined using COALESCE function
MAX / SUM / MIN / COUNT / AVG	SUM ( dimension level, 'specific'/'current'/'all', kpi,  dimension level )	SUM ( [Customer.Customer Code] ,  'specific',  [Measures.Exposure],  [Account.Account Number] ) 	"specific = aggregate exposure measure on customer code level for dataset. (Dataset contains the distinct values of exposure according to account number level).

This is customised function and is applicable for measures which are defined at a particular level.
In this example Exposure is always at Customer Code level."

Use below filters on functions based on query requirements:
- TimeBetween:- use this function if any date, month, year range is mentioned in query 
- TrendNumber:- for year-over-year/period comparisons, If require add 2 and more trendnumber in a query 
- Head/Tail:- for top/bottom N queries
- percentage:- Percentage cumulative calculations
- runningsum/percentageofrunningsum:- for cumulative calculations runningsum([<group name>].[<level name>],'sumacrossrows'/'percentagerunningsumacrossrows')
- rank:- it will provide rank for values rank([<group name>].[<level name>],'rankAcrossRows')

</functions>

<examples>

user query:Mutual Fund Name wise Trade Price & Quantity ?
Expected Response:select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Bulk Deal Trade].[Trade Price] as "Trade Price", [Bulk Deal Trade].[Traded Quantity] as "Traded Quantity" from [Cube].[{cube_name}]

user query:What is the Total Revenue and Total Revenue From Operations Between the year 2012 to 2017?
Expected Response:-select [Time].[Year] as "Year", [Financial Data].[Total Revenue] as "Total Revenue", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",TimeBetween(20120101,20171231,[Time].[Year], false) from [Cube].[{cube_name}]

user query:-Which months has the value of balance average amount between 40,00,00,000 to 2,00,00,00,000 ?           
Expected Response:-select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount Average] between 400000000.00 and 2000000000.00

user query:-Top 5 Cities on Average Balance Amount.
Expected Response:-select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average",Head([Branch Details].[City],[Business Drivers].[Balance Amount Average],5,undefined) from [Cube].[{cube_name}]

user query:-Please provide Cities and Average Balance Amount where Average Balance Amount is more than 5000 and Count of Customers more than 10.
Expected Response:-select [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Count of Customers] > 10.00 and [Business Drivers].[Balance Amount Average] > 5000.00

user query:-what is the count of customers based on rating ?
Expected Response:-select [External Funding].[Rating] as "Rating", [Business Drivers].[Count of Customers] as "Count of Customers" from [Cube].[{cube_name}]

user query:-Please provide Year and Average Balance Amount and % change from previous year for past 2 years
Expected Response:-select [Time].[Year] as "Year", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", TrendNumber([Business Drivers].[Balance Amount Average],[Time].[Year],2,'percentage') as "YOY % Change" from [Cube].[{cube_name}]

user query:-What is the closing price of Industry ?
Expected Response:-select [Industry Details].[Industry Name] as "Industry Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[{cube_name}]

user query:-What is the employee count based on Industry Group and Industry?
Expected Response:-select [Industry Details].[Industry Group Name] as "Industry Group Name", [Industry Details].[Industry Name] as "Industry Name", [EPF Mesures].[Employee Count] as "Employee Count" from [Cube].[{cube_name}]

user query:-What are the closing price of AXIS, HDFC, ICICI & LIC Mutual Funds ?
Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Share Price BSE].[Close Price] as "Close Price" from [Cube].[{cube_name}] where [Mutual Fund Investment].[Mutual Fund Name] in ('AXIS','HDFC','ICICI','LIC')

user query:-What are the Cash Ratio Current Ratio and Quick Ration Based on Month from 2012 to 2017 ? 
Expected Response:-select [Time].[Month] as "Month", [Financial Ratios].[Cash Ratio] as "Cash Ratio", [Financial Ratios].[Current Ratio] as "Current Ratio", 
[Financial Ratios].[Quick Ratio] as "Quick Ratio",TimeBetween(20120101,20171231,[Time].[Month], false) from [Cube].[{cube_name}]

user query:-Provide Current Ratio, Cash Ratio, and Quick ration segregated by industry group?
Expected Response:-select [Industry Details].[Industry Group Name] as "Industry Group Name", [Financial Ratios].[Current Ratio] as "Current Ratio", [Financial Ratios].[Cash Ratio] as "Cash Ratio", [Financial Ratios].[Quick Ratio] as "Quick Ratio" from [Cube].[{cube_name}]

user query:-Provide the Year and Total Revenue and Value of Previous Year.
Expected Response:-select [Time].[Year] as "Year", [Financial Data].[Total Revenue] as "Total Revenue", 
TrendNumber([Financial Data].[Total Revenue],[Time].[Year],1,'value') as "Trend 1" from [Cube].[{cube_name}]

user query:-Provide the Quarter wise %Change from Previous year for Total Revenue.
Expected Response:-select [Time].[Quarter] as "Quarter", [Financial Data].[Total Revenue] as "Total Revenue", TrendNumber([Financial Data].[Total Revenue],[Time].[Quarter],1,'percentage') as "Trend" from [Cube].[{cube_name}]

user query:-What are the Bottom 4 state based on Total Debit ?
Expected Response:-select [Branch Details].[State] as "State", [Financial Data].[Total Debit] as "Total Debit",Tail([Branch Details].[State],[Financial Data].[Total Debit],4,undefined) from [Cube].[{cube_name}]

user query:-Which Mutual Funds have the trade price greater than 272 but less than 276 ?
Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Bulk Deal Trade].[Trade Price] as "Trade Price" from [Cube].[{cube_name}] where [Bulk Deal Trade].[Trade Price] < 276.00 and [Bulk Deal Trade].[Trade Price] > 272.00

user query:-What is the Balance Amount for Mutual funds other than HDFC,SBI,Nippon,HSBC,ICICI,IDFC,AXIS ?
Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[{cube_name}] where [Mutual Fund Investment].[Mutual Fund Name] not in ('SBI','Nippon','HDFC','HSBC','ICICI','IDFC','AXIS')

user query:-What is the total revenue from operation between March 2015 to September 2018 ?
Expected Response:-select [Time].[Month] as "Month", [Financial Data].[Total Revenue From Operations] as "Total Revenue From Operations",TimeBetween(20150301,20180930,[Time].[Month], false) from [Cube].[{cube_name}]

user query:-What is the % of Running Sum of Balance amount with respect to Mutual Fund.
Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", 
percentageofrunningsum([Business Drivers].[Balance Amount],'percentagerunningsumacrossrows') as "% of Running sum of Balance Amount" from [Cube].[{cube_name}]
            
user query:-Provide the Close price of Index Nifty and Sensex based on Index Name that doesn't contains "Nifty".
Expected Response:-select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[{cube_name}] where [Benchmark Index Details].[Index Name] not like '%Nifty%'. 

user query:-Provide the list of months not having the balance average amount between 40,00,00,000 to 2,00,00,00,000.               
Expected Response:-select [Time].[Month] as "Month", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount Average] not between 400000000.00 and 2000000000.00

user query:-Provide the Month wise Balance Amount with Mutual Fund Name with Balance Amount.
Expected Response:-select [Time].[Month] as "Month", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount" from [Cube].[{cube_name}]

user query:-Provide the Close price of Index Nifty and Sensex based on Index Name that contains "Nifty".
Expected Response:-select [Benchmark Index Details].[Index Name] as "Index Name", [Benchmark Index].[Index Close Price] as "Index Close Price", [Benchmark Index].[NIFTY 500] as "NIFTY 500", [Benchmark Index].[SENSEX 50] as "SENSEX 50" from [Cube].[{cube_name}] where [Benchmark Index Details].[Index Name] like '%Nifty%' 

user query:-What is the Month vise Total Revenue and Total Debit between 1st Jan 2014 to 31st Dec 2019 ?
Expected Response:-select [Time].[Month] as "Month", [Financial Data].[Total Revenue] as "Total Revenue", [Financial Data].[Total Debit] as "Total Debit",TimeBetween(20140101,20191231,[Time].[Month], false) from [Cube].[{cube_name}]

user query:-Show me the Customer Name and Mutual Fund Name with Balance Amount Greater than 0 and Balance Average Amount with % of Balance Amount.
Expected Response:-select [Customer Details].[Customer Name] as "Customer Name", [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Business Drivers].[Balance Amount] as "Balance Amount", [Business Drivers].[Balance Amount Average] as "Balance Amount Average", percentage([Business Drivers].[Balance Amount],'percentColumn') as "% of Balance Amount" from [Cube].[{cube_name}] where [Business Drivers].[Balance Amount] > 0.00

user query:-Please provide zone and city wise Average Balance Amount.
Expected Response:-select [Branch Details].[Zone] as "Zone", [Branch Details].[City] as "City", [Business Drivers].[Balance Amount Average] as "Balance Amount Average" from [Cube].[{cube_name}]

user query:-Provide the Mutual Fund and their Quantity along with Month on Month Quantity and Quarter on Quarter Quantity.
Expected Response:-select [Mutual Fund Investment].[Mutual Fund Name] as "Mutual Fund Name", [Fund Investment Details].[Mutual Fund Quantity] as "Mutual Fund Quantity", [Fund Investment Details].[Mutual Fund Quantity MoM] as "Mutual Fund Quantity MoM", [Fund Investment Details].[Mutual Fund Quantity QoQ] as "Mutual Fund Quantity QoQ" from [Cube].[{cube_name}]

</examples>
Example Approach:
- If base query has dimensions and measures and new query asks to add or remove some dimension or measure, modify the base query for the corresponding dimension and measure only, rest of the base query needs to be preserved.
- If base query has the above mentioned filters like where, Head,  preserve them and only ADD new ones if specifically asked
- If base query has already time ranges and new query is asking new time range forget the previous time range and remove invalidated funtions.
- If base query has an function and the new query ask for the same function we can have both the function
- Try to pick queries answer from <example> if mention

Your task is to Modify the base query while keeping its existing structure and elements intact.
Return only the modified query without any explanations."""

            result = self.llm.invoke(final_prompt)
            output = result.content
            pred_query = self.cleanup_gen_query(output)
            
            return pred_query

        except Exception as e:
            logging.error(f"Error generating conversational OLAP query: {e}")
            raise

    def cleanup_gen_query(self, pred_query):
        """Clean up the generated query by removing unnecessary formatting."""
        pred_query = pred_query.replace("```sql","").replace("\n", "").replace("```","")
        check = pred_query.replace("```","")
        final_query = check.replace("sql","")
        return final_query
    
class QueryContext:
    """Store context for each query"""
    def __init__(self,
        query: str = None,
        dimensions: Dict=None,
        measures: Dict=None,
        olap_query: str=None,
        timestamp: float =None,
        context_type: str = None,  
        parent_query: Optional[str] = None):
        self.query= query # type: ignore
        self.dimensions= dimensions
        self.measures=measures
        self.olap_query=olap_query
        self.timestamp=timestamp
        self.context_type=context_type 
        self.parent_query=parent_query

class ConversationManager:
    """Manages conversation context and follow-up detection"""
    def __init__(self):
        self.context_window = []  # Stores recent queries with context
        self.max_context_size = 5
        
    def add_context(self, query_context: QueryContext):
        self.context_window.append(query_context)
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def get_recent_context(self) -> List[QueryContext]:
        return self.context_window
    

class OLAPQueryProcessor(LLMConfigure):
    """Enhanced OLAP processor with conversation memory"""
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        try:
            # Initialize other components
            print("config_path in cube_olap=", config_path)
            self.llm_config = LLMConfigure(config_path)
            self.load_json = self.llm_config.load_config(config_path)
            self.llm = self.llm_config.initialize_llm()
            self.embedding = self.llm_config.initialize_embedding()
            self.dim_measure = DimensionMeasure(self.llm, self.embedding, self.load_json)
            self.final = FinalQueryGenerator(query="", dimensions=None, measures=None, llm=self.llm)
            self.query_history = []

        except Exception as e:
            logging.error(f"Error initializing EnhancedOLAPQueryProcessor: {e}")
            raise

    def process_query(self, query: str, cube_id: str, prev_conv: dict, cube_name: str, include_conv: str = "no") -> Tuple[str, str, float]:

        print(f"In process query :- User query : {query} ")
        try:
            cube_dir = os.path.join(vector_db_path, cube_id)
            if not os.path.exists(cube_dir):
                return query, "Cube data doesn,t exists", 0.0, "",""

            start_time = time.time()
            dimensions = self.dim_measure.get_dimensions(query, cube_id, prev_conv)
            measures = self.dim_measure.get_measures(query, cube_id, prev_conv)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures")

            # Choose the appropriate query generator based on include_conv
            if include_conv.lower() == "yes" and prev_conv.get('query'):
                query_generator = ConversationalQueryGenerator(query, dimensions, measures, self.llm)
            else:
                query_generator = FinalQueryGenerator(query, dimensions, measures, self.llm)

            final_query = query_generator.generate_query(query, dimensions, measures, prev_conv, cube_name)
            
            processing_time = time.time() - start_time
            return query, final_query, processing_time, dimensions, measures
            
        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            raise

    def process_query_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str, cube_name: str) -> Tuple[str, str, float, Dict, Dict]:
        """Process a query with error correction."""
        try:
            start_time = time.time()
            # Get corrected dimensions and measures
            dimensions = self.dim_measure.get_dimensions_with_error(query, cube_id, prev_conv, error)
            measures = self.dim_measure.get_measures_with_error(query, cube_id, prev_conv, error)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures after error correction")

            final_query = self.final.generate_query(query, dimensions, measures, prev_conv, cube_name)
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures
        except Exception as e:
            logging.error(f"Error in query processing with error correction: {e}")
            raise

    def process_query_with_feedback(self, query: str, cube_id: str, prev_conv: dict, feedback_type: str, cube_name: str) -> Tuple[str, str, float, Dict, Dict]:
        """Process a query with user feedback consideration."""
        try:
            start_time = time.time()
            # Get refined dimensions and measures based on feedback
            dimensions = self.dim_measure.get_dimensions_with_feedback(query, cube_id, prev_conv, feedback_type)
            measures = self.dim_measure.get_measures_with_feedback(query, cube_id, prev_conv, feedback_type)

            if not dimensions or not measures:
                raise ValueError("Failed to extract dimensions or measures after feedback processing")

            final_query = self.final.generate_query(query, dimensions, measures, prev_conv, cube_name)
            processing_time = time.time() - start_time
            
            return query, final_query, processing_time, dimensions, measures

        except Exception as e:
            logging.error(f"Error in query processing with feedback: {e}")
            raise

def main():
    """Enhanced main function with better conversation handling"""
    setup_logging()
    config_path = "config.json"
    
    processor = OLAPQueryProcessor(config_path)
    
    print(Fore.CYAN + "\n=== OLAP Query Conversation System ===")
    print(Fore.CYAN + "Type 'exit' to end the conversation.\n")
    
    while True:
        try:
            query = input(Fore.GREEN + "Please enter your query: ")
            
            if query.lower() == 'exit':
                print(Fore.YELLOW + "\nThank you for using the OLAP Query System! Goodbye!")
                break
            
            # Process query with enhanced context handling
            original_query, final_query, processing_time,dimensions,measures = processor.process_query(query)            
            print(Fore.CYAN + f"\nProcessing time: {processing_time:.2f} seconds\n")
            print()  # Add spacing for readability

 
                               
        except Exception as e:
            logging.error(f"Error in conversation: {e}")
            print(Fore.RED + f"\nI encountered an error: {str(e)}")
            print(Fore.YELLOW + "Please try rephrasing your question or ask something else.\n")
            continue

if __name__ == "__main__":
    main()




import multiprocessing
import pickle
import shutil
import tempfile
from fastapi import APIRouter,FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import jwt
from langchain_community.vectorstores.chroma import Chroma
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal
import json
import os
from datetime import datetime
import logging
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
import asyncio
from pathlib import Path
import uvicorn
from cube_query_v3 import OLAPQueryProcessor
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever 
import nltk 
import pickle
import os


# Initialize FastAPI app
app = FastAPI(title="OLAP Cube Management API")
router = APIRouter()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserFeedbackRequest(BaseModel):
    user_feedback: str
    feedback: Literal["accepted", "rejected"]
    cube_query: str
    cube_id: str
    cube_name: str

class UserFeedbackResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None


class CubeErrorRequest(BaseModel):
    user_query: str
    cube_id: str
    error_message: str
    cube_name: str

# class QueryRequest(BaseModel):
#     user_query: str
#     cube_id: int

class QueryResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None
    dimensions: str
    measures: str

class CubeDetailsRequest(BaseModel):
    cube_json_dim: List[Dict]
    cube_json_msr: List[Dict]
    cube_id: str

class CubeQueryRequest(BaseModel):
    user_query: str
    cube_id: str
    cube_name: str
    include_conv: Optional[str] = "no" 

class CubeErrorResponse(BaseModel):
    message: str
    cube_query: Optional[str] = None

class CubeDetailsResponse(BaseModel):
    message: str

class ClearChatRequest(BaseModel):
    cube_id: str

class ClearChatResponse(BaseModel):
    status: str


# Configuration and storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
history_file = os.path.join(BASE_DIR, "conversation_history.json")
vector_db_path = os.path.join(BASE_DIR, "vector_db")
config_file = os.path.join(BASE_DIR, "config.json")

def save_documents_to_json(documents: List[dict], cube_id: str, doc_type: str, base_dir: str) -> None:
    """Save documents to JSON file"""
    try:
        # Create directory if it doesn't exist
        cube_dir = os.path.join(base_dir, cube_id)
        os.makedirs(cube_dir, exist_ok=True)
        
        # Save to JSON file
        file_path = os.path.join(cube_dir, f"{cube_id}_{doc_type}.json")
        with open(file_path, 'w') as f:
            json.dump(documents, f, indent=2)            
    except Exception as e:
        logging.error(f"Error saving {doc_type} documents: {str(e)}")
        raise


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='olap_main_api.log'
)
def format_measure_documents(measure_json: List[dict]) -> List[Document]:
    """Convert measure JSON data into Langchain Document objects"""
    measure_texts = []
    for measure in measure_json:
        text = (f"Group Name:{measure['Group Name']}--"
                f"Level Name:{measure['Level Name']}--"
                f"Description:{measure['Description']}")
        
        doc = Document(
            page_content=text,
            metadata={
                "group_name": measure['Group Name'],
                "level_name": measure['Level Name']
            }
        )
        measure_texts.append(doc)
    return measure_texts

def save_bm25_documents(documents: List[Document], cube_id: str, base_path: str):
    """Save documents for BM25 retrieval"""
    cube_dir = os.path.join(base_path, cube_id)
    os.makedirs(cube_dir, exist_ok=True)
    
    bm25_file = os.path.join(cube_dir, f"{cube_id}_measures.pkl")
    with open(bm25_file, 'wb') as f:
        pickle.dump(documents, f)

class LLMConfigure:
    """
    Class responsible for loading and configuring LLM and embedding models from a config file.
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise

    def initialize_llm(self):
        """Initializes and returns the LLM model."""
        try:
            # Simulate LLM initialization using the config
            #self.llm = self.config['llm']
            self.llm = AzureChatOpenAI(openai_api_key= self.config['llm']['OPENAI_API_KEY'],
                                      model=self.config['llm']['model'],
                                      temperature=self.config['llm']['temperature'],
                                      api_version= self.config['llm']["OPENAI_API_VERSION"],
                                      azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      seed=self.config['llm']["seed"]
            )
            return self.llm
        except KeyError as e:
            logging.error(f"Missing LLM configuration in config file: {e}")
            raise

    def initialize_embedding(self):
        """Initializes and returns the Embedding model."""
        try:
            #embedding initialization using the config
            self.embedding = AzureOpenAIEmbeddings(deployment = self.config['embedding']['deployment'],
                                      azure_endpoint = self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                                      openai_api_key = self.config['llm']['OPENAI_API_KEY'],
                                      show_progress_bar = self.config['embedding']['show_progress_bar'],
                                      disallowed_special = (),
                                      openai_api_type = self.config['llm']['OPENAI_API_TYPE']
                          
                        )
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise

llm_config = LLMConfigure(config_file)
llm = llm_config.initialize_llm()
embedding = llm_config.initialize_embedding()

class History:
    def __init__(self, history_file: str = history_file):
        self.history_file = history_file        
        try:
            # Create file if it doesn't exist
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w') as f:
                    json.dump({"users": {}}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to create history file: {str(e)}")
            raise
        
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)                    
                    # Migrate old format to new format if needed
                    if "users" not in data:
                        migrated_data = {"users": {}}
                        for key, value in data.items():
                            if key != "cube_id":  # Skip the old top-level cube_id
                                migrated_data["users"][key] = {}
                                if isinstance(value, list):
                                    old_cube_id = data.get("cube_id", "")
                                    migrated_data["users"][key][old_cube_id] = value
                        return migrated_data
                    
                    return data
            return {"users": {}}
        except Exception as e:
            logging.error(f"Error loading conversation history: {str(e)}")
            return {"users": {}}

    def save(self, history: Dict):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Verify the save
            with open(self.history_file, 'r') as f:
                saved_data = json.load(f)
                
        except Exception as e:
            logging.error(f"Error saving conversation history: {str(e)}")
            raise

    def update(self, user_id: str, query_data: Dict, cube_id: str, cube_name: str = None):
        try:
            
            # Initialize nested structure if needed
            if "users" not in self.history:
                self.history["users"] = {}
                
            if user_id not in self.history["users"]:
                self.history["users"][user_id] = {}
            
            if cube_id not in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
            
            # Add new conversation without cube_id inside the block
            new_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": query_data["query"],
                "dimensions": query_data["dimensions"],
                "measures": query_data["measures"],
                "response": query_data["response"]
            }
            
            # Add cube_name if provided
            if cube_name:
                new_conversation["cube_name"] = cube_name
            
            self.history["users"][user_id][cube_id].append(new_conversation)
            # Keep last 5 conversations for this user and cube
            self.history["users"][user_id][cube_id] = self.history["users"][user_id][cube_id][-5:]
            self.save(self.history)
        
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
            raise
    
    def retrieve(self, user_id: str, cube_id: str):
        """Retrieve the most recent conversation for this user and cube"""
        try:            
            # Check if we have history for this user and cube
            if "users" not in self.history:
                self.history["users"] = {}
            
            if user_id not in self.history["users"]:
                self.history["users"][user_id] = {}
                return self._empty_conversation(cube_id)
            
            if cube_id not in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
                return self._empty_conversation(cube_id)
            
            try:
                conversations = self.history["users"][user_id][cube_id]
                if not conversations:
                    return self._empty_conversation(cube_id)
                    
                last_conversation = conversations[-1]
                return last_conversation
            except IndexError:
                logging.info("No conversations found in history")
                return self._empty_conversation(cube_id)
        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            return self._empty_conversation(cube_id)
    
    def _empty_conversation(self, cube_id: str, cube_name: str = None):
        """Helper method to return an empty conversation structure"""
        empty_conv = {
            "timestamp": datetime.now().isoformat(),
            "query": "",
            "dimensions": "",
            "measures": "",
            "response": ""
        }
        
        if cube_name:
            empty_conv["cube_name"] = cube_name
            
        return empty_conv
    
    def clear_history(self, user_id: str, cube_id: str):
        """Clear conversation history for a specific user and cube"""
        try:
            if "users" in self.history and user_id in self.history["users"] and cube_id in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
                self.save(self.history)
                return True
            return False
        except Exception as e:
            logging.error(f"Error clearing history: {str(e)}")
            return False



class ImportHistory:
    def __init__(self, history_file: str = IMPORT_HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading import history: {e}")
            return {}
    
    def save(self, history: Dict):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving import history: {e}")

    def update(self, user_id: str, cube_id: str, status: str):
        if user_id not in self.history:
            self.history[user_id] = []

        new_import = {
            "timestamp": datetime.now().isoformat(),
            "cube_id": cube_id,
            "status": status
        }

        self.history[user_id].append(new_import)
        self.history[user_id] = self.history[user_id][-5:]
        self.save(self.history)

# Token verification
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, options={"verify_signature": False})
        user_details = payload.get("preferred_username")
        if not user_details:
            raise ValueError("No user details in token")
        
        return user_details
    except Exception as e:
        logging.error(f"Token verification failed: {e}")
        #raise HTTPException(status_code=401, detail="Invalid token")

# Initialize OLAP processor dictionary
olap_processors = {}

async def process_query(user_query: str, cube_id: str, user_id: str, cube_name="Credit One View", include_conv="no") -> Dict:
    try:
        # Get or create processor for this user
        with open(r'conversation_history.json') as conv_file: 
            conv_json = json.load(conv_file)
            
            # Initialize user structure if needed
            if "users" not in conv_json:
                conv_json["users"] = {}
                
            if conv_json.get("users", {}).get(user_id) is None:
                print("Initializing user in conversation history")
                if "users" not in conv_json:
                    conv_json["users"] = {}
                conv_json["users"][user_id] = {}
                with open(r'conversation_history.json', 'w') as conv_file:
                    json.dump(conv_json, conv_file)

        # Create empty previous conversation context
        if include_conv.lower() == "no":
            prev_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": "",
                "dimensions": "",
                "measures": "",
                "response": "",
                "cube_id": cube_id,
                "cube_name": cube_name
            }
        else:
            # Get existing conversation history for this cube
            history_manager = History()
            prev_conversation = history_manager.retrieve(user_id, cube_id)
            # Ensure cube_name is set (it might be missing in older records)
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = cube_name

        # Get processor for this user
        olap_processors[user_id] = OLAPQueryProcessor(config_file)
        processor = olap_processors[user_id]

        # Process query and get results
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            user_query, cube_id, prev_conversation, cube_name, include_conv
        )
        
        # Prepare response data
        response_data = {
            "query": query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,
        }

        # Only update history if include_conv is "yes"
        if include_conv.lower() == "yes":
            history_manager = History()
            history_manager.update(user_id, response_data, cube_id, cube_name)

        return {
            "message": "success",
            "cube_query": final_query,
            "dimensions": dimensions,
            "measures": measures
        }
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return {
            "message": f"failure{e}",
            "cube_query": None,
            "dimensions": "",
            "measures": ""
        }

async def process_cube_details(cube_json_dim, cube_json_msr, cube_id: str) -> Dict:
    try:
        # Define paths
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        # IMPORTANT: Create a completely fresh directory approach
        # Instead of trying to modify the existing directory,
        # we'll create everything in a temporary location first
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectories
            temp_cube_dir = os.path.join(temp_dir, cube_id)
            temp_dim_dir = os.path.join(temp_cube_dir, "dimensions")
            temp_msr_dir = os.path.join(temp_cube_dir, "measures")
            
            os.makedirs(temp_cube_dir, exist_ok=True)
            os.makedirs(temp_dim_dir, exist_ok=True)
            os.makedirs(temp_msr_dir, exist_ok=True)
            
            # Save dimension and measure JSON documents to temp location
            temp_dim_file = os.path.join(temp_cube_dir, f"{cube_id}_dimensions.json")
            with open(temp_dim_file, 'w', encoding='utf-8') as f:
                json.dump(cube_json_dim, f, indent=2)
                
            temp_msr_file = os.path.join(temp_cube_dir, f"{cube_id}_measures.json")
            with open(temp_msr_file, 'w', encoding='utf-8') as f:
                json.dump(cube_json_msr, f, indent=2)
            
            # Format measure documents
            measure_docs = format_measure_documents(cube_json_msr)
            
            # Save BM25 documents
            temp_bm25_file = os.path.join(temp_cube_dir, f"{cube_id}_measures.pkl")
            with open(temp_bm25_file, 'wb') as f:
                pickle.dump(measure_docs, f)
            
            # Process documents for vector stores
            cube_str_dim = [f"Group Name:{d['Group Name']}--Level Name:{d['Level Name']}--Description:{d['Description']}" for d in cube_json_dim]
            text_list_dim = [Document(i) for i in cube_str_dim]
            
            cube_str_msr = [f"Group Name:{d['Group Name']}--Level Name:{d['Level Name']}--Description:{d['Description']}" for d in cube_json_msr]
            text_list_msr = [Document(i) for i in cube_str_msr]
            
            # Create vector stores in temporary location
            vectordb_dim = Chroma.from_documents(
                documents=text_list_dim,
                embedding=embedding,
                persist_directory=temp_dim_dir
            )
            
            vectordb_msr = Chroma.from_documents(
                documents=text_list_msr,
                embedding=embedding,
                persist_directory=temp_msr_dir
            )
            
            # Ensure the vector stores are properly persisted
            vectordb_dim.persist()
            vectordb_msr.persist()
            
            # Now, delete the existing cube directory (if exists)
            if os.path.exists(cube_dir):
                logging.info(f"Deleting existing cube directory: {cube_dir}")
                shutil.rmtree(cube_dir, ignore_errors=True)
            
            # Create the target directory
            os.makedirs(os.path.join(vector_db_path, cube_id), exist_ok=True)
            
            # Copy the successfully created content from temp to actual location
            shutil.copytree(temp_cube_dir, cube_dir, dirs_exist_ok=True)
            
        # Verify the transfer was successful
        if os.path.exists(os.path.join(cube_dir, f"{cube_id}_dimensions.json")) and \
           os.path.exists(os.path.join(cube_dir, f"{cube_id}_measures.json")):
            logging.info(f"Successfully processed cube details for cube_id: {cube_id}")
            return {"message": "success"}
        else:
            raise Exception("Failed to verify the copied files")
            
    except Exception as e:
        logging.error(f"Error processing cube details: {e}")
        return {"message": f"failure:{e}"}

@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: CubeQueryRequest, user_details: str = Depends(verify_token)):
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        if not os.path.exists(cube_dir):
            return QueryResponse(
                message="failure",
                cube_query="Cube data doesn't exist",
                dimensions="",
                measures=""
            )
            
        user_id = f"user_{user_details}"
        
        # Initialize History manager here
        history_manager = History()
        
        if request.include_conv.lower() == "no":
            prev_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": "",
                "dimensions": "",
                "measures": "",
                "response": "",
                "cube_id": cube_id,
                "cube_name": request.cube_name
            }
        else:
            # Get history specific to this cube
            prev_conversation = history_manager.retrieve(user_id, request.cube_id)
            # Ensure cube_name is set
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = request.cube_name

        # Process using OLAP processor
        olap_processors[user_id] = OLAPQueryProcessor(config_file)
        processor = olap_processors[user_id]
        
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.cube_name,
            request.include_conv
        )
        
        # Update history
        response_data = {
            "query": request.user_query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query
        }
        
        # Always update history now
        history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
        
        return QueryResponse(
            message="success",
            cube_query=final_query,
            dimensions=dimensions,
            measures=measures
        )
    
    except HTTPException as he:
        logging.error(f"HTTP Exception in generate_cube_query: {str(he)}")
        return QueryResponse(
            message="failure", 
            cube_query=f"{he}",
            dimensions="",
            measures=""
        )
    except Exception as e:
        logging.error(f"Error in generate_cube_query: {str(e)}")
        return QueryResponse(
            message=f"failure", 
            cube_query=f"{e}",
            dimensions="",
            measures=""
        )


@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(request: CubeDetailsRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        print("user name:{}".format(user_details))
        print("request json:{}".format(request.cube_json_dim))
        result = await process_cube_details(
            request.cube_json_dim,
            request.cube_json_msr,
            request.cube_id
        )
        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        return CubeDetailsResponse(message=f"failure:{he}")
    except Exception as e:
        logging.error(f"Error in import_cube_details: {e}")
        return CubeDetailsResponse(message=f"failure:{e}")

@app.post("/genai/clear_chat", response_model=ClearChatResponse)
async def clear_chat(request: ClearChatRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        
        # Load the existing conversation history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Check if the users structure exists
            if "users" in history_data and user_id in history_data["users"]:
                # Check if this cube_id exists for this user
                if request.cube_id in history_data["users"][user_id]:
                    # Clear the conversations for this cube
                    history_data["users"][user_id][request.cube_id] = []
                    
                    # Save the updated history
                    with open(history_file, 'w') as f:
                        json.dump(history_data, f, indent=4)
                    
                    return ClearChatResponse(status="success")
            
            return ClearChatResponse(status="no matching cube_id found")
        else:
            return ClearChatResponse(status="no matching cube_id found")
    
    except Exception as e:
        logging.error(f"Error in clear_chat: {e}")
        return ClearChatResponse(status=f"failure: {str(e)}")


@app.post("/genai/cube_error_injection", response_model=CubeErrorResponse)
async def handle_cube_error(request: CubeErrorRequest, user_details: str = Depends(verify_token)):
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        if os.path.exists(cube_dir): 
            user_id = f"user_{user_details}"
            history_manager = History()
            prev_conversation = history_manager.retrieve(user_id)
            
            query, final_query, processing_time, dimensions, measures = OLAPQueryProcessor(config_file).process_query_with_error(
                request.user_query,
                request.cube_id,
                prev_conversation,
                request.error_message,
                request.cube_name
            )
            response_data = {
                "query": query,
                "dimensions": dimensions,
                "measures": measures,
                "response": final_query,
            }
            
            history_manager.update(user_id, response_data)
            
            return CubeErrorResponse(
                message="success",
                cube_query=final_query,
                error_details={
                    "original_error": request.error_message,
                    "correction_timestamp": datetime.now().isoformat()
                }
            )
        else:
            return QueryResponse(
                message="failure",
                cube_query="Cube data doesn't exists"
            )
    except Exception as e:
        logging.error(f"Error in handle_cube_error: {e}")
        return CubeErrorResponse(
            message="failure",
            cube_query=None,
            error_details={"error_type": "processing_error", "details": str(e)}
        )


@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(
    request: UserFeedbackRequest,
    user_details: str = Depends(verify_token)
):
    """Handle user feedback for cube queries"""
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        if os.path.exists(cube_dir): 

            user_id = f"user_{user_details}"
        
            if request.feedback == "rejected":
                # Get or create processor
                if user_id not in olap_processors:
                    olap_processors[user_id] = OLAPQueryProcessor(config_file)
                
                processor = olap_processors[user_id]
                history_manager = History()
                prev_conv = history_manager.retrieve(user_id)
                
                # Add feedback to context
                prev_conv["user_feedback"] = request.user_feedback
                prev_conv["feedback_query"] = request.cube_query
                
                # Process query with feedback context
                query, final_query, _, dimensions, measures = processor.process_query(
                    request.user_feedback,
                    request.cube_id, 
                    prev_conv,
                    request.cube_name
                )
                
                # Update history
                response_data = {
                    "query": request.user_feedback,
                    "dimensions": dimensions,
                    "measures": measures,
                    "response": final_query
                }
                history_manager.update(user_id, response_data)
                
                return UserFeedbackResponse(
                    message="success",
                    cube_query=final_query
                )
                
            return UserFeedbackResponse(
                message="success", 
                cube_query="None"
            )
        else:
            return QueryResponse(
                message="failure",
                cube_query="Cube data doesn't exists"
            )
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return UserFeedbackResponse(
            message="failure",
            cube_query=None
        )
    
# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        os.makedirs(CUBE_DETAILS_DIR, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        os.makedirs(os.path.dirname(IMPORT_HISTORY_FILE), exist_ok=True)

        for file in [IMPORT_HISTORY_FILE, history_file]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump({}, f)
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        raise

app.include_router(router)

if __name__ == "__main__":
    num_cores =multiprocessing.cpu_count()
    optimal_workers = 2* num_cores + 1
    uvicorn.run("olap_details_generat:app", host="172.26.150.165", port=8085, reload=True, workers = optimal_workers)
