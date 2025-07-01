import logging
import time
import json
import os
import requests
from datetime import date
from colorama import Fore, Style, init
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
history_file = os.path.join(BASE_DIR, "conversation_history.json")
vector_db_path = os.path.join(BASE_DIR, "vector_db")
config_file = os.path.join(BASE_DIR, "config.json")

def load_documents_from_json(cube_id: str, doc_type: str, base_dir: str) -> List[Document]:
    """Load documents from JSON file"""
    try:
        file_path = os.path.join(base_dir, cube_id, f"{cube_id}_{doc_type}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cube data doesn't exist")
            
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

def setup_logging():
    """Store errors in log folder datewise and token consumptions."""
    today = date.today()
    log_folder = './log'
  
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    logging.basicConfig(
        filename=f"{log_folder}/{today}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class LLMConfigure:
    """Class responsible for loading and configuring LLM and embedding models from a config file."""

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
            self.llm = AzureChatOpenAI(
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                model=self.config['llm']['model'],
                temperature=self.config['llm']['temperature'],
                api_version=self.config['llm']["OPENAI_API_VERSION"],
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
            self.embedding = AzureOpenAIEmbeddings(
                deployment=self.config['embedding']['deployment'],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                show_progress_bar=self.config['embedding']['show_progress_bar'],
                disallowed_special=(),
                openai_api_type=self.config['llm']['OPENAI_API_TYPE']
            )
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise

class DimensionMeasure:
    """Class responsible for extracting dimensions and measures from the natural language query."""

    def __init__(self, llm: str, embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore

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
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
                
                # Initialize BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)
                    
                # Set up vector store directory
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                
                load_embedding_dim = Chroma(persist_directory=cube_dim, embedding_function=self.embedding)
                vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})

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
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "measures", vector_db_path)
                
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)

                cube_msr = os.path.join(vector_db_path, cube_id, "measures")
                load_embedding_msr = Chroma(persist_directory=cube_msr, embedding_function=self.embedding)
                vector_retriever = load_embedding_msr.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                
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
            logging.error(f"Error Extracting Measure: {e}")
            raise

class FinalQueryGenerator(LLMConfigure):
    """Class responsible for generating the final OLAP query based on dimensions and measures."""
    
    def __init__(self, query, dimensions: None, measures: None, llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        
    def call_gpt(self, final_prompt: str):
        """Function responsible for generating final query"""
        API_KEY = self.config['llm']['OPENAI_API_KEY']
        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }
        
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
        
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
        
        output = response.json()
        token_details = output['usage']
        output = output["choices"][0]["message"]["content"]
        return output, token_details

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

            User Query: ####{query}####
            
            $$$$
            Dimensions: {dimensions}
            $$$$

            &&&&
            Measures: {measures}
            &&&&

            Generate a precise single-line Cube query that exactly matches these requirements:"""

            print(Fore.CYAN + '   Generating OLAP cube Query......................\n')
            result = self.llm.invoke(final_prompt)
            output = result.content
            pred_query = self.cleanup_gen_query(output)
            print(f"{pred_query}")
            
            logging.info(f"Generated OLAP Query: {pred_query}")
            return pred_query
        
        except Exception as e:
            logging.error(f"Error generating OLAP query: {e}")
            raise
    
    def cleanup_gen_query(self, pred_query):
        pred_query = pred_query.replace("```sql", "").replace("\n", "").replace("```", "")
        check = pred_query.replace("```", "")
        final_query = check.replace("sql", "")
        return final_query

class ConversationalQueryGenerator(LLMConfigure):
    def __init__(self, query, dimensions: None, measures: None, llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm

    def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
        """Generate query using conversation context by preserving and extending existing query."""
        try:
            if not dimensions or not measures:
                raise ValueError("Both dimensions and measures are required to generate a query.")
                
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
        pred_query = pred_query.replace("```sql", "").replace("\n", "").replace("```", "")
        check = pred_query.replace("```", "")
        final_query = check.replace("sql", "")
        return final_query

class OLAPQueryProcessor(LLMConfigure):
    """Enhanced OLAP processor with conversation memory"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        try:
            self.llm_config = LLMConfigure(config_path)
            self.load_json = self.llm_config.load_config(config_path)
            self.llm = self.llm_config.initialize_llm()
            self.embedding = self.llm_config.initialize_embedding()
            self.dim_measure = DimensionMeasure(self.llm, self.embedding, self.load_json)
            self.final = FinalQueryGenerator(query="", dimensions=None, measures=None, llm=self.llm)

        except Exception as e:
            logging.error(f"Error initializing OLAPQueryProcessor: {e}")
            raise

    def process_query(self, query: str, cube_id: str, prev_conv: dict, cube_name: str, include_conv: str = "no") -> Tuple[str, str, float]:
        try:
            cube_dir = os.path.join(vector_db_path, cube_id)
            if not os.path.exists(cube_dir):
                return query, "Cube data doesn't exist", 0.0, "", ""

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
            original_query, final_query, processing_time, dimensions, measures = processor.process_query(query)            
            print(Fore.CYAN + f"\nProcessing time: {processing_time:.2f} seconds\n")
                               
        except Exception as e:
            logging.error(f"Error in conversation: {e}")
            print(Fore.RED + f"\nI encountered an error: {str(e)}")
            print(Fore.YELLOW + "Please try rephrasing your question or ask something else.\n")
            continue

if __name__ == "__main__":
    main()

















import multiprocessing
import shutil
import tempfile
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Literal

import jwt
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document

from cube_query_v3 import OLAPQueryProcessor

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

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='olap_main_api.log'
)

class LLMConfigure:
    """Class responsible for loading and configuring LLM and embedding models from a config file."""

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
            self.llm = AzureChatOpenAI(
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                model=self.config['llm']['model'],
                temperature=self.config['llm']['temperature'],
                api_version=self.config['llm']["OPENAI_API_VERSION"],
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
            self.embedding = AzureOpenAIEmbeddings(
                deployment=self.config['embedding']['deployment'],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                show_progress_bar=self.config['embedding']['show_progress_bar'],
                disallowed_special=(),
                openai_api_type=self.config['llm']['OPENAI_API_TYPE']
            )
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise

# Initialize LLM and embedding
llm_config = LLMConfigure(config_file)
llm = llm_config.initialize_llm()
embedding = llm_config.initialize_embedding()

class History:
    def __init__(self, history_file: str = history_file):
        self.history_file = history_file        
        try:
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
                    if "users" not in data:
                        migrated_data = {"users": {}}
                        for key, value in data.items():
                            if key != "cube_id":
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
        except Exception as e:
            logging.error(f"Error saving conversation history: {str(e)}")
            raise

    def update(self, user_id: str, query_data: Dict, cube_id: str, cube_name: str = None):
        try:
            if "users" not in self.history:
                self.history["users"] = {}
                
            if user_id not in self.history["users"]:
                self.history["users"][user_id] = {}
            
            if cube_id not in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
            
            new_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": query_data["query"],
                "dimensions": query_data["dimensions"],
                "measures": query_data["measures"],
                "response": query_data["response"]
            }
            
            if cube_name:
                new_conversation["cube_name"] = cube_name
            
            self.history["users"][user_id][cube_id].append(new_conversation)
            self.history["users"][user_id][cube_id] = self.history["users"][user_id][cube_id][-5:]
            self.save(self.history)
        
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
            raise
    
    def retrieve(self, user_id: str, cube_id: str):
        """Retrieve the most recent conversation for this user and cube"""
        try:            
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
        # Uncomment the line below if you want to enforce token verification
        # raise HTTPException(status_code=401, detail="Invalid token")

# Initialize OLAP processor dictionary
olap_processors = {}

async def process_cube_details(cube_json_dim, cube_json_msr, cube_id: str) -> Dict:
    try:
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
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
            
            # Delete the existing cube directory (if exists)
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
        
        # Initialize History manager
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
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            if "users" in history_data and user_id in history_data["users"]:
                if request.cube_id in history_data["users"][user_id]:
                    history_data["users"][user_id][request.cube_id] = []
                    
                    with open(history_file, 'w') as f:
                        json.dump(history_data, f, indent=4)
                    
                    return ClearChatResponse(status="success")
            
            return ClearChatResponse(status="no matching cube_id found")
        else:
            return ClearChatResponse(status="no matching cube_id found")
    
    except Exception as e:
        logging.error(f"Error in clear_chat: {e}")
        return ClearChatResponse(status=f"failure: {str(e)}")

@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(
    request: UserFeedbackRequest,
    user_details: str = Depends(verify_token)
):
    """Handle user feedback for cube queries"""
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        if not os.path.exists(cube_dir):
            return UserFeedbackResponse(
                message="failure",
                cube_query="Cube data doesn't exist"
            )

        user_id = f"user_{user_details}"
        
        if request.feedback == "rejected":
            if user_id not in olap_processors:
                olap_processors[user_id] = OLAPQueryProcessor(config_file)
            
            processor = olap_processors[user_id]
            history_manager = History()
            prev_conv = history_manager.retrieve(user_id, cube_id)
            
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
            history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
            
            return UserFeedbackResponse(
                message="success",
                cube_query=final_query
            )
            
        return UserFeedbackResponse(
            message="success", 
            cube_query="None"
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
    num_cores = multiprocessing.cpu_count()
    optimal_workers = 2 * num_cores + 1
    uvicorn.run("olap_details_generat:app", host="172.26.150.165", port=8085, reload=True, workers=optimal_workers)
