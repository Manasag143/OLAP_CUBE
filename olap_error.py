# REMOVE THESE IMPORTS at the top of the file:
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

# CHANGE 1: In get_dimensions method of DimensionMeasure class
def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extracts dimensions from the query."""
    try:
        with get_openai_callback() as dim_cb:
            # ... (prompt template remains the same) ...
            
            print(Fore.RED + '    Identifying Dimensions group name and level name......................\n')
            
            # REMOVE THESE LINES:
            # documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
            # bm25_retriever = BM25Retriever.from_documents(documents, k=10)
            
            # Set up vector store directory
            cube_dir = os.path.join(vector_db_path, cube_id)
            cube_dim = os.path.join(cube_dir, "dimensions")
            
            load_embedding_dim = Chroma(persist_directory=cube_dim, embedding_function=self.embedding)
            vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})

            # REMOVE THESE LINES:
            # ensemble_retriever = EnsembleRetriever(
            #     retrievers=[bm25_retriever, vector_retriever],
            #     weights=[0.5, 0.5]
            # )
            
            # CHANGE: Use vector_retriever instead of ensemble_retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=vector_retriever,  # CHANGED from ensemble_retriever
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=query_dim,
                        input_variables=["question", "context"]  # CHANGED from ["query", "context"]
                    ),
                    "verbose": True
                }
            )

            # Get results
            result = qa_chain({"query": query})  # REMOVED context parameter
            dim = result['result']
            print(Fore.GREEN + '    Identified Group and level name :        ' + str(dim))
            logging.info(f"Extracted dimensions :\n {dim}")
            return dim

    except Exception as e:
        logging.error(f"Error extracting dimensions : {e}")
        raise

# CHANGE 2: In get_measures method of DimensionMeasure class
def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extracts measures from the query."""
    try:
        with get_openai_callback() as msr_cb:
            # ... (prompt template remains the same) ...

            print(Fore.RED + '    Identifying Measure group name and level name......................\n')
            
            # REMOVE THESE LINES:
            # documents = load_documents_from_json(cube_id, "measures", vector_db_path)
            # bm25_retriever = BM25Retriever.from_documents(documents, k=10)

            cube_msr = os.path.join(vector_db_path, cube_id, "measures")
            load_embedding_msr = Chroma(persist_directory=cube_msr, embedding_function=self.embedding)
            vector_retriever = load_embedding_msr.as_retriever(search_type="similarity", search_kwargs={"k": 20})
            
            # REMOVE THESE LINES:
            # ensemble_retriever = EnsembleRetriever(
            #     retrievers=[bm25_retriever, vector_retriever],
            #     weights=[0.5, 0.5]
            # )

            # CHANGE: Use vector_retriever instead of ensemble_retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=vector_retriever,  # CHANGED from ensemble_retriever
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=query_msr,
                        input_variables=["question", "context"]  # CHANGED from ["query", "context"]
                    ),
                    "verbose": True
                }
            )
            
            result = qa_chain({"query": query})  # REMOVED context parameter
            msr = result['result']

            print(Fore.GREEN + '    Measures result :        ' + str(result)) 
            logging.info(f"Extracted measures :\n {msr}")  
            return msr
    
    except Exception as e:
        logging.error(f"Error Extracting Measure: {e}")
        raise
