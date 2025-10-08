#Document comparison agent
"""
Pipelined - Implemented on open web ui for Document Comparison -> Identify contradictions

1. Add two files in to the chat
2. Point out the contradictions between the two documents
"""

from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from typing import List
from typing import List, Union, Generator, Iterator
from typing import ClassVar
import os
import re
from typing import List
from pydantic import BaseModel, Field
import requests

from langchain_openai import ChatOpenAI #langgraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver #as  the memmory is only runtime use sqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler


from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import os
from sentence_transformers import SentenceTransformer, util,  SimilarityFunction
from sentence_transformers import CrossEncoder
from sentence_transformers import SparseEncoder

from sklearn.cluster import AgglomerativeClustering

from sentence_transformers import SentenceTransformer
from langchain.prompts.chat import ChatPromptTemplate
import torch
 
from langchain_core.documents import Document

import csv


# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


def compare_documents(extracted_file_contents: list):
    print("---------STRATED---------")

    
    # raw_text_1 = extracted_file_contents[0][0]
    # raw_text_2 = extracted_file_contents[0][1]

    # doc_1 = Document(page_content=raw_text_1, metadata={"source": "manual_input_1"})
    # doc_2 = Document(page_content=raw_text_2, metadata={"source": "manual_input_2", "author": "User"})

    # documents = [doc_1, doc_2]

    #Initialize the llm as openai based api from ollama cloud models
    llm = ChatOpenAI(
        model="gpt-oss:120b",
        api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
        base_url="https://ollama.com/v1",
        # temperature=0,
        # max_tokens=None,
        # timeout=None,
        # max_retries=2,
        # organization="...",
        # other params...
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("---------LLM/Embeddings Initialized---------")
    #Get the documents
    folder_path = "/home/tharaka/thakshana/doc_comparison/data"  # folder containing all your PDFs
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            all_docs.append(docs)  # append all pages to the main list
    
    # folder_path = "/home/tharaka/thakshana/doc_comparison/data"  # folder containing all your PDFs
    # all_docs = []

    # all_docs = documents

    print(f"Loaded a total of {len(all_docs)} documents ")
    print("---------Document Extracted---------")

    # Split
    splits_dict={}
    for i, doc in enumerate(all_docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)
        splits_dict[f"doc {i}"] = splits
    print(len(splits_dict))
    #splits_dict -> {doc 1:[all the splits as list], doc 2: [all the splits as a list]}

    print("Splits :\n",splits_dict["doc 0"][0],splits_dict["doc 1"][0],"......") #show split of a seleted doc

    for key, value in splits_dict.items():
        print(key,": NO of chunks created",len(value))
    print("---------Splits Created---------")
    # Embed
    vectorstores = {}
    for key, value in splits_dict.items():

        vectorstores[key] = Chroma.from_documents(documents=value,
                                            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    print("---------vectorstores Created---------")

    retrievers={}
    for key, value in vectorstores.items():
        retrievers[key] = value.as_retriever()



    print("---------retrievers Created---------")

    # # Prompt
    # prompt = hub.pull("rlm/rag-prompt")

    # # LLM
    # llm = ChatOpenAI(
    #     model="gpt-oss:120b",
    #     api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
    #     base_url="https://ollama.com/v1",
    #     # organization="...",
    #     # other params...
    # )

    # # Post-processing
    # def format_docs(all_docs):
    #     return "\n\n".join(doc.page_content for doc in all_docs)

    # # Chain
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # # Question
    # rag_chain.invoke("Chatgpt?")

    """---

    **CLUSTER**

    ---

    ☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁☁
    """

    print("---------Pages Extraction---------")
    pages={}
    for i, doc in enumerate(all_docs):
        pages[f"doc {i}"]=[]
        for j in doc:
            pages[f"doc {i}"].append(j.page_content)


    print("---------responses Extraction---------")
    responses={}
    for i in range(len(pages)):
        responses[f"doc {i}"]=[]
        for page in pages[f"doc {i}"][:10]:
            prompt = f"""
            You need to understand the content of this page: {page}

            Think about what aspects could differ if the same topic appeared in another similar document.
            Based only on this page and your careful understanding, identify the **headings** that describe possible differences to check.Make the heading more detailed

            Focus **only on technical or impactful aspects**.
            List **only** the headings, without any explanations or details.
            """


            response = llm.invoke(prompt)
            responses[f"doc {i}"].append(response.content)

    print("---------senteces Extraction---------")
    senteces = {}
    for key,value in responses.items():
        senteces[key] = ""
        for i in range(len(value)):
            senteces[key] += value[i]

    sentences_list={}
    for key,value in senteces.items():
        sentences_list[key] = [line for line in value.split("\n") if line.strip()]

    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    print("---------Embedding Creation---------")
    embeddings = {}
    for key,value in sentences_list.items():
        embeddings[key] = []
        for i in range (len(value)):
            embeddings[key].append(embedding.embed_query(value[i]))




    """method 1

    check similarity between each of the embeddings and then take the maximun chunk out

    method 2 will be through clustering
    """
    print("---------similarity Calculation ---------")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = model.similarity(embeddings["doc 0"], embeddings["doc 1"])

    # Get argmax indices per row
    indices = torch.argmax(similarities, dim=1)

    print(indices)   # tensor([1, 3, 1])

    values, indices = torch.max(similarities, dim=1)
    print(values)   # max values
    print(indices)  # indices of max values

    # Get top 3 per row
    comparing_headings=[]
    values, indices = torch.topk(similarities, k=3, dim=1)
    indices = indices.tolist()
    values = values.tolist()
    for i in range(len(indices)):   #len(indices)
        #print(f"i: ", indices[i], values[i])
        if values[i][0] < 0.5:
            continue
            print(sentences_list["doc 0"][i] , "NO MATCH DETECTED")
            continue
        print(sentences_list["doc 0"][i])
        print(sentences_list["doc 1"][indices[i][0]])
        comparing_headings.append([sentences_list["doc 0"][i],sentences_list["doc 1"][indices[i][0]]])
        print('-' * 50)


    print(f"---------comparing_headings Creation {len(comparing_headings)}---------")

    #### RETRIEVAL and GENERATION ####


    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are given two contexts from different sources.

    Context 1:
    {context1}

    Context 2:
    {context2}

    Reason:
    {reason}

    Question: Are there any contradictions between the two contexts?
    If yes, list them clearly. If no, say "No contradictions found" and Explain the reaons only if Reason is true
    """)

    # LLM
    llm = ChatOpenAI(
        model="gpt-oss:120b",
        api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
        base_url="https://ollama.com/v1",
        # organization="...",
        # other params...
    )

    # Post-processing
    def format_docs(all_docs):
        return "\n\n".join(doc.page_content for doc in all_docs)

    # Chain
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

    def get_topk_docs(pair, k=5):
        docs1 = retrievers["doc 0"].get_relevant_documents(pair[0], k=k)
        docs2 = retrievers["doc 1"].get_relevant_documents(pair[1], k=k)
        return docs1, docs2

    output=""
        # File to store results

    # File path inside the folder
    output_file = os.path.join(folder_path, "doc_pair_comparison_results.csv")

    # Write header
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Doc_Pair", "Docs1_Content", "Docs2_Content", "Comparison_Result"])

    for pair in comparing_headings[:10]: # To test we only compare first 10 
        print(pair)
        docs1 ,docs2 = get_topk_docs(pair)
        docs1_f = format_docs(docs1)
        docs2_f = format_docs(docs2)
        response = rag_chain.invoke({"context1": docs1_f, "context2": docs2_f,"reason": "NO"})
        print(response)
        if "no contradiction" not in response.lower():  
            output += response
          
        # Append row to CSV
        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([pair, docs1_f, docs2_f, response])
            print('- -' * 50)
    #dendogram-> conncet all the nodes via low distance are conneted first method to final single cluster(n+n,n+c,c+c)  Then cut the dendogram from the threshhold value and use thoscluserts
    return output


class State(TypedDict):
    graph_state: str
    current_node: str
    no_files_uploaded: int
    file_contents: list

################################################# PIPE CLASS #################################################
class Pipeline:
    """
    Pipeline implementation using Open Web UI for Document comparison via Large Language Models (LLMs).

    Workflow:
    Initialization:
    Features:
    Version:
        - 0.1v
    """
    
    class Valves(BaseModel):
        """
        Configuration container class for storing constants related to
        API endpoints, model settings, debugging flags, and file paths.

        These values are used globally throughout the agent pipeline.
        """
        ################################################# Configuration #################################################
        # Base URLs
        TOKEN: str 
        OPEN_WEB_UI_BASE_URL: str 
        OLLAMA_BASE_URL: str 
        OPENAI_BASE_URL: str  #llama.cpp model access
        MODEL: str 
        # Debugging and processing settings
        DEBUG: bool = False
        NUM_RE_PROCESS: int 

        # SQLite database path for storing chat metadata
        DB_PATH: ClassVar[str] = r"K_HR.db"
        """Relative path to the SQLite database for storing chat states and metadata."""

    def __init__(self):
        """
        Initializes the pipeline class.

        This method runs only once when the pipeline is activated 
        (e.g., when the backend server restarts or reinitializes the agent).

        It sets up default values for state management, token authentication,
        memory structures, and existing file tracking.
        """
        print('*' * 10 + "CLASS PIPE CALLED" + '*' * 10)

        # Load configuration parameters
        self.valves = self.Valves(
            **{
                "TOKEN": os.getenv("TOKEN", ""),
                "OPEN_WEB_UI_BASE_URL": os.getenv("OPEN_WEB_UI_BASE_URL", "http://localhost:8080/api/v1"),
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:8080/ollama"),
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "http://localhost:8080/openai"),
                "MODEL": os.getenv("MODEL", "DeepSeek-R1-0528-Qwen3-8B-Q2_K_L.gguf"),
                "DEBUG": os.getenv("DEBUG_MODE", "false").lower() == "true",
                "NUM_RE_PROCESS": os.getenv("NUM_RE_PROCESS", 1),
            }
        )

        # Agent name identifier
        self.name = "Doc_Agent"
        self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjUzYWU1OGQ1LTI0YzQtNGE3NC1iOTI5LWNiMjNlOWE3NDU2ZCJ9.5xXDsBfke4PhYq1ZeAKQliZkRm2lKUP7Ty1HBqu-l7A" #self.valves.TOKEN
        self.debug(f"Token __INIT__ :{self.token}")
        self.debug('-' * 50)
        self.NO_of_all_files_uploaded = len(self.get_file_ids_names_for_selected_user(None,self.token)) #logic crashes when tehre are parallel users
        self.debug(f"NO of all files uploaded before the pipeline start :{self.NO_of_all_files_uploaded}")
        self.debug('-' * 50)
        # Runtime memory to temporarily store parsed requirements by chat -> no need after DB integration
        self.memory_requirments_dict = {}

        # Count files already available in knowledge base
        #self.file_count = len(self.get_file_names_ids(self.token))

        # Mapping from chat IDs to knowledge base IDs
        self.chatids_to_knowledgeids = {}

        # Store evaluation results of candidates
        self.evaluation_results = {}

        # Set initial internal state of the agent
        self.state = "REQUIRMENTS_UPDATE" #within the pipe function it checks and update
        
        self.threads = {} # contain chat_id : thread id
        self.graphs = {} # contain chat_id : graph_obj
    async def on_startup(self):
        # This function is called when the server is started.
        print(f"AGENT_ON:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is shutdown.
        print(f"AGENT_ON:{__name__}")
        pass
        
        
    def debug(self, *args, **kwargs) -> None:
        """
        Prints debug messages if DEBUG mode is enabled in the valves configuration.
        Parameters:
            - *args, **kwargs: Variable argument messages to be printed.
        """
        if self.valves.DEBUG:
            print(*args, **kwargs)
            
    def get_file_ids_names_for_selected_user(self,user_id: str, token: str) -> list:
        """
        Get all uploaded files with their IDs and names.
        Returns:
            List of dictionaries for a given user: [{id1: name1}, {id2: name2}, ...]
            
        If no user is being present it return all the files
        """

        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/files/'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            files = response.json()
            if user_id:
                file_names = [{file['id']:file['filename']} for file in files[:] if file['user_id']==user_id] 
            else:
                file_names =  [{f['id']: f['filename']} for f in files]
            return file_names
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []   

    
################################################ 2 Functions in knowledge base ######################################################
        
# ----------------------------------------------------------------------------
# 2.1 Retrieve the latest file IDs (up to a specified count)
# ----------------------------------------------------------------------------
    def get_file_ids(self, token, count) -> list:
        """
        Get the IDs of the most recently uploaded files.
        Returns:
            List of file IDs: [id1, id2, ..., idN]
        Note: Make sure that the list is sorted correctly by upload time in the API.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/files/'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    #Issue: refer to the api instead of local /.data dir where possibility of not matching in some senarios
        response = requests.get(url, headers=headers)
        if count == 0:
            return []
        elif response.status_code == 200:
            files = response.json()
            file_ids = [file['id'] for file in files[-count:]]  # need to confirm whether the top file is the updated ones always*******
            return file_ids
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []

        # ----------------------------------------------------------------------------
# 2.9 Get detailed content of specific files (by file IDs) --> Used when file_ids are known
# ----------------------------------------------------------------------------
    def get_file_details(self, token, file_ids) -> list:
        """
        Retrieve the ID, filename, and content of each file in the knowledge base by file IDs.

        Args:
            token (str): JWT authorization token.
            file_ids (list): List of file IDs to fetch.

        Returns:
            list: A list of [file_id, filename, content] for each file.
        """
        file_details = []
        for id in file_ids:
            url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/files/{id}'
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }    
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                file_details.append([data["id"], data["filename"], data["data"]["content"]])
            else:
                return {
                    "error": f"Failed to fetch file {id}",
                    "status_code": response.status_code,
                    "details": response.text
                }
        return file_details

    
    def is_general(self, state: State) -> Literal["General_chat_test_node","Begin_node"]:
        """
        check whether the user need to move with the general stage
        """
        user_message = state["graph_state"]
        return "General_chat_test_node" if (user_message.lower()=="yes") else "Begin_node"

    
    def Begin_node(self, state: State):
        print("-- Graph begin --")
        state["current_node"] = "Begin_node"
        return state

    def General_node(self, state: State):
        print("G e n e r a l")
        state["current_node"] = "General_node"
        return state                      #{"graph_state": state['graph_state']}
        ...
        
    def General_chat_test_node(self, state: State):
        print("General_chat_test_node")
        state["current_node"] = "General_chat_test_node"
        user_message = state["graph_state"]
        llm = ChatOpenAI(

            #To use ollama cloud models
            model="gpt-oss:120b",
            api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
            base_url="https://ollama.com/v1",
            
            # #To use llama.cpp models
            # model = self.valves.MODEL,
            # base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            # api_key = self.token #accesing owui and it handles the rest
        )

        human_message = user_message
        system_message = "You are an Human Resource agent, your task is to reply user with warm greeting which is less than 50 words"
        response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])
        state["graph_state"] = response.content
        return state
    
    def requirements_updated(self, state: State):
        print("-- requirements_updated --")
        state["current_node"] = "requirements_updated"
        return state
    
    def DOC_processor(self, state: State):
        print("-- DOC_processor -- ")
        state["current_node"] = "DOC_processor"
        extracted_file_contents = state["file_contents"]
        user_message = state["graph_state"] # need to change this to directly use the text extracted from the be
        llm = ChatOpenAI(

            #To use ollama cloud models
            model="gpt-oss:120b",
            api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
            base_url="https://ollama.com/v1",
            
            # #To use llama.cpp models
            # model = self.valves.MODEL,
            # base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            # api_key = self.token #accesing owui and it handles the rest
        )
        
        return state

    
    def General_chat_node(self, state: State): #use global token - pipeline class
        print("General_chat_node")
        state["current_node"] = "General_chat_node" # as this is a interrupt node this function won't call -> fix it now it calls by maming as node
        user_message = state["graph_state"]

        llm = ChatOpenAI(

            #To use ollama cloud models
            model="gpt-oss:120b",
            api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
            base_url="https://ollama.com/v1",
            
            # #To use llama.cpp models
            # model = self.valves.MODEL,
            # base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            # api_key = self.token #accesing owui and it handles the rest
        )

        human_message = user_message
        system_message = "You are an Human Resource agent, your task is to reply user message with words less than 100"
        response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])
        state["graph_state"] = response.content
        return state
    
    def is_file_upload_node(self, state: State):
        print("is_file_upload_node")
        state["current_node"] = "is_file_upload_node" #
        return state
    
    def is_file_upload(self, state: State)-> Literal["General_chat_test_node","DOC_processor","Begin_node"]:
        if state["no_files_uploaded"] == 0 :
            user_message = state["graph_state"]
            if user_message.lower() == "yes":
                return "General_chat_test_node"
            else:  
                return "Begin_node"
        else: 
            return "DOC_processor"      
    
    def start_node(self, state: State)-> Literal["Begin_node","DOC_processor"]:
        if state["no_files_uploaded"] == 0 :
            return "Begin_node"
        else: 
            return "DOC_processor"  



    def graph_builder(self) -> dict:
        
        builder = StateGraph(State)
        # Define nodes: these do the work
        builder.add_node("Begin_node", self.Begin_node)
        builder.add_node("DOC_processor",self.DOC_processor)
        builder.add_node("General_chat_test_node",self.General_chat_test_node)
        builder.add_node("General_chat_node",self.General_chat_node)
        builder.add_node("is_file_upload_node",self.is_file_upload_node) 
        # Logic
        builder.add_conditional_edges(START, self.start_node)
        builder.add_edge("Begin_node","is_file_upload_node")
        builder.add_conditional_edges("is_file_upload_node", self.is_file_upload)    
        builder.add_edge("DOC_processor", END)
        builder.add_edge("General_chat_test_node", "General_chat_node")
        builder.add_edge("General_chat_node", END)
        
        # Persistent checkpointer: saved in a local SQLite file
        checkpointer = SqliteSaver.from_conn_string("sqlite:///chat_memory.db")
        memory = MemorySaver()
        graph = builder.compile(interrupt_before=["General_chat_node","is_file_upload_node"],checkpointer=memory)
        return graph
        ...      
            
    def get_current_chat_id(self, token) -> str:
        """
        Fetches the latest (most recent) chat ID from the conversation list
        using the Open WebUI API endpoint. This represents the ongoing chat.

        Args:
            token (str): Bearer token for authentication.

        Returns:
            str: Chat ID of the latest conversation, or [] on error.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/chats/list?page=1'  # Retrieves most recent chats (first page)
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            chat = response.json()
            chat_id = chat[0]['id']
            return chat_id
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []        
            
            
            
            
    ################################################ pipe ######################################################
# ----------------------------------------------------------------------------
# pipe function is going to run always when a new message is being prompted
# ----------------------------------------------------------------------------

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes user messages and routes them through
        different states such as requirements input, CV processing, or general chat.

        Parameters:
        - user_message (str): User input typed in the UI.
        - model_id (str): Model ID selected for inference (e.g., openai/gpt-4, llama3).
        - messages (List[dict]): Contextual chat history including system, user, and assistant messages.
        - body (dict): Additional Open WebUI settings (e.g., streaming config, max tokens, temperature).

        Returns:
        - Union[str, Generator, Iterator]: Streamed or final model response.
        """
        # ----------------------------------------------------------------------------
        # DEBUG AND PIPE ACTIVATION LOG
        # ----------------------------------------------------------------------------
        user_output = "This is a test message" # o/p message send into the ui
        print(body)
        
        if self.valves.DEBUG:
            print("------------------DEBUG MODE : ON--------------------", '\n')
        else:
            print("------------------DEBUG MODE : OFF-------------------", '\n')

        self.debug("PIPE FUNCTION ACTIVATED")
        self.token =  self.valves.TOKEN
        self.debug('-' * 50)
        self.debug(f"api_key :{self.token}")
        self.debug('-' * 50)
        print(f"pipe:{__name__}")
        self.debug('-' * 50)
        user_id = (body["user"]["id"])
        self.debug(f"user_id :{user_id}")
        self.debug('-' * 50)
        
        # Get the uploaded file ids -> issue: not specific with user parallel file upload can be a issue ------------------------------------------------
        all_files_uploaded = self.get_file_ids_names_for_selected_user(None,self.token)
        self.debug(f"all_files_uploaded :{all_files_uploaded}")
        self.debug('-' * 50)
        NO_of_all_files_uploaded = len(self.get_file_ids_names_for_selected_user(user_id,self.token))
        self.debug(f"NO of all files uploaded by this user :{NO_of_all_files_uploaded}")
        self.debug('-' * 50)
        self.debug(f"self.NO_of_all_files_uploaded earlier: {self.NO_of_all_files_uploaded}")
        self.debug('-' * 50)
        No_of_files_uploaded_for_processing = NO_of_all_files_uploaded - self.NO_of_all_files_uploaded
        self.debug(f"NO of all files uploaded for processing: {No_of_files_uploaded_for_processing}")
        self.debug('-' * 50)
        self.NO_of_all_files_uploaded = NO_of_all_files_uploaded
        
        file_ids = self.get_file_ids(self.token,No_of_files_uploaded_for_processing)
        print(file_ids)
        file_details = self.get_file_details(self.token,file_ids)
        #print(file_details)

        chat_id = self.get_current_chat_id(token=self.token)
        self.debug(f"get_current_chat_id: {chat_id}")
        self.debug('-' * 50)
        #Initialize the llm based on langgraph ------------------------------------------------
        llm = ChatOpenAI(

            #To use ollama cloud models
            model="gpt-oss:120b",
            api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # if you prefer to pass api key in directly instaed of using env vars
            base_url="https://ollama.com/v1",
            
            # #To use llama.cpp models
            # model = self.valves.MODEL,
            # base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            # api_key = self.token #accesing owui and it handles the rest
        )
        
        #Test model invocation ------------------------------------------------
        if False: #if self.valves.DEBUG
            try:
                sys_msg = SystemMessage(content="You Task is to verify that you get the user message to do that response wit the same user message")
                response = llm.invoke([sys_msg] + [HumanMessage(content = user_message)])
                self.debug(f"LLM Works fine: {response.content}")
                self.debug('-' * 50)
            except Exception as e:
                self.debug(f"LLM invocation failed: {e}")
        
        #input
        message = {"graph_state": user_message, "current_node": "START","no_files_uploaded": No_of_files_uploaded_for_processing,"file_contents":file_details}
        if chat_id in self.graphs:
            graph = self.graphs[chat_id]
            self.debug(f"Graph is available: {self.graphs[chat_id]}")
            self.debug('-' * 50)
            if chat_id in self.threads:
                thread = self.threads[chat_id]
                self.debug(f"chat_id is in the thread: {thread}")
                self.debug('-' * 50)
            else:
                thread = {"configurable": {"thread_id": f"{chat_id}"}}
                self.threads[chat_id] = thread
                self.debug(f"chat_id is NOT in the thread: {thread}")
                self.debug('-' * 50)
        else:
            
            graph = self.graph_builder()
            self.graphs[chat_id] = graph
            self.debug(f"Graph Creation: {graph}")
            self.debug('-' * 50)
            thread = {"configurable": {"thread_id": f"{chat_id}","callbacks": [langfuse_handler]}}
            self.threads[chat_id] = thread
            self.debug(f"chat_id is NOT in the thread: {thread}")
            self.debug('-' * 50)
            
        state = graph.get_state(thread)
        self.debug(f"state of the graph : {state}")
        self.debug('-' * 50)
        if "General_chat_node" in state.next:
            self.debug(f"General_chat_node - INTERRUPT ")
            self.debug('-' * 50)
            graph.update_state(thread, {"graph_state": 
                                user_message},as_node=state.values["current_node"] )
            message = None
        elif "is_file_upload_node" in state.next:
            self.debug(f"is_file_upload_node - INTERRUPT ")
            state.values["no_files_uploaded"] = No_of_files_uploaded_for_processing
            state.values["file_contents"] = file_details
            self.debug('-' * 50)
            graph.update_state(thread, {"graph_state": 
                                user_message,"file_contents":file_details,"no_files_uploaded":No_of_files_uploaded_for_processing},as_node=state.values["current_node"] )
            message = None
        for event in graph.stream(message, thread, stream_mode="values"):
            # Review-> events are the messages passing through each edges
            print(f"event:{event}")
            # Get state and look at next node
            state = graph.get_state(thread)
            #self.debug(f"state of the graph : {state}")
            self.debug('-' * 50)
            if "General_chat_test_node" == event["current_node"]:
                user_output = event["graph_state"]    
            elif "General_chat_node" == event["current_node"]:
                user_output = event["graph_state"] 
            elif "is_file_upload_node" in state.next:
                user_output = "Please upload DOCUMENTS\n Wanna have a chat?(yes/no)"
            elif "DOC_processor" == event["current_node"]:
                user_output = str(compare_documents(file_details))
                

            self.debug(f"NEXT state of the graph : {state.next}")
            
            self.debug('-' * 50)
            self.debug(f"user_output: {user_output} ")
            self.debug('-' * 50)
            
        return user_output