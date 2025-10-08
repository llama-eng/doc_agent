"""
Pipelined - Implemented on open web ui for candidate CV evaluation through LLM

1. Add requirements into the chat --> until it gets verified
2. Add CV's --> create a knowledge base from the CV's and return the explicit llm call given as default
3. Use the chat to ask necessary informations based on the knowledge base

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
 
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()
class State(TypedDict):
    graph_state: str
    current_node: str
    no_files_uploaded: int
################################################# PIPE CLASS #################################################
class Pipeline:
    """
    Pipeline implementation using Open Web UI for candidate CV evaluation via Large Language Models (LLMs).

    Workflow:
        

    Initialization:
    
    Features:
        

    Version:
        - 2.0.0
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
        self.name = "Agentic_HR"
        self.token = self.valves.TOKEN
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
    
    def requirements_test(self, state: State) -> Literal["requirements_updated","interrupt_node"]:
        """
        Validates the input requirements string.

        Each line in the string must be in the format: <requirement>,<rating>
        where rating is an integer from 1 to 5.

        Parameters:
        - requirements (str): Multiline string with requirements.
        - chat_id: Chat identifier (not used in this version but may store data later).

        Returns:
        - bool: True if requirements are valid (error rate ≤ 10%), else False.
        """
        requirements = state["graph_state"]
        
        error_line_count = 0
        error_percentage = 0.10  # Maximum allowed error: 10%
        error_limit = int(len(requirements.splitlines()) * error_percentage)

        if len(requirements) == 0:
            return False

        for line in requirements.splitlines():
            pattern = r'^(.+?),\s*([1-5])$'
            match = re.match(pattern, line.strip())
            if match:
                text = match.group(1).strip()
                rating = int(match.group(2))
                # You may update memory here using chat_id if needed
            else:
                print(f"Invalid format in line: {line}")
                error_line_count += 1

        return "requirements_updated" if error_line_count <= error_limit else "interrupt_node"
    
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
            model = self.valves.MODEL,
            base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            api_key = self.token #accesing owui and it handles the rest
        )
        human_message = user_message
        system_message = "You are an Human Resource agent, your task is to reply user with wam greeting which is less than 50 words"
        response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])
        state["graph_state"] = response.content
        return state
    
    def requirements_updated(self, state: State):
        print("-- requirements_updated --")
        state["current_node"] = "requirements_updated"
        return state
    
    def CV_processor(self, state: State):
        print("-- CV_processor -- ")
        state["current_node"] = "CV_processor"
        user_message = state["graph_state"] # need to change this to directly use the text extracted from the be
        llm = ChatOpenAI(
            model = self.valves.MODEL,
            base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            api_key = self.token #accesing owui and it handles the rest
        )
        human_message = user_message
        system_message = (
    "You are a Human Resources agent. Your task is to evaluate the given CV. "
    "Always provide your evaluation in the following format:\n"
    "<overall Score range from 0-100>: float or int\n"
    "<Reason>: string\n"
    "Ensure the score reflects the candidate's suitability and the reason clearly explains your assessment."
)
        response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])
        state["graph_state"] = response.content
        return state
    
    def interrupt_node(self, state: State):
        print("-- interrupt_node --")
        state["current_node"] = "interrupt_node"
        return state
    
    def General_chat_node(self, state: State): #use global token - pipeline class
        print("General_chat_node")
        state["current_node"] = "General_chat_node" # as this is a interrupt node this function won't call -> fix it now it calls by maming as node
        user_message = state["graph_state"]
        llm = ChatOpenAI(
            model = self.valves.MODEL,
            base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            api_key = self.token #accesing owui and it handles the rest
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
    
    def is_file_upload(self, state: State)-> Literal["requirements_updated","CV_processor"]:
        return "requirements_updated" if state["no_files_uploaded"] == 0 else "CV_processor"
        
    
    def graph_builder(self) -> dict:
        
        builder = StateGraph(State)
        # Define nodes: these do the work
        builder.add_node("Begin_node", self.Begin_node)
        builder.add_node("interrupt_node", self.interrupt_node)
        builder.add_node("requirements_updated", self.requirements_updated)
        builder.add_node("CV_processor",self.CV_processor)
        builder.add_node("General_chat_test_node",self.General_chat_test_node)
        builder.add_node("General_chat_node",self.General_chat_node)
        builder.add_node("is_file_upload_node",self.is_file_upload_node) 
        # Logic
        builder.add_edge(START, "Begin_node")
        builder.add_conditional_edges("Begin_node", self.requirements_test)
        builder.add_conditional_edges("interrupt_node", self.is_general)
        #builder.add_edge("requirements_updated", "CV_processor")
        builder.add_edge("requirements_updated", "is_file_upload_node")
        builder.add_conditional_edges("is_file_upload_node", self.is_file_upload)
        
        builder.add_edge("CV_processor", END)
        builder.add_edge("General_chat_test_node", "General_chat_node")
        builder.add_edge("General_chat_node", END)
        
        # Persistent checkpointer: saved in a local SQLite file
        checkpointer = SqliteSaver.from_conn_string("sqlite:///chat_memory.db")
        memory = MemorySaver()
        graph = builder.compile(interrupt_before=['interrupt_node',"General_chat_node","is_file_upload_node"],checkpointer=memory)
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
        self.debug(f"self.NO_of_all_files_uploaded: {self.NO_of_all_files_uploaded}")
        self.debug('-' * 50)
        No_of_files_uploaded_for_processing = NO_of_all_files_uploaded - self.NO_of_all_files_uploaded
        self.debug(f"NO of all files uploaded for processing: {No_of_files_uploaded_for_processing}")
        self.debug('-' * 50)
        self.NO_of_all_files_uploaded = NO_of_all_files_uploaded
        
    
        chat_id = self.get_current_chat_id(token=self.token)
        self.debug(f"get_current_chat_id: {chat_id}")
        self.debug('-' * 50)
        #Initialize the llm based on langgraph ------------------------------------------------
        llm = ChatOpenAI(
            model = self.valves.MODEL,
            base_url = f"{self.valves.OPENAI_BASE_URL}/v1",
            api_key = self.token #accesing owui and it handles the rest
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
        message = {"graph_state": user_message, "current_node": "START","no_files_uploaded": No_of_files_uploaded_for_processing}
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
        if "interrupt_node" in state.next:
            self.debug(f"interrupt_node- INTERRUPT")
            self.debug('-' * 50)
            graph.update_state(thread, {"graph_state": 
                                user_message},as_node=state.values["current_node"]) #use as node to mention start node
            message = None
        elif "General_chat_node" in state.next:
            self.debug(f"General_chat_node - INTERRUPT ")
            self.debug('-' * 50)
            graph.update_state(thread, {"graph_state": 
                                user_message},as_node=state.values["current_node"] )
            message = None
        elif "is_file_upload_node" in state.next:
            self.debug(f"is_file_upload_node - INTERRUPT ")
            state.values["no_files_uploaded"] = No_of_files_uploaded_for_processing
            self.debug('-' * 50)
            graph.update_state(thread, {"graph_state": 
                                user_message,"no_files_uploaded":No_of_files_uploaded_for_processing},as_node=state.values["current_node"] )
            message = None
        for event in graph.stream(message, thread, stream_mode="values"):
            # Review-> events are the messages passing through each edges
            print(f"event:{event}")
            # Get state and look at next node
            state = graph.get_state(thread)
            self.debug(f"state of the graph : {state}")
            self.debug('-' * 50)
            if "interrupt_node" in state.next:
                user_output = "Please re-submit the requirements \n Else do you want to have a genral chat ? (yes/no)"
            elif "General_chat_test_node" == event["current_node"]:
                user_output = event["graph_state"]    
            elif "General_chat_node" == event["current_node"]:
                user_output = event["graph_state"] 
            elif "is_file_upload_node" in state.next:
                user_output = "Please upload cv's"
            elif "CV_processor" == event["current_node"]:
                user_output = event["graph_state"]
            self.debug(f"NEXT state of the graph : {state.next}")
            self.debug('-' * 50)
            self.debug(f"user_output: {user_output} , ")
            self.debug('-' * 50)
            
        return user_output