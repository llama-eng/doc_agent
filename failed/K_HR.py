"""
Pipelined - Implemented on open web ui for candidate CV evaluation through LLM

1. Add requirements into the chat --> until it gets verified
2. Add CV's --> create a knowledge base from the CV's and return the explicit llm call given as default
3. Use the chat to ask necessary informations based on the knowledge base

"""

#V3.1 
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import requests
from typing import List
import json
import re
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import time
import sqlite3
from typing import ClassVar
import pandas as pd
from dotenv import load_dotenv
import os
from langfuse.openai import openai

################################################# PIPE CLASS #################################################
class Pipeline:
    """
    Pipeline implementation using Open Web UI for candidate CV evaluation via Large Language Models (LLMs).

    Workflow:
        1. Add job requirements into the chat interface until they are verified.
        2. Upload candidate CVs, which are then used to create a knowledge base.
        3. Use the chat interface to query information and obtain candidate evaluations based on the knowledge base.

    Initialization:
        - Sets the agent name to 'K_HR_Agent'.
        - Uses a static JWT token for authentication with the Open Web UI APIs.
        - Initializes internal structures for tracking requirements, processed files, evaluations, and pipeline states.
        - The pipeline starts in the "REQUIRMENTS_UPDATE" state awaiting job requirement input.

    Features:
        - Manages chat states and interactions with the Open Web UI backend.
        - Handles knowledge base creation and association of candidate CV files.
        - Integrates with OpenAI or Ollama LLMs for CV evaluation and scoring.
        - Supports multi-stage processing: requirements verification, CV ingestion, and general chat handling.

    Version:
        - 2.2.0
    """
    
    class Valves(BaseModel):
        """
        Configuration container class for storing constants related to
        API endpoints, model settings, debugging flags, and file paths.

        These values are used globally throughout the agent pipeline.
        """

        ################################################# Configuration #################################################

        # # Base URLs for API services used by the OpenWebUI pipeline
        # OPEN_WEB_UI_BASE_URL: ClassVar[str] = "http://localhost:8080/api/v1"
        # """Base URL for accessing Open WebUI API endpoints."""

        # OLLAMA_BASE_URL: ClassVar[str] = "http://localhost:8080/ollama" #Not config yet need to go with openai
        # """Base URL for interacting with Ollama models."""

        # OPENAI_BASE_URL: ClassVar[str] = "http://localhost:8080/openai" #http://localhost:8080/openai/v1/chat/completions
        # """Proxy endpoint for OpenAI-like APIs via Ollama (e.g., port 11500 redirection)."""

        # # Default model to use (loaded by Ollama)
        # MODEL: ClassVar[str] = "deepseek-r1:7b"
        # """Default LLM model identifier used for CV evaluation and prompt generation."""

        # # Debugging and reprocessing flags
        # DEBUG: ClassVar[str] = "TRUE"
        # """Enable/disable debug mode globally in the pipeline."""

        # NUM_RE_PROCESS: ClassVar[int] = 1
        # """Number of times a CV will be re-evaluated for consistency checks."""
        # Base URLs
        TOKEN: str 
        OPEN_WEB_UI_BASE_URL: str 
        OLLAMA_BASE_URL: str 
        OPENAI_BASE_URL: str 
        # Default model
        MODEL: str 

        # Debugging and processing settings
        DEBUG: str 
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
                "DEBUG": os.getenv("OLLAMA_BASE_URL", "TRUE"),
                "NUM_RE_PROCESS": os.getenv("NUM_RE_PROCESS", 1),
            }
        )

        # Agent name identifier
        self.name = "K_HR_Agent"

        # Static bearer token for accessing Open WebUI APIs
        #load_dotenv()
        #self.token = os.getenv("PIPELINE_ACCESS_TOKEN")
        #self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijc4MDhmZDcyLWRlNWMtNDlkOS1iYTQ0LWFkNTE2MDg1N2MxYSJ9.lDbbiePl59jidejfTQSUSDXSVZn3ONJbCul0WpKnKvs"
        self.token = self.valves.TOKEN

        #self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImM3ODNhZmQwLTIzZTUtNGYwZS05OWQzLTM4MmQzZDRjODA5NCJ9.vYpeTJuAS6pEcZCrrsZ4lZj4nowhd14Yth57B8aL3Wg"
        # Runtime memory to temporarily store parsed requirements by chat -> no need after DB integration
        self.memory_requirments_dict = {}

        # Count files already available in knowledge base
        self.file_count = len(self.get_file_names_ids(self.token))

        # Mapping from chat IDs to knowledge base IDs
        self.chatids_to_knowledgeids = {}

        # Track document IDs that have already been processed
        # self.processed_ids = [] #-> no need after DB integration

        # Store evaluation results of candidates
        self.evaluation_results = {}

        # Set initial internal state of the agent
        self.state = "REQUIRMENTS_UPDATE" #within the pipe function it checks and update
        

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"AGENT_ON:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is shutdown.
        print(f"AGENT_ON:{__name__}")
        pass
    
#     async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
#         #print(f"pipe:{__name__}")
#         #print(f"body['messages']: {body['messages'][0]['content']}")
# #         body['messages'][0]['content'] = """######PIPELINE INJECTED#########
        
# #         Architect and develop high-performance networking stacks for embedded platforms, 3
# # Implement and optimize networking protocols (TCP/IP, UDP, OSPF, BGP, VLANs, PPPoE, L2/L3 forwarding, multicast, QoS, etc.)., 3
# # Develop real-time packet processing, traffic shaping, and high-throughput data paths, 3
# # Optimize latency, throughput, and power efficiency in embedded networking applications., 3
# # Design and implement multi-threaded networking applications for embedded Linux and RTOS, 4
# # Engage in FPGA design and development, including coding, simulation, and testing (optional), 2
# # Develop low-level firmware, device drivers, and BSPs for networking SoCs, 4
# #         """
#         #print(f"body['messages']: {body}")
#         return body

################################################ 1 VERIFICATION-STAGES ######################################################

    def set_state(self, new_state):
        """Set the current internal state of the pipeline."""
        self.state = new_state

    def is_state(self, check_state):
        """Check if the current state matches the given state."""
        return self.state == check_state

# ----------------------------------------------------------------------------
# 1.1 Validate format of requirements
# ----------------------------------------------------------------------------
    def is_valid_requirements(self, requirements: str, chat_id) -> bool:
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

        return error_line_count <= error_limit

# ----------------------------------------------------------------------------
# 1.2 Fetch knowledge base ID by name
# ----------------------------------------------------------------------------
    def get_knowledge_id_by_name(self, token, knowledge_name="company data"):
        """
        Retrieves the ID of a knowledge base given its name.

        Parameters:
        - token (str): Authorization bearer token.
        - knowledge_name (str): Name of the knowledge base (case-insensitive).

        Returns:
        - str or None: ID of the matching knowledge base, or None if not found.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/list'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return {
                "error": "Failed to fetch knowledge list",
                "status_code": response.status_code,
                "details": response.text
            }

        knowledge_list = response.json()
        for knowledge in knowledge_list:
            if knowledge.get("name", "").lower() == knowledge_name.lower():
                return knowledge.get("id")

        return None


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

        if response.status_code == 200:
            files = response.json()
            file_ids = [file['id'] for file in files[-count:]]  # need to confirm whether the top file is the updated ones always*******
            return file_ids
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
        
# ----------------------------------------------------------------------------
# 2.2 Retrieve all file IDs and their names
# ----------------------------------------------------------------------------
    def get_file_names_ids(self, token) -> list:
        """
        Get all uploaded files with their IDs and names.
        Returns:
            List of dictionaries: [{id1: name1}, {id2: name2}, ...]
        """

        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/files/'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            files = response.json()
            file_names = [{file['id']:file['filename']} for file in files[:]] 
            #file_ids = [file['id'] for file in files[:]]
            return file_names
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []

# ----------------------------------------------------------------------------
# 2.3 Check if a knowledge base with a given name exists
# ----------------------------------------------------------------------------
    def check_knowledge_exsistence(self, token, name) -> str:
        """
        Check if a knowledge base with the specified name already exists.
        Returns:
            Knowledge ID if exists, otherwise False.
        """
        
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/'  # The endpoint that returns all knowledge entries only works in earlier versions
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)  # Use GET to fetch list
        response.raise_for_status()  # Always good to check for HTTP errors
        
        for k in response.json():
            if k["name"] == name:
                return k["id"]
        return False

            
# ----------------------------------------------------------------------------
# 2.4 Create a new knowledge base
# ----------------------------------------------------------------------------
    def create_knowledge(self, token, name="Example Knowledge", description="Example description") -> str:
        """
        Create a new knowledge base with the given name and description.
        Returns:
            ID of the newly created knowledge base.
        """  
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/create'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "name": name,
            "description": description,
            "data": {},
            "access_control": {}
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()['id']  

# ----------------------------------------------------------------------------
# 2.5 Add a file to a specified knowledge base
# ----------------------------------------------------------------------------
    def add_file_to_knowledge(self, token, knowledge_id, file_id) -> dict:
        """
        Add a file to an existing knowledge base.
        Returns:
            API response containing the status of the operation.
        """

        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/{knowledge_id}/file/add'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        data = {'file_id': file_id}
        response = requests.post(url, headers=headers, json=data)
        return response.json()

# ----------------------------------------------------------------------------
# 2.6 Get all file IDs in a knowledge base
# ----------------------------------------------------------------------------
    def get_knowledge_file_id(self, token, knowledge_id) -> list:
        """
        Retrieve all file IDs associated with a given knowledge base.
        Returns:
            List of file IDs if found, or error dict if failed.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/api/v1/knowledge/list'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {
                "error": f"Failed to fetch knowledge list",
                "status_code": response.status_code,
                "details": response.text
            }

        knowledge_list = response.json()
        for knowledge in knowledge_list:
            if knowledge.get("id") == knowledge_id:
                return knowledge.get("data", {}).get("file_ids", [])

        return {
            "error": f"Knowledge ID {knowledge_id} not found"
        }

    # ----------------------------------------------------------------------------
    # 2.7 Get the full textual content of a knowledge base (as a single string)
    # ----------------------------------------------------------------------------
    def get_knowledge(self, token, knowledge_id) -> str:
        """
        Retrieve the combined text content of all files in the knowledge base.
        Returns:
            A single string containing all the file contents.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/{knowledge_id}'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        knowledge_details = ""
        if response.status_code == 200:
            for content in response.json()["files"]:
                knowledge_details +=  content["data"]["content"]
            return knowledge_details
        else:
            return {
                "error": f"Failed to fetch knowledge {knowledge_id}",
                "status_code": response.status_code,
                "details": response.text
            }
    
    # ----------------------------------------------------------------------------
    # 2.8 Get detailed content of a knowledge base (as a list per file) ---> check for new updates
    # ----------------------------------------------------------------------------
    def get_knowledge_details(self, token, knowledge_id) -> list:
        """
        Retrieve the ID, filename, and content of each file in the knowledge base.
        Returns:
            A list of [file_id, filename, content] for each file.
        """
        url = f'{self.valves.OPEN_WEB_UI_BASE_URL}/knowledge/{knowledge_id}'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        knowledge_details = []
        if response.status_code == 200:
            for content in response.json()["files"]:
                knowledge_details.append([content["id"],content["filename"],content["data"]["content"]])  
            return knowledge_details
        else:
            return {
                "error": f"Failed to fetch knowledge {knowledge_id}",
                "status_code": response.status_code,
                "details": response.text
            }
            
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



    ################################################ Chat Utilities ################################################
    # ----------------------------------------------------------------------------
    # Function: get_current_chat_id
    # Description: 
    #   Fetches the most recent chat ID from the chat list.
    #   This is typically the latest/ongoing conversation started via a message.
    # Parameters:
    #   token (str) - Authorization token for API access.
    # Returns:
    #   str - The latest chat ID (if available), else an empty list.
    # ----------------------------------------------------------------------------
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

    ################################################ Database Utilities ############################################
    def analyze_DB(self,db_path):
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Step 1: List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(" Tables in the database:")
        for table in tables:
            print(f" - {table[0]}")

        # Step 2: Preview the content of each table
        for table_name in tables:
            print(f"\nPreview of table: {table_name[0]}")
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", conn)
                print(df.columns.tolist())
                print(df)
            except Exception as e:
                print(f"Could not read table {table_name[0]}: {e}")

        # Optional: Explore specific table in full
        # df_full = pd.read_sql_query("SELECT * FROM <your_table_name>", conn)

        # Cleanup
        conn.close()
    
    # ----------------------------------------------------------------------------
    # Function: database_update
    # Description:
    #   Creates or updates the local SQLite database storing chat metadata.
    #   Ensures necessary schema exists, inserts new rows if required, 
    #   and updates knowledge_id and requirements fields.
    # ----------------------------------------------------------------------------
    def database_update(self, db_path, updates):
        """
        Updates the local SQLite database storing chat-related metadata.

        Steps:
        1. Create 'chat' table if not present.
        2. Insert missing chat IDs from the `updates` dictionary.
        3. Ensure required columns exist (`knowledge_id`, `requirements`).
        4. Update chat records with given `knowledge_id` and `requirements`.

        Args:
            db_path (str): Path to the SQLite database file.
            updates (dict): Dictionary with chat_id as key and (knowledge_id, requirements) tuple as value.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Step 1: Create chat table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                knowledge_id TEXT,
                requirements TEXT
            );
        """)
        conn.commit()

        # Step 2: Insert any new chat IDs
        existing_ids = set(row[0] for row in cursor.execute("SELECT id FROM chat").fetchall())
        for chat_id in updates.keys():
            if chat_id not in existing_ids:
                cursor.execute(
                    "INSERT INTO chat (id, title, created_at) VALUES (?, ?, datetime('now'))",
                    (chat_id, f"Chat for {chat_id}",)
                )
        conn.commit()

        # Step 3: Ensure required columns exist
        cursor.execute("PRAGMA table_info(chat)")
        columns = [row[1] for row in cursor.fetchall()]
        if "knowledge_id" not in columns:
            cursor.execute("ALTER TABLE chat ADD COLUMN knowledge_id TEXT")
        if "requirements" not in columns:
            cursor.execute("ALTER TABLE chat ADD COLUMN requirements TEXT")
        conn.commit()
        print("✅ Columns 'knowledge_id' and 'requirements' ready in 'chat' table.")

        # Step 4: Apply updates
        for chat_id, (knowledge_id, requirements) in updates.items():
            if knowledge_id is not None:
                cursor.execute("""
                    UPDATE chat
                    SET knowledge_id = ?, requirements = ?
                    WHERE id = ?
                """, (knowledge_id, requirements, chat_id))
            else:
                cursor.execute("""
                    UPDATE chat
                    SET requirements = ?
                    WHERE id = ?
                """, (requirements, chat_id))

        conn.commit()
        conn.close()
        print("✅ Chat updates applied.")

    # ----------------------------------------------------------------------------
    # Function: database_retriever
    # Description:
    #   Retrieves stored metadata (knowledge_id and requirements) for a chat ID.
    #   Verifies existence of the database and table before attempting access.
    # ----------------------------------------------------------------------------
    def database_retriever(self, db_path, chat_id):
        """
        Retrieves the stored 'knowledge_id' and 'requirements' for a given chat ID
        from the local SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
            chat_id (str): The chat ID to fetch data for.

        Returns:
            tuple(str, str): (knowledge_id, requirements) if found.
            bool: False if not found or on error.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Step 1: Verify chat table exists
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='chat';
            """)
            if not cursor.fetchone():
                print("Table 'chat' does not exist.")
                conn.close()
                return False

            # Step 2: Check if chat table has records
            cursor.execute("SELECT COUNT(*) FROM chat;")
            count = cursor.fetchone()[0]
            if count == 0:
                print("Table 'chat' exists but is empty.")
                conn.close()
                return False

            # Step 3: Fetch the desired record
            cursor.execute("""
                SELECT knowledge_id, requirements FROM chat WHERE id = ?
            """, (chat_id,))
            result = cursor.fetchone()
            conn.close()

            if result:
                knowledge_id, requirements = result
                return knowledge_id, requirements
            else:
                print(f"No record found with chat_id: {chat_id}.")
                return False

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            conn.close()
            return False



# ----------------------------------------------------------------------------
# 1.0 PROMPT GENERATOR FUNCTION
# Description: Generates a structured user and system prompt for CV evaluation
# ----------------------------------------------------------------------------
    def prompt_gen_user_system(self, requirements, data):
        """
        Generates user and system prompts to guide the LLM in evaluating a candidate's CV
        against a set of job requirements with prioritization.

        Parameters:
        - requirements (str): The job description or requirements string, optionally with priorities.
        - data (str): The content of the candidate's CV or profile to be evaluated.

        Returns:
        - Tuple[str, str]: A pair of strings representing:
            - user_message: the formatted input including candidate CV and requirements
            - system_message: the system role message to set the behavior/persona of the LLM
        """

        # ----------------------------------------------------------------------------
        # SYSTEM MESSAGES – Define role of the assistant (LLM) for evaluation
        # ----------------------------------------------------------------------------
        system_message_L10 = """
            You are an HR assistant who evaluates multiple CVs for given set of job requirements
        """

        system_message_L11 = (
            """ You are an HR assistant who evaluates multiple CVs to find the best candidates for job requirements or specific skills. 
            Provide clear, evidence-based rankings or summaries or a score when asked about expertise. Keep responses professional and focused on helping hiring decisions."""
        )

        system_message_L20 = (
            """ You are an HR agent evaluating a candidate against a list of job requirements, each with an assigned priority indicating its importance. (higher priority implies higher value)
            When scoring the candidate from 0 to 100, consider both how well the candidate meets each requirement and the relative priority of that requirement.
            Requirements with higher priority should have a greater influence on the overall score than lower-priority ones.
            Return only the final numeric score without any explanations or additional text.
            """
        )

        system_message_21 = (
            """ You are an expert HR assistant who evaluates multiple candidate CVs against specific job requirements. 

            Your task is to carefully analyze the candidate's CV and the full list of requirements. Then:

            1. Assign a score out of 100 reflecting how well the candidate meets all the requirements.
            2. Provide a concise summary explaining why you gave this score.
            - Mention which important requirements the candidate fulfills.
            - Highlight any key requirements that are missing or weak.

            Keep your response professional, clear, and focused on helping hiring decisions."""
        )

        system_message_22 = (
            """You are an expert HR assistant specialized in evaluating candidate CVs strictly against detailed job requirements. Your goal is to:

            1. Carefully analyze the candidate's CV and the full list of job requirements.
            2. Assign a strict numeric score from 0 to 100 representing how well the candidate meets ALL the requirements holistically.
            3. Use the full 0-100 range:
            - Scores above 80 indicate strong suitability with most or all requirements met.
            - Scores below 40 indicate poor suitability with many critical requirements unmet.
            - Scores in between reflect partial or moderate fit.
            4. Consider both technical skills, experience, certifications, projects, and leadership where relevant.
            5. Avoid assigning the same score to different candidates unless their fit is truly identical.
            6. Do NOT include any explanations, comments, or reasoning in your output. Output ONLY the numeric score as an integer or decimal number.
            7. When the candidate partially meets certain requirements or lacks direct experience, reduce the score accordingly but reflect transferable skills, related experience, or relevant knowledge proportionally.
            8. Focus on key required skills 
            9. Ignore irrelevant details that do not affect the candidate's suitability for the job.
            10. Respond ONLY with the score in numeric format.
            """
        )

        # ----------------------------------------------------------------------------
        # USER MESSAGES – Contains candidate CV and job requirement context
        # ----------------------------------------------------------------------------
        user_message_L11 = f"""
        <context>
        {data}
        </context>

        <requirements>
        {requirements}
        </requirements>

        Evaluate the candidate against the given requirements, which are in the format: requirement, priority.

        Score the candidate from 0 to 100 based on how well each requirement is fulfilled, considering its priority.

        ONLY RETURN A NUMBER BETWEEN 0 AND 100.  
        DO NOT INCLUDE ANY TEXT OR EXPLANATIONS.
        """

        user_message_L10 = f"""
        <requirements>
        {requirements}
        </requirements>

        Evaluate how well the candidate matches the requirements.

        Score from 0 to 100, where 0 = no match and 100 = perfect match.

        ONLY RETURN A NUMBER BETWEEN 0 AND 100.  
        DO NOT INCLUDE ANY TEXT OR EXPLANATIONS.
        """

        # ----------------------------------------------------------------------------
        # OUTPUT SELECTED PROMPT PAIR (USER , SYSTEM)
        # ----------------------------------------------------------------------------
        return user_message_L11, system_message_L20

# ----------------------------------------------------------------------------
# Function: call_openai_llm
# Description: Calls OpenAI-compatible endpoint via Open WebUI with custom body
# ----------------------------------------------------------------------------    
    def call_openai_llm(
        self, user_message: str, model_name: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Calls the OpenAI API (via Open WebUI) using a standard chat completion endpoint.

        Parameters:
        - user_message (str): The main user prompt to send to the model.
        - model_name (str): The model ID to be used for inference.
        - messages (List[dict]): Conversation context/history.
        - body (dict): Additional parameters including streaming, system/user content, etc.

        Returns:
        - Union[str, Generator, Iterator]: Final or streamed model response.
        """
        print(f">>>> LLM PROCESSING ..........")
        
        # For debugging: disable streaming and capture complete response
        body['stream'] = False
        
        # Print user info if available in body
        if "user" in body:
            print("-.-" * 50)
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print(f"# Body: {body}")
            print("-.-" * 50)
        
        try:
            # Make POST request to Open WebUI's /chat/completions endpoint
            r = requests.post(
                url=f"{self.valves.OPENAI_BASE_URL}/v1/chat/completions",
                json={
                    **body,
                    "model": model_name,
                    "temperature": 0.1,
                    "top_p": 1,
                },
                headers={
                    "Authorization": f"Bearer {self.token}"
                },
            )
            r.raise_for_status()  # Catch HTTP error responses early

            # Handle streaming response
            if body["stream"]:
                response = ""
                print(f"Streamed response object: {r.iter_lines()}")
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")

                        if decoded_line.startswith("data: "):
                            json_part = decoded_line[len("data: "):]

                            if json_part.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(json_part)
                                delta = data["choices"][0]["delta"]
                                content = delta.get("content", "")
                                response += content
                                print(content, end="", flush=True)
                            except Exception as e:
                                print(f"\n[Error parsing chunk]: {e}\nChunk: {decoded_line}")
                return response
            
            # Handle full (non-streamed) response
            else:
                print(r.json())
                for i, choice in enumerate(r.json()["choices"]):
                    self.debug(f"Result {i+1}: {choice['message']['content']}")
                
                response = r.json()["choices"][0]["message"]["content"]
                
                # Remove <think> sections if present
                if "<think>" in response:
                    index = response.find("</think>")
                    response = response[index + len("</think>"):]
                
                self.debug("STREAM - FALSE")
                return response

        except Exception as e:
            return f"Error: {e}"

    
                
    # ----------------------------------------------------------------------------
    # Function: call_llama_cpp
    # Description: Makes a call to a local Llama.cpp instance running on port 8081
    # ----------------------------------------------------------------------------
    def call_llama_cpp(self, prompt: str, model: str = "hragent_pipeline"):
        """
        Calls the locally hosted Llama.cpp model via HTTP API.

        Parameters:
        - prompt (str): The user query or content to be sent.
        - model (str): The model identifier (default is 'hragent_pipeline').

        Returns:
        - str: Model-generated response or error message.
        """
        url = "http://127.0.0.1:8081/v1/completions"
        headers = {'Content-Type': 'application/json'}
        
        data = {
            "prompt": prompt,
            "max_tokens": 3000,
            "model": model
        }

        print(" ******** EVALUATION ********  ")

        try:
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("choices", [{}])[0].get("text", "No response")
            else:
                return f"Error: {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
        
# ----------------------------------------------------------------------------
# Function: call_local_llm
# Description: LLM call using Open WebUI 
# ----------------------------------------------------------------------------
    def call_local_llm(
        self, user_message: str, model: str, body: dict, token: str
    ) -> Union[str, Generator, Iterator]:
        """
        Calls the Open WebUI-compatible LLM endpoint with token authentication.

        Parameters:
        - user_message (str): Input prompt from user.
        - model (str): The model to use (e.g., llama3, custom RAG pipeline).
        - body (dict): Full body of the request (messages, stream flag, etc.).
        - token (str): JWT or API token for authorization.

        Returns:
        - Union[str, Generator, Iterator]: Full response or line-streamed output.
        """
        print(f"LLM: {model}")
        print('\n')
        print("######################################")
        print(f"# Message: {user_message}")
        print(f"# Body: {body}")
        print("######################################")

        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_BASE_URL}/v1/chat/completions",
                json={**body, "model": model},
                headers={"Authorization": f"Bearer {token}"},
                stream=body.get("stream", False)
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

        
# ----------------------------------------------------------------------------
# Function: debug
# Description: Custom debug printer based on valves.DEBUG flag
# ----------------------------------------------------------------------------
    def debug(self, *args, **kwargs):
        """
        Prints debug messages if DEBUG mode is enabled in the valves configuration.
        
        Parameters:
        - *args, **kwargs: Variable argument messages to be printed.
        """
        if self.valves.DEBUG:
            print(*args, **kwargs)


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
        print('-' * 50)
        print("PIPE FUNCTION ACTIVATED")
        print(f"token :{self.token}")
        self.token =  self.valves.TOKEN
        print(f"token :{self.token}")
        print('-' * 50)
        print(f"pipe:{__name__}")
        print(self.valves, body)
        self.analyze_DB(self.valves.DB_PATH)
        
        if self.valves.DEBUG:
            print("------------------DEBUG MODE : ON--------------------", '\n')
        else:
            print("------------------DEBUG MODE : OFF--------------------", '\n')
        
        # ----------------------------------------------------------------------------
        # INITIAL STATE AND FILE COUNT CHECKING
        # ----------------------------------------------------------------------------
        chat_id = self.get_current_chat_id(token=self.token)
        added_file_count = len(self.get_file_names_ids(self.token)) - self.file_count
        
        self.debug(f"chat_id : {chat_id}")
        self.debug(f"added_file_count is :{added_file_count}")
        self.debug(f"self.memory_requirments_dict : {self.memory_requirments_dict}")

        # ----------------------------------------------------------------------------
        # DETERMINE STAGE BASED ON DB RECORD AND FILE COUNT
        # ----------------------------------------------------------------------------
        if self.database_retriever(self.valves.DB_PATH, chat_id):
            if added_file_count > 0:
                self.set_state("CV_PROCESSOR")
                self.debug("MODE:CV_PROCESSOR")
            else:
                self.set_state("GENERAL")
                self.debug("MODE:GENERAL")
        else:
            self.set_state("REQUIRMENTS_UPDATE")
            self.debug("MODE:REQUIRMENTS_UPDATE")

        # ----------------------------------------------------------------------------
        # STAGE: REQUIRMENTS_UPDATE – Handle job requirement entry
        # ----------------------------------------------------------------------------
        if self.is_state("REQUIRMENTS_UPDATE"):
            print('-' * 10 + " Requiremnts Checking")
            self.file_count = len(self.get_file_names_ids(self.token))
            if self.is_valid_requirements(user_message, chat_id):
                updates = {
                    chat_id: (None, user_message),
                }
                self.database_update(self.valves.DB_PATH, updates)
                self.debug("requirements being added")
                yield "Requiremnts Verified - please enter the CV's for process"
            else:
                self.debug("requirements are not in the expected format")
                yield "Please re submit the requiremnts"

        # ----------------------------------------------------------------------------
        # STAGE: CV_PROCESSOR – Process and evaluate CVs against requirements
        # ----------------------------------------------------------------------------
        elif self.is_state("CV_PROCESSOR"):
            print('-' * 10 + " CV's processing")
            
            knowledge_id, requirement = self.database_retriever(self.valves.DB_PATH, chat_id=chat_id)
            self.debug(f"requirements connected --- requirements are : {requirement[:100]} ...")

            # Create knowledge base if not already created
            if self.check_knowledge_exsistence(token=self.token, name=chat_id) == False:
                knowledge_id = self.create_knowledge(
                    token=self.token,
                    name=chat_id,
                    description="Set of candidate cv's for a job"
                )
                self.debug(f"NEW Knowledge base created : {knowledge_id}")
                
                updates = {
                    chat_id: (knowledge_id, requirement),
                }
                self.database_update(self.valves.DB_PATH, updates)
            else:
                self.debug("Knowledge base exists")

            # Refresh file count and get newly uploaded files
            added_file_count = len(self.get_file_names_ids(self.token)) - self.file_count
            self.debug(f"added_file_count is :{added_file_count}")
            self.file_count = len(self.get_file_names_ids(self.token))

            file_ids = self.get_file_ids(token=self.token, count=added_file_count)
            self.debug(f"uploaded file id's : {file_ids}")
            processing_ids = file_ids
            # Add uploaded files to knowledge base
            for id in file_ids:
                self.add_file_to_knowledge(token=self.token, knowledge_id=knowledge_id, file_id=id)
                self.debug(f"new file:{id} added to the knowledge base")

            # Retrieve and process each CV in the knowledge base
            #cvs = self.get_knowledge_details(token=self.token, knowledge_id=knowledge_id) won't work with higher versions 5.20
            cvs = self.get_file_details(token=self.token, file_ids=processing_ids)
            self.debug("Get the knowledge base")
            self.debug(f"processed_ids : {processing_ids}")
            self.debug(f"cv's content : {cvs[:1]}")
            finished_ids = []
            for cv in cvs:
                id = cv[0]
                name = cv[1]
                data = cv[2]

                if id in finished_ids:
                    continue
                
                finished_ids += [id]
                self.debug(f"processed_ids updated : {finished_ids}")
                self.debug(f"PROCESSING.....{name}")

                yield ('\n' + '-' * 10)
                yield (f"PROCESSING.....{name}")
                yield ('-' * 10 + '\n')
                yield ('\n')

                # Multi-pass evaluation (e.g., re-ranking)
                for i in range(self.valves.NUM_RE_PROCESS):
                    yield ('\n' + '-' * 10)
                    yield (f"evaluation-{i} : ")

                    # Generate prompt (user + system messages)
                    local_user_message, system_message = self.prompt_gen_user_system(requirement, data)

                    dummy_body = body
                    dummy_body["messages"] = [{"role": 'system', "content": system_message}]
                    dummy_body["messages"].append({"role": 'user', "content": local_user_message + user_message})
                    
                    time.sleep(1)  # Delay to avoid rate limiting
                    
                    # Call LLM through OpenWebUI
                    response = self.call_openai_llm(local_user_message + user_message, self.valves.MODEL, messages, dummy_body)
                    self.evaluation_results.setdefault(name, []).append(response)

                    yield str(response)

            yield (str(self.evaluation_results))

        # ----------------------------------------------------------------------------
        # STAGE: GENERAL – Generic chat interface fallback
        # ----------------------------------------------------------------------------
        elif self.is_state("GENERAL"):
            yield (self.call_openai_llm(user_message, self.valves.MODEL, messages, body))

        # ----------------------------------------------------------------------------
        # STAGE: UNDEFINED – Safety fallback
        # ----------------------------------------------------------------------------
        else:
            yield "NON-SPECIFIED STAGE"

        return 1