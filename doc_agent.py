#Document comparison agent
"""
Pipelined - Implemented on open web ui for Document Comparison -> Identify contradictions

1. Add two files in to the chat
2. Point out the contradictions between the two documents
"""
# Standard library imports
import os
import re
import csv
from typing import List, Union, Generator, Iterator, Optional, ClassVar
from typing import List, Dict, Tuple
from typing_extensions import TypedDict, Literal

# Third-party imports
import requests
import bs4
import torch
from sklearn.cluster import AgglomerativeClustering

# Sentence Transformers
from sentence_transformers import SentenceTransformer, util, SimilarityFunction, CrossEncoder, SparseEncoder

# Pydantic
from pydantic import BaseModel, Field

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain import hub

# LangGraph imports
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# LangFuse
from langfuse.langchain import CallbackHandler

from langchain_core.documents import Document

import shutil
import time

######################################################### Things to do #########################################################
"""
2. use the same agentic chunks to used to create labels 

issues
If the files won't uploaded but choose from panel it will update the swagger be so code identify as file upload when extracting the etxt from the be

"""
######################################################### Things to do #########################################################


################################################################################################################
#                                               INITIALIZATION
################################################################################################################

# ----------------------------------------------------------------------------
# 1. Langfuse Callback Handler for LangChain Tracing
# ----------------------------------------------------------------------------
# Used for tracing LLM calls, logging, and debugging within LangChain pipelines.
langfuse_handler = CallbackHandler()

#api_key = "78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF"
# ----------------------------------------------------------------------------
# 2. LLM Initialization (Ollama Cloud)
# ----------------------------------------------------------------------------
# Configure the LLM with Ollama cloud models. You can pass the API key directly
# or use environment variables. Adjust model parameters as needed.
MODEL = "gpt-oss:120b"  # Available models: ["deepseek-v3.1:671b-cloud", "gpt-oss:20b-cloud", "gpt-oss:120b-cloud", "kimi-k2:1t-cloud", "qwen3-coder:480b-cloud"]

llm = ChatOpenAI(
    model=MODEL,
    api_key="78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF",  # optional, or set via env
    base_url="https://ollama.com/v1",
    # Optional tuning parameters:
    # temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # organization="...",
)

# ----------------------------------------------------------------------------
# 3. Embedding Model Setup
# ----------------------------------------------------------------------------
# Using HuggingFace all-MiniLM-L6-v2 for generating embeddings for labels and chunks
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------------------------------------------------------
# 4. Pipeline Hyperparameters
# ----------------------------------------------------------------------------
THRESHOLD_FOR_LABEL_SELECTION = 0.5  # Minimum similarity score to match labels (lower = more permissive)
TYPE_OF_CHUNK_RETRIEVAL = "label_bind"  # ["semantic_search", "label_bind", "both"]
PAGE_SIZE_FOR_CHUNKING = 2000  # Number of characters per simulated page when chunking raw text
DEEP_LEVEL = 'safe'  # Level of label extraction detail
NUM_OF_CHUNKS_RETRIVE_FOR_EACH_LABEL = 5  # Top-k chunks to retrieve per label
NUM_OF_LABELS_SELECTED_FOR_EACH = 1  # How many top labels to select for comparison

# ----------------------------------------------------------------------------
# 5. Local Paths and Memory
# ----------------------------------------------------------------------------
LOCAL_FOLDER_PATH = "/home/tharaka/thakshana/doc_comparison/data"  # Folder with PDF files
memory = {}  # Will store mapping: label → page index
all_docs_global = []  # Global storage for all processed documents


################################################################################################################
#                            DOCUMENT COMPARISON AGENT
################################################################################################################



##############################################################
# 1. CHUNK CREATION
##############################################################

# ----------------------------------------------------------------------------
# 1.1 Recursive Chunking
# ----------------------------------------------------------------------------
def split_creation(
    all_docs: List[List[Document]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    debug: bool = True
) -> Dict[str, List[Document]]:
    """
    Split LangChain Document objects into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        all_docs (List[List[Document]]): Nested list of LangChain Document objects.
            Each sublist corresponds to a document and contains page-level Document objects.
        chunk_size (int, optional): Maximum number of characters per chunk. Defaults to 1000.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 200.
        debug (bool, optional): If True, print debug info and example splits. Defaults to False.

    Returns:
        Dict[str, List[Document]]: Dictionary of split documents.
            Format: { "doc 0": [split1, split2, ...], "doc 1": [split1, split2, ...] }
            Each split is a Document object.
    """

    all_docs_splits: Dict[str, List[Document]] = {}

    for i, doc in enumerate(all_docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(doc)
        all_docs_splits[f"doc {i}"] = splits

    if debug:
        print(f"Total documents split: {len(all_docs_splits)}")
        for key, value in all_docs_splits.items():
            print(f"{key}: {len(value)} splits created")
        # Example of first splits
        # for doc_key in list(all_docs_splits.keys())[:2]:
        #     print(f"Sample split for {doc_key}: {all_docs_splits[doc_key][0]}")

    return all_docs_splits

# ----------------------------------------------------------------------------
# 1.2 Agentic Chunking (Semantic / LLM-based)
# ----------------------------------------------------------------------------
def agentic_split_creation(
    all_docs: List[List[Document]],
    debug: bool = False,
    use_context2: bool = False
) -> Dict[str, List[Document]]:
    """
    Agentic chunking of LangChain Document pages using an LLM.

    Args:
        all_docs (List[List[Document]]): Nested list of Document objects (pages per doc).
        debug (bool, optional): If True, prints debug info. Defaults to False.
        use_context2 (bool, optional): If True, uses previous/next page context. Defaults to False.

    Returns:
        Dict[str, List[Document]]: Dictionary of agentically split Document objects.
            Format: { "doc 0": [chunk1, chunk2, ...], "doc 1": [chunk1, chunk2, ...] } each chunks are document object it self
    """

    # Chunking instructions aimed at semantic differences
    chunking_prompt = """
    You are a semantic chunking agent. Divide the given page into chunks based on **meaning and conceptual differences**:
    - Identify distinct semantic ideas, topics, or points of discussion.
    - Each chunk should represent a coherent concept or argument.
    - Preserve all information; do NOT omit any sentences, numbers, names, or details.
    - Maintain the original order of content.
    - Avoid arbitrary splits based on length; only split when the meaning shifts.
    - Minimize the number of chunks while ensuring each captures a unique semantic point.
    - Output a list of chunks (no summaries, labels, or explanations).
    """
    chunking_prompt_a = """
    You are a chunking agent. Your task is to divide the given page (Context 1) into several logical chunks.
    - Create boundaries only where necessary, based on the content’s meaning.
    - Remove unnecessary indentations as they do not affect the main idea.
    - Ensure each chunk represents a coherent section of the text.
    """
    chunking_prompt_b = """
        You are a chunking agent. Your task is to divide the given page (Context 1) into several logical chunks.
        - Create boundaries only where necessary, based on the content’s meaning.
        - Remove unnecessary indentations as they do not affect the main idea.
        - Ensure each chunk represents a coherent section of the text.
        - Do NOT omit, summarize, or ignore any information from the context.
        - Every sentence, number, date, name, or detail must be preserved in its entirety within the chunks.
        - Maintain the original order of the content within and across chunks.
        - Try to use lesser no of chunking as possible
            EX:
                3.2. GPT-3
            Launched in 2020, GPT-3 is the third and most advanced version of the GPT series, featuring several enhancements over GPT-2. GPT-3 is pretrained on a massive dataset called the WebText2, which contains hundreds of gigabytes of text from diverse sources, including web pages, books, and articles [111]. The model is significantly larger than GPT-2, with 175 billion parameters, making it one of the largest AI language models available. GPT-3 excels at various NLP tasks, such as text generation, summarization, translation, and code generation, often with minimal fine-tuning. The model's size and complexity allow it to generate more coherent, context-aware, and human-like text compared to GPT-2. GPT-3 is available through the OpenAI API, enabling developers and researchers to access the model for their applications [112].
            --------------------------------------------------
            Here are some of the pros and cons of GTP-3.
            (c) Pros:
            (i) Wide range of natural language processing tasks: GPT-3 can be used for a wide range of natural language processing tasks, including language translation, text summarization, and question answering.
            (ii) High-quality text generation: GPT-3 is known for its ability to generate high-quality human-like text, which has a wide range of applications, including chatbots and content creation.
            (iii) Large-scale architecture: GPT-3's architecture is designed to handle large amounts of data, which makes it suitable for applications that require processing of large datasets.
            (iv) Zero-shot learning capabilities: GPT-3 has the ability to perform some tasks without explicit training, which can save time and resources.
            --------------------------------------------------
            (d) Cons:
            (i) Large computational requirements: GPT-3's large model size and complex architecture require significant computational resources, making it difficult to deploy on devices with limited computational resources.
            (ii) Limited interpretability: GPT-3's complex architecture makes it difficult to interpret its internal workings, which can be a challenge for researchers and practitioners who want to understand how it makes its predictions.
            (iii) Language-specific: Like other transformer-based models, GPT-3 is primarily trained on English language data and may not perform as well on other languages without additional training or modifications.
            (iv) Ethical concerns: GPT-3's capabilities raise ethical concerns about its potential misuse and the need for responsible deployment.

            we can have all of the above into one chunk
        """

    chunking_prompt_c = """
        You are a **Chunking Agent**, specialized in dividing text into semantically coherent sections.

        Your goal is to divide the given text (Context 1) into the *smallest necessary number of chunks* while ensuring each chunk is complete and meaningful.

        ### STRICT INSTRUCTIONS:

        1. **Do not omit or summarize** any information.  
        Every word, number, name, and symbol from the original text must appear in your output.

        2. **Preserve full sentences and paragraphs.**  
        Never cut through a sentence or word (e.g., avoid “moneti- zation”).  
        Join broken or hyphenated words if they belong together.

        3. **Use as few chunks as possible** while keeping logical coherence.  
        Only split when there is a clear semantic or structural boundary (e.g., heading change, list start, or topic shift).

        4. **Remove unnecessary indentation or line breaks** that do not affect meaning.

        5. Do not include any explanations or commentary outside the chunks.


        """

    # Structured output schema
    class Classification(BaseModel):
        chunks: List[str] = Field(description=chunking_prompt_c)

    structured_llm = llm.with_structured_output(Classification)

    tagging_prompt = ChatPromptTemplate.from_template("""
    Context 1: Current page to analyze
    Context 2: {context2}

    Passage:
    {context1}
    """)

    all_docs_splits_f: Dict[str, List[Document]] = {}

    for doc_idx, doc in enumerate(all_docs):
        doc_splits: List[Document] = []

        for page_no, page in enumerate(doc):
            context2_text = "None"
            if use_context2:
                prev_page = doc[page_no - 1].page_content if page_no > 0 else ""
                next_page = doc[page_no + 1].page_content if page_no < len(doc) - 1 else ""
                context2_text = prev_page + "\n" + next_page

            prompt = tagging_prompt.invoke({"context1": page.page_content, "context2": context2_text})
            print(f"Processing : {page_no+1}/{len(doc)}")   
            response = structured_llm.invoke(prompt)
            print(f"Doc {doc_idx} Page {page_no} → {len(response.chunks)} chunks")

            # Convert chunks into Document objects
            for chunk_text in response.chunks:
                doc_splits.append(Document(
                    page_content=chunk_text,
                    metadata={"source": "user_input", "length": len(chunk_text)}
                ))

            if debug:
                print(f"Doc {doc_idx} Page {page_no} → {len(response.chunks)} chunks")
                print("-" * 100)

        all_docs_splits_f[f"doc {doc_idx}"] = doc_splits

    if debug:
        for key, value in all_docs_splits_f.items():
            print(f"{key}: {len(value)} chunks created")

    return all_docs_splits_f


##############################################################
# 2. VECTOR EMBEDDINGS
##############################################################

# ----------------------------------------------------------------------------
# 2.1 Create Chroma Vectorstores
# ----------------------------------------------------------------------------
def vectorstores_creation(
    all_docs_splits: Dict[str, list[Document]],
    embedding_model
) -> Dict[str, Chroma]:

    """
    Create Chroma vector stores for each set of document splits.

    Args:
        all_docs_splits (Dict[str, List[Document]]): 
            Dictionary of document splits.
            Format: { "doc 0": [split1, split2, ...], "doc 1": [split1, ...] }
        embedding_model: LangChain-compatible embedding model.

    Returns:
        Dict[str, Chroma]: Dictionary of Chroma vectorstores.
            Format: { "doc 0": Chroma_vectorstore, "doc 1": Chroma_vectorstore, ... }

    Notes:
        - Each vectorstore is persisted in a unique directory.
        - Existing directories are removed before creating a new vectorstore.
    """

    print("---------Vectorstores Creation---------")
    vectorstores: Dict[str, Chroma] = {}
    timestamp = int(time.time())

    for doc_key, splits in all_docs_splits.items():
        persist_dir = f"./db_file_{doc_key}_{timestamp}"

        # Remove old DB directory if it exists
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print(f"Deleted existing DB at {persist_dir}")

        # Create Chroma vectorstore for this document
        vectorstores[doc_key] = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        print(f"Vectorstore created for {doc_key} with {len(splits)} splits.")

    return vectorstores

##############################################################
# 3. LABEL EXTRACTION
##############################################################

# ----------------------------------------------------------------------------
# 3.1 Build System Message for LLM
# ----------------------------------------------------------------------------
def build_system_message(deep_level):
    base_message = (
        "You are a logical extraction agent. Your task is to identify and extract ALL possible "
        f"{deep_level} conceptual labels from the user's document text. "
        "Each label represents a distinct logical or semantic idea expressed in the document, "
        "which could later be compared against other documents to detect similarities or contradictions.\n\n"
        "Strict rules:\n"
        "- Output ONLY clean, meaningful label titles.\n"
        "- Do NOT include author names, affiliations, numbers, dates, citations, or examples.\n"
        "- Avoid repeating labels that refer to the same concept using different wording.\n"
        "- Preserve capitalization as it appears in the text.\n"
        "- Avoid commentary, explanation, or numbering.\n"
        "- Output must be a simple list (one label per line).\n\n"
    )

    if deep_level.lower() == "high":
        level_details = (
            "Focus ONLY on broad, high-level conceptual labels that define the overall structure or intent "
            "of the document (e.g., 'Research Problem', 'Proposed Solution', 'Results Summary'). "
            "Ignore internal breakdowns or fine-grained details."
        )
    elif deep_level.lower() == "medium":
        level_details = (
            "Extract both the main conceptual labels and key sub-concepts that describe meaningful internal ideas "
            "(e.g., 'Data Processing Method', 'Model Training Strategy', 'Error Analysis'). "
            "Avoid sentence fragments or inline phrases that do not represent self-contained ideas."
        )
    elif deep_level.lower() == "deep":
        level_details = (
            "Extract every distinct conceptual label possible — including all sub-ideas, internal processes, "
            "and reasoning segments that represent separate logical components. "
            "Each label should describe a meaningful part of the document’s reasoning, approach, or evidence. "
            "Avoid listing trivial or linguistic patterns."
        )
    else:
        level_details = "Extract all meaningful conceptual labels representing distinct logical ideas.Only upto 3 for a given chunk"

    example = (
        "\n\nExample:\n"
        "**Text:** 'This paper proposes a transformer-based architecture to improve translation accuracy.'\n"
        "→ Possible label: **Proposed Architecture**\n"
        "**Text:** 'The evaluation shows significant improvement over baselines.'\n"
        "→ Possible label: **Performance Evaluation**"
    )

    return base_message + level_details + example


# ----------------------------------------------------------------------------
# 3.2 Label Creation (LLM or Non-LLM)
# ----------------------------------------------------------------------------
def label_creation(all_docs,deep_level = "high") -> dict:
    """
    Input:
        all_docs : langchain document object
    Output:
        pages: dict  
            Format: {doc 1: [page1, page2, ..., pagen], doc 2: [page1, page2, ..., pagem]}
        responses: dict  
            Format: {doc 1: [response1_set, response2_set, ..., responsen_set], 
                     doc 2: [response1_set, response2_set, ..., responsem_set]}
        senteces: dict  
            Format: {doc 1: [sentences_set1], doc 2: [sentences_set1]}
        sentences_list: dict  
            Format: {doc 1: [sentence1, sentence2, ..., sentencen], doc 2: [sentence1, sentence2, ..., sentencem]}
    """

    
    sys_msg = build_system_message(deep_level)
    # Extract pages from all documents
    print("---------Pages Extraction---------")
    pages = {}
    for i, doc in enumerate(all_docs):
        pages[f"doc {i}"] = []
        for j in doc:
            pages[f"doc {i}"].append(j.page_content)

    # Extract responses (headings) for each page using LLM
    print("---------Responses Extraction---------")
    responses = {}
    for i in range(len(pages)):
        responses[f"doc {i}"] = []
        count = 0
        for page in pages[f"doc {i}"][:10]:
            
            prompt = f"""
            You need to understand the content of this page: {page}

            Think about what aspects could differ if the same topic appeared in another similar document.
            Based only on this page and your careful understanding, identify the **LABELS** that describe possible differences to check. Make the LABELS more detailed.
            List **only** the LABELS, without any explanations or details.
            """
            system_message =  SystemMessage(content=sys_msg)
            human_message = HumanMessage(content=prompt)
            response = llm.invoke([system_message, human_message]) #response = set of responses \n sepearate
            responses[f"doc {i}"].append(response.content) # responses[key] = [[],[]] list of lists


    # Combine all responses into a single string per document
    print("---------Sentences Extraction---------") 
    senteces = {}
    for key, value in responses.items():
        senteces[key] = ""
        
        for i in range(len(value)):
            senteces[key] += value[i]

    # Split the combined responses into a list of sentences per document
    sentences_list = {}
    for key, value in senteces.items():
        sentences_list[key] = [line for line in value.split("\n") if line.strip()]

    return sentences_list


def label_creation_with_memory(
    all_docs: List[List[Document]],
    deep_level: str = "high"
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """
    Extract descriptive labels from document pages using an LLM and build memory mapping.

    Args:
        all_docs (list[list[Document]]): Nested list of LangChain Document objects.
            Each sublist corresponds to a document and contains page-level Document objects.
        deep_level (str, optional): Level of detail for system message instructions. Defaults to "high".

    Returns:
        sentences_list (dict): Labels extracted from each document.
            Format: { "doc 0": [label1, label2, ...], "doc 1": [label1, ...] }
        memory_inside (dict): Maps each label to its corresponding page number.
            Format: { "doc 0": {label1: page_no, label2: page_no, ...}, ... }

    Notes:
        - Uses structured output via Pydantic BaseModel for LLM responses.
        - Each label is mapped to the page number it was extracted from.
        - Designed to work with LangChain LLMs supporting structured output.
    """
    
    # Define the structured output schema for LLM
    class Classification(BaseModel):
        response: List[str] = Field(
            description="List of LABELS extracted from page (no explanations)."
        )

    structured_llm = llm.with_structured_output(Classification)

    # Build system message based on desired detail level
    sys_msg = build_system_message(deep_level)

    # --- Step 1: Extract pages from all documents ---
    print("---------Pages Extraction---------")
    pages = {}
    for doc_idx, doc in enumerate(all_docs):
        pages[f"doc {doc_idx}"] = [page.page_content for page in doc]

    # --- Step 2: Extract labels for each page using LLM ---
    print("---------Responses Extraction---------")
    responses = {}
    memory_inside = {}

    for doc_idx, page_contents in pages.items():
        responses[doc_idx] = []
        memory_inside[doc_idx] = {}
        total_pages = len(page_contents)

        for page_no, page_text in enumerate(page_contents):
            prompt = f"""
            You are given the following page content:
            {page_text}

            Your task:
            1. Carefully read and understand the content of this page.
            2. Ignore irrelevant characters, extra newlines, or formatting artifacts.
            3. Think critically about how this page’s content might differ if the same topic appeared in another similar document.
            4. Based solely on this page, identify detailed and descriptive **LABELS** representing possible aspects or points of difference to check.

            Output:
            - Provide **only** a list of LABELS (no explanations or reasoning).
            """

            system_message = SystemMessage(content=sys_msg)
            human_message = HumanMessage(content=prompt)
            
            # Invoke structured LLM
            response = structured_llm.invoke([system_message, human_message])

            # Debugging output
            print(f"Loading {page_no+1}/{total_pages}.....")
            print(page_text)
            print('*' * 100)
            print(response.response)

            # Update memory mapping and responses
            for label in response.response:
                memory_inside[doc_idx][label] = page_no
            responses[doc_idx].extend(response.response)

    # Final outputs
    sentences_list = responses
    return sentences_list, memory_inside


def NO_LLM_label_creation_with_memory(
    all_docs: List[List[Document]]
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """
    Extract page-level 'labels' without using an LLM and create a memory mapping.

    Args:
        all_docs (List[List[Document]]): Nested list of LangChain Document objects.
            Each sublist corresponds to a document containing page-level Document objects.

    Returns:
        sentences_list (Dict[str, List[str]]): Page-level text for each document.
            Format: { "doc 0": [page1, page2, ...], "doc 1": [page1, ...] }
        memory_inside (Dict[str, Dict[str, int]]): Maps each page text to its page number.
            Format: { "doc 0": {page_text1: page_no, page_text2: page_no, ...}, ... }

    Notes:
        - This function does not call an LLM.
        - Each page's content is used as a 'label'.
        - Useful for testing or pipelines where LLM processing is not required.
    """

    # --- Step 1: Extract pages from documents ---
    print("---------Pages Extraction---------")
    pages = {
        f"doc {i}": [page.page_content for page in doc]
        for i, doc in enumerate(all_docs)
    }

    # --- Step 2: Create memory mapping and responses ---
    print("---------Responses Extraction---------")
    responses = {}
    memory_inside = {}

    for doc_idx, page_texts in pages.items():
        responses[doc_idx] = []
        memory_inside[doc_idx] = {}

        total_pages = len(page_texts)

        for page_no, page_text in enumerate(page_texts):
            # In this version, the 'label' is just the page content itself
            response = [page_text]

            # Debug output
            print(f"Loading {page_no + 1}/{total_pages}.....")
            print(page_text)
            print(response)
            print("*" * 100)
            

            # Update memory and responses
            for label in response:
                memory_inside[doc_idx][label] = page_no
            responses[doc_idx].extend(response)

    sentences_list = responses
    return sentences_list, memory_inside


def agentic_label_creation(
    all_docs: List[List[Document]],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """
    Agentic semantic label creation for document pages/chunks.

    Args:
        all_docs (List[List[Document]]): Nested list of LangChain Document objects.
            Each sublist corresponds to a document and contains page-level Document objects.

    Returns:
        sentences_list (Dict[str, List[str]]): Extracted labels for each document/page.
        memory_inside (Dict[str, Dict[str, int]]): Maps each label to its page index.

    Notes:
        - Uses structured LLM output via Pydantic BaseModel.
        - Labels are high-level semantic concepts per page.
    """

    # Semantic labeling instructions
    semantic_labeling_prompt = """
    You are a semantic labeling agent. Generate 1–3 high-level labels:
    - Capture key entities, roles, and actions, without relying on exact names, dates, or numbers.
    - Focus on semantics and relationships.
    - Labels must be concise and abstract enough to match similar content elsewhere.
    - Output only the list of LABELS (no explanations).
    """

    # Structured output schema
    class Classification(BaseModel):
        response: List[str] = Field(description=semantic_labeling_prompt)

    structured_llm = llm.with_structured_output(Classification)

    # Prepare tagging prompt template
    tagging_prompt = ChatPromptTemplate.from_template("""
    You are given a chunk/page of a PDF document.
    Task:
    - Identify high-level semantic labels representing this page's content.
    - Output only the list of LABELS.

    Passage:
    {context1}
    """)

    # --- Step 1: Extract page content from Document objects ---
    pages: Dict[str, List[str]] = {}
    for doc_idx, doc in enumerate(all_docs):
        pages[f"doc {doc_idx}"] = [page.page_content for page in doc]

    # --- Step 2: Process each page and collect labels ---
    responses: Dict[str, List[str]] = {}
    memory_inside: Dict[str, Dict[str, int]] = {}

    print("---------Agentic Label Extraction---------")
    for doc_idx, page_texts in pages.items():
        responses[doc_idx] = []
        memory_inside[doc_idx] = {}
        total_pages = len(page_texts)

        for page_no, page_text in enumerate(page_texts):
            # Format page text into prompt
            prompt = tagging_prompt.invoke({"context1": page_text})

            # LLM call
            response = structured_llm.invoke(prompt)

            # Debugging
            print(page_text)
            print(f"Loading {page_no+1}/{total_pages} in {doc_idx}...")
            print("*" * 100)
            print(response.response)

            # Update memory mapping and responses
            for label in response.response:
                memory_inside[doc_idx][label] = page_no
            responses[doc_idx].extend(response.response)

    sentences_list = responses
    return sentences_list, memory_inside


##############################################################
# 4. LABEL MAPPING / SIMILARITY
############################################################### 

# ----------------------------------------------------------------------------
# 4.1 Map Labels Between Documents
# ----------------------------------------------------------------------------
def mapping_labels(labels: dict, threshold: float = 0.5, k: int = 3, similarity_model: str = "all-MiniLM-L6-v2"):
    """
    Map labels between two documents based on semantic similarity.

    Args:
        labels (dict): Format: 
            {
                "doc 0": [label1, label2, ..., labeln],
                "doc 1": [label1, label2, ..., labelm]
            }
        threshold (float, optional): Minimum similarity score to consider a match. Defaults to 0.5.
        k (int, optional): Number of top matches per label. Defaults to 3.

    Returns:
        comparing_headings (list[list[str]]): List of top matched label pairs:
            [[doc0_label1, doc1_label1], [doc0_label2, doc1_label2], ...]
    """
    print("---------Embedding Creation---------")
    embeddings = {}

    # Generate embeddings for each label
    for doc_key, label_list in labels.items():
        embeddings[doc_key] = [embedding_model.embed_query(label) for label in label_list]

    # Convert to tensors for similarity calculation
    embeddings_0 = torch.tensor(embeddings["doc 0"])
    embeddings_1 = torch.tensor(embeddings["doc 1"])

    # Compute similarity matrix using cosine similarity
    model = SentenceTransformer(similarity_model)
    similarities = model.similarity(embeddings_0, embeddings_1)  # nxm similarity matrix

    # Get top-k matches for each label
    values, indices = torch.topk(similarities, k=k, dim=1)  # values: scores, indices: positions
    values = values.tolist()
    indices = indices.tolist()

    comparing_headings = []
    for i, row_values in enumerate(values):
        # Skip if highest similarity is below threshold
        if row_values[0] < threshold:
            continue

        top_index = indices[i][0]
        comparing_headings.append([labels["doc 0"][i], labels["doc 1"][top_index]])

        # Debugging output
        print(f"{labels['doc 0'][i]} -> {labels['doc 1'][top_index]}")
        print('-' * 50)

    print(f"---------Comparing Headings Created: {len(comparing_headings)}---------")
    return comparing_headings


##############################################################
# 5. CONTRADICTION DETECTION
##############################################################

# 5.1 LLM-based Comparison of Two Texts
# ----------------------------------------------------------------------------
def labelized_llm(context1: str, context2: str):
    """
    Compare two text contexts using an LLM and identify contradictions.

    Args:
        context1 (str): Text from the first source.
        context2 (str): Text from the second source.

    Returns:
        BaseModel: Structured output containing:
            - contradiction_result (str): 'CONTRADICTION' or 'NO CONTRADICTION'.
            - reason (Optional[str]): Explanation if a contradiction exists.
    
    Example:
        >>> response = labelized_llm("Alice likes apples.", "Alice hates apples.")
        >>> response.contradiction_result
        'CONTRADICTION'
    """

    # Define a chat prompt template for LLM comparison
    tagging_prompt = ChatPromptTemplate.from_template("""
    You are an expert analyst tasked with detecting contradictions between two contexts. 
    Carefully and logically analyze the information in both contexts. Focus **only** on the properties defined in the 'Classification' schema.

    Instructions:
    1. Compare Context 1 and Context 2 thoroughly.
    2. Consider only the context related to the context 1 within the context2 for the unrelated contexts just ignore
    2. Identify if there are any contradictions between the two contexts.
    3. Consider subtle logical conflicts, opposing statements, or reversed conditions.
    4. Ignore information not relevant to the properties being compared.
    5. Be precise and concise in your reasoning.

    Passage:
    Context 1:
    {context1}

    Context 2:
    {context2}

    Output Format:
    - CONTRADICTION if any direct logical conflict exists.
    - NO CONTRADICTION if the statements are consistent or if differences do not constitute a conflict.
    - If CONTRADICTION, provide a brief reason explaining the conflict.
    """)

    # Define the structured output schema for the LLM
    class Classification(BaseModel):
        contradiction_result: str = Field(
            description="Are there any contradictions between Context 1 and Context 2? Must be either 'CONTRADICTION' or 'NO CONTRADICTION'."
        )
        reason: Optional[str] = Field(
            default=None,
            description="Provide a brief reason only if there is a CONTRADICTION."
        )

    # Wrap LLM to enforce structured output
    structured_llm = llm.with_structured_output(Classification)

    # Format the prompt with the provided contexts
    prompt = tagging_prompt.invoke({"context1": context1, "context2": context2})

    # Call the LLM and get the structured response
    response = structured_llm.invoke(prompt)

    # Optional: print response for debugging
    print(response)

    return response

##############################################################
# 6. CONTEXT RETRIEVAL
##############################################################

# ----------------------------------------------------------------------------
# 6.1 Retrieve Top-K Relevant Documents
# ----------------------------------------------------------------------------
def get_topk_docs(
        pair: list,
        k: int = 5,
        data: list = None,
        memory: dict = None,
        type_is: str = None,
        retrievers: dict = None
    ):
    """
    Retrieve top-k relevant documents for a given pair of labels.

    Depending on the selected retrieval type, this function fetches documents from
    either retrievers (semantic search), memory-based bindings (label binding), or both.

    Args:
        pair (list[str]): A list of two label strings, e.g., [label_doc1, label_doc2].
        k (int, optional): Number of top relevant documents to retrieve. Defaults to 5.
        data (list[list[Document]], optional): Nested list of Document objects.
            Used when `type_is` is "label_bind" or "both".
        memory (dict, optional): Dictionary mapping document labels to indices
            within `data`, e.g., memory['doc 0'][label_doc1] -> index.
        type_is (str): Retrieval mode. Must be one of:
            ["semantic_search", "label_bind", "both"].
        retrievers (dict, optional): Dictionary containing retriever objects for
            "doc 0" and "doc 1". Required if using "semantic_search" or "both".

    Returns:
        tuple: (docs1, docs2)
            Each is a list of Document objects corresponding to the input pair.
        tuple: (docs1_f, docs2_f)
            Each is a string containing the combined text from the retrieved documents
            corresponding to each label in the input pair.

    """
    # Define allowed modes
    allowed_types = ["semantic_search", "label_bind", "both"]

    # Helper to join multiple pages into a single string
    format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

    # Validate mode
    if type_is is None or type_is.lower() not in allowed_types:
        raise ValueError(f"type_is must be one of {allowed_types}, got '{type_is}'")

    type_is = type_is.lower()

    # --- Case 1: Semantic search retrieval ---
    if type_is == "semantic_search":
        docs1 = retrievers["doc 0"].get_relevant_documents(pair[0], k=k)
        docs2 = retrievers["doc 1"].get_relevant_documents(pair[1], k=k)
        docs1_f, docs2_f = format_docs(docs1), format_docs(docs2)

    # --- Case 2: Label binding retrieval ---
    elif type_is == "label_bind":
        docs1 = [data[0][memory["doc 0"][pair[0]]]]
        docs2 = [data[1][memory["doc 1"][pair[1]]]]
        docs1_f, docs2_f = docs1[0].page_content, docs2[0].page_content

    # --- Case 3: Combined retrieval ---
    elif type_is == "both":
        # Retrieve from semantic search
        docs1_sem = retrievers["doc 0"].get_relevant_documents(pair[0], k=k)
        docs2_sem = retrievers["doc 1"].get_relevant_documents(pair[1], k=k)

        # Retrieve from label binding
        docs1_lbl = [data[0][memory["doc 0"][pair[0]]]]
        docs2_lbl = [data[1][memory["doc 1"][pair[1]]]]

        # Merge text representations
        docs1_f = format_docs(docs1_sem) + "\n\n" + docs1_lbl[0].page_content
        docs2_f = format_docs(docs2_sem) + "\n\n" + docs2_lbl[0].page_content

        # Merge document lists for consistency
        docs1 = docs1_sem + docs1_lbl
        docs2 = docs2_sem + docs2_lbl

    return docs1_f, docs2_f



##############################################################
# 7. FILE PROCESSING
##############################################################

# ----------------------------------------------------------------------------
# 7.1 Convert Raw Text to Document Objects
# ----------------------------------------------------------------------------
def make_alldocs_as_document_object(
        extracted_file_contents: list,
        page_size: int = 1000
    ) -> list:
    """
    Convert raw Text contents into structured Document objects, simulating page-level segmentation.

    Args:
        extracted_file_contents (list[list]): A list of files, where each file entry is:
            [file_id, filename, file_content].
        page_size (int, optional): The character length used to split each file's content into
            pseudo-pages. Defaults to 1000.

    Returns:
        list[list[Document]]: A list of lists, where each sublist corresponds to one file.
                              Each file contains a list of page-level `Document` objects with metadata.

    Notes:
        - Designed to produce a consistent structure with get_files_from_local_dir_pypdf().
    """
    print(f"Making Document Object Of Size {page_size}...")
    all_docs = []

    # Process each extracted file
    for file_entry in extracted_file_contents:
        # Expect each entry to contain [file_id, filename, file_content]
        if len(file_entry) < 3:
            print("!Skipping invalid entry:", file_entry)
            continue

        file_id, filename, file_content = file_entry[:3]
        print(f"Processing file: {filename} (ID: {file_id})")

        # Split file content into fixed-size chunks (simulating pages)
        doc_pages = [
            Document(
                page_content=file_content[i:i + page_size],
                metadata={"source": "user_input", "file_id": file_id, "filename": filename, "length": page_size}
            )
            for i in range(0, len(file_content), page_size)
        ]

        # Append list of page documents for this file
        all_docs.append(doc_pages)

    return all_docs


# ----------------------------------------------------------------------------
# 7.2 Load PDF Files from Local Directory
# ----------------------------------------------------------------------------
def get_files_from_local_dir_pypdf(folder_path: str = "/home/tharaka/thakshana/doc_comparison/data") -> list:
    """
    Load all PDF files from a local directory and return their page-level Document objects.

    Args:
        folder_path (str): Absolute or relative path to the directory containing PDF files.
                           Defaults to "/home/tharaka/thakshana/doc_comparison/data".

    Returns:
        list[list[Document]]: A list of documents, where each element corresponds to one PDF file.
                              Each PDF file is represented as a list of `Document` objects 
                              (one per page, as returned by `PyPDFLoader.load()`).
    Notes:
        - Only files with the `.pdf` extension are processed.
        - Each PDF is loaded using `PyPDFLoader` from LangChain.
    """
    all_docs = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Process only PDF files
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            # Initialize PyPDFLoader for the current file
            loader = PyPDFLoader(file_path)

            # Load document pages as a list of Document objects
            docs = loader.load()

            # Append the list of pages for this PDF to the main result
            all_docs.append(docs)

    return all_docs


    
##############################################################
# 8. Main Comparison Function
##############################################################
def compare_documents(extracted_file_contents: list):
    """
    Input:
        extracted_file_contents : list
            Currently unused, placeholder for extracted content if needed.
    Output:
        output : str
            Concatenated reasons for contradictions detected between document pairs.
    """
    print("---------STARTED---------")
    file_names_processing = []
    for file_content in extracted_file_contents:
        file_names_processing = extracted_file_contents[1]+extracted_file_contents[0] # extracted_file_contents =  ['61c701ba-91a3-480b-a283-145fc6298f4b', '130806ca141 test 1.pdf',"content"]

    print("---------Document Extraction---------")
    # Get the documents from the folder

    all_docs = []
    all_docs = make_alldocs_as_document_object(extracted_file_contents,page_size = PAGE_SIZE_FOR_CHUNKING)
    print(f"Loaded a total of {len(all_docs)} documents")
    
    # Only take the first 2 pages from each document for testing
    #all_docs = [l[:2] for l in all_docs]

    print("---------Split Creation---------")
    all_docs_splits = split_creation(all_docs)

    print("---------Vectorstores Creation---------")
    all_docs_vectorstores = vectorstores_creation(all_docs_splits, embedding_model)

    print("---------Retrievers Creation---------")
    retrievers = {}
    for key, value in all_docs_vectorstores.items():
        retrievers[key] = value.as_retriever()

    print("---------Label Creation---------")
    # Extract labels/sentences from documents
   
   # labels = agentic_label_creation(all_docs)  # sentences_list: dict {doc 1:[sentence1,...], doc 2:[sentence1,...]}

    #labels,memory = label_creation_with_memory(all_docs,DEEP_LEVEL) 
    all_docs_agentic_chunking = []
    for key,value in all_docs_splits.items():
        all_docs_agentic_chunking.append(value)
    all_docs = all_docs_agentic_chunking
    labels,memory = NO_LLM_label_creation_with_memory(all_docs)

    l1 = len(labels["doc 1"])
    l0 = len(labels["doc 0"])
    details = f"Labels created by doc0 {l1}\n Labels created by doc1 {l0}\n"

    print("---------Similarity Calculation---------")
    comparing_headings = mapping_labels(labels,threshold=THRESHOLD_FOR_LABEL_SELECTION,k=NUM_OF_LABELS_SELECTED_FOR_EACH)
    details += f"Total labels Compared: {len(comparing_headings)} \n "
    #### RETRIEVAL and GENERATION ####

    # File path to save CSV results
    output_file = os.path.join(LOCAL_FOLDER_PATH, "doc_pair_comparison_results.csv")

    # Write CSV header
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Doc_Pair", "Docs1_Content", "Docs2_Content", "Comparison_Result"])

    output = ""
    # Compare first n pairs of headings
    for pair in comparing_headings[:]:
        print(pair)
        docs1_f, docs2_f = get_topk_docs(pair,k=NUM_OF_CHUNKS_RETRIVE_FOR_EACH_LABEL, type_is=TYPE_OF_CHUNK_RETRIEVAL,data = all_docs,memory=memory,retrievers=retrievers)
        
        # Get LLM response for the document pair
        response = labelized_llm(docs1_f, docs2_f)
        print((docs1_f, docs2_f))
        print(response)

        # Append reason to output if a contradiction is found
        if (response.reason):
            output += response.reason
        output+='\n\n'

        # Append row to CSV
        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([pair, docs1_f, docs2_f, response])
            print('-' * 50)

    # Note: Dendrogram clustering logic can be applied later if needed
    #dendogram-> conncet all the nodes via low distance are conneted first method to final single cluster(n+n,n+c,c+c)  Then cut the dendogram from the threshhold value and use thoscluserts
    return details+output


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

    
################################################ 2 Functions into open-webUI api calls ######################################################
        
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
# 2.2 Get detailed content of specific files (by file IDs) --> Used when file_ids are known
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

 # ----------------------------------------------------------------------------
# 2.3 Fetches the latest (most recent) chat ID from the conversation list
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
 
################################################ 3 Graph ######################################################

# ----------------------------------------------------------------------------
# 3.1 NODES -> functions them self
# ----------------------------------------------------------------------------
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
            api_key="89faf3ae18ef4693bc964f3deca63919.2qaqcagLcSilt0trMvydIdGk",  # pass api key in directly instaed of using env vars
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
            api_key="89faf3ae18ef4693bc964f3deca63919.2qaqcagLcSilt0trMvydIdGk",  # pass api key in directly instaed of using env vars
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
            api_key="89faf3ae18ef4693bc964f3deca63919.2qaqcagLcSilt0trMvydIdGk",  # pass api key in directly instaed of using env vars
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


# ----------------------------------------------------------------------------
# 3.2 Build graph
# ----------------------------------------------------------------------------
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
            api_key="89faf3ae18ef4693bc964f3deca63919.2qaqcagLcSilt0trMvydIdGk", 
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