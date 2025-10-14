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
1. w/o making labels through llm let labels be the same chunk for efficiency and check the correctness
2. use the same agentic chunks to used to create labels 

issues
If the files won't uploaded but choose from panel it will update the swagger be so code identify as file upload when extracting the etxt from the be

"""
######################################################### Things to do #########################################################


# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()
#Initialize the llm as openai based api from ollama cloud models
#api_key = "78a7821bbadc498b9938a40aeddeb87b.CRx89un9mZspHxQblP6HbuCF"

# Tune up params
THRESHOLD_FOR_LABEL_SELECTION = 0.7 # low means farther away labels also be considered
TYPE_OF_CHUNK_RETRIEVAL = "label_bind" # ["sementic_search", "label_bind", "both"]
PAGE_SIZE_FOR_CHUNKING = 500 # CHARS
LOCAL_FOLDER_PATH = "/home/tharaka/thakshana/doc_comparison/data"
DEEP_LEVEL = 'safe'
MODEL = "gpt-oss:120b"  # available models = ["deepseek-v3.1:671b-cloud", "gpt-oss:20b-cloud", "gpt-oss:120b-cloud", "kimi-k2:1t-cloud", "qwen3-coder:480b-cloud"]
NUM_OF_CHUNKS_RETRIVE_FOR_EACH_LABEL = 5 #
NUM_OF_LABELS_SELECTED_FOR_EACH = 1 #


llm = ChatOpenAI(
    model=MODEL,
    api_key="89faf3ae18ef4693bc964f3deca63919.2qaqcagLcSilt0trMvydIdGk",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="https://ollama.com/v1",
    # temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # organization="...",
    # other params...
)
    
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
memory = {}
all_docs_global = []
################################################ 1 Core Functionalities of a Document Comparison Agent ######################################################
        
# ----------------------------------------------------------------------------
# 1.10 Make Chunks -> recursive chunk creation
# ----------------------------------------------------------------------------
def split_creation(all_docs) -> dict:
    """
    Input:
        all_docs : langchain document object
    Output:
        all_docs_splits_f : dict  
            Format: {doc 1: [split1, split2, ..., splitn], doc 2: [split1, split2, ..., splitm]} # each split is document object it self
    """

    print("---------Splits Creation---------")
    all_docs_splits = {}

    for i, doc in enumerate(all_docs):
        # Initialize text splitter with chunk size 1000 and overlap 200
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)
        all_docs_splits[f"doc {i}"] = splits

    print(len(all_docs_splits))
    # all_docs_splits -> {doc 1: [all the splits as list], doc 2: [all the splits as list]}
    print("Splits :\n", all_docs_splits["doc 0"][0], all_docs_splits["doc 1"][0], "......")  
    # show split of a selected doc

    for key, value in all_docs_splits.items():
        print(key, ": NO of splits created", len(value))


    
    return all_docs_splits

## ----------------------------------------------------------------------------
# 1.11 Make Chunks -> agentic chunk creation
# ----------------------------------------------------------------------------
def agentic_split_creation(all_docs) -> dict:
    """
    Input:
        all_docs : langchain document object
    Output:
        splits : dict  
            Format: {doc 1: [split1, split2, ..., splitn], doc 2: [split1, split2, ..., splitm]}
    """

    print("---------Agentic Splits Creation---------")
    all_docs_splits = {}

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
    # Create a chat prompt template to compare two contexts
    tagging_prompt = ChatPromptTemplate.from_template("""
    You are given two contexts 
    Context 1:Current page to analyze:.
    Context 2: Context (for understanding only the previous and next pages)

    Passage:
    Context 1:
    {context1}

    Context 2:
    {context2}


    """)

    # Define the structured output schema
    class Classification(BaseModel):
            chunks: List[str] = Field(
                description=chunking_prompt_b
            )
    # Wrap LLM with structured output
    structured_llm = llm.with_structured_output(Classification)
  

    splits = []

    for i, doc in enumerate(all_docs):
        # Initialize text splitter with chunk size 1000 and overlap 200

        for page in doc[:]:
       
            prompt = tagging_prompt.invoke({"context1": page.page_content, "context2": "None"}) #use context two if we need to provide an additonal overview
            response = structured_llm.invoke(prompt)
            splits.append(response)

        all_docs_splits[f"doc {i}"] = splits

    print(len(all_docs_splits))
    # all_docs_splits -> {doc 1: [all the splits as list], doc 2: [all the splits as list]}
    print("Splits :\n", all_docs_splits["doc 0"][0])  
    # show split of a selected doc

    for key, value in all_docs_splits.items():
        print(key, ": NO of splits created", len(value))

        # make the same format previously udes in splits
    all_docs_splits_merged = {}
    for key, value in all_docs_splits.items():
        all_docs_splits_merged[key] = []
        for i in value:
            print(i.chunks)
            all_docs_splits_merged[key] += i.chunks
        
    all_docs_splits = all_docs_splits_merged

    all_docs_splits_f = {}
    for key, value in all_docs_splits.items():
      all_docs_splits_f[key] = []
      for each in value:
        all_docs_splits_f[key].append(Document(
            page_content=each,
            metadata={"source": "user_input", "length": len(each)}
        ))
    return all_docs_splits_f

# ----------------------------------------------------------------------------
# 1.2 Make Embeddings and Store within Vectorspaces
# ----------------------------------------------------------------------------
def vectorstores_creation(all_docs_splits, embedding_model) -> dict:
    """
    Input:
        all_docs_splits : langchain splits
    Output:
        vectorstores : dict  
            Format: {vectorstore 1: [Embed1, Embed2, ..., Embedn], vectorstore 2: [Embed1, Embed2, ..., Embedm]}
    """

    # Create vector embeddings for each document split
    print("---------Vectorstores Creation---------")
    vectorstores = {}
    var = int(time.time())
    for key, value in all_docs_splits.items():
        persist_dir = f"./db_file{key}{var}"

        # --- Remove old DB directory if it exists ---
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)  # deletes all old data
            print("Delete the exsisting DB")
        vectorstores[key] = Chroma.from_documents(documents=value, embedding=embedding_model,persist_directory=persist_dir)

    return vectorstores

# ----------------------------------------------------------------------------
# 1.3 Make the system prompt -> prompt tune is  needed after testing
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
# 1.4 Make Labels
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

def label_creation_with_memory(all_docs,deep_level = "high") -> dict:
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

        memory = memory['doc 0']["Confidential Information Use Restrictions"] # file no , label name ...
    """


    # Define the structured output schema
    class Classification(BaseModel):
        response: List[str] = Field(
            description="List **only** the LABELS, without any explanations or details. as a list"
        )

    structured_llm = llm.with_structured_output(Classification)

    sys_msg = build_system_message(deep_level)
    # Extract pages from all documents
    print("---------Pages Extraction---------")
    pages = {}
    for i, doc in enumerate(all_docs):
        pages[f"doc {i}"] = []
        for j in doc:
            pages[f"doc {i}"].append(j.page_content)
    #Document obj -> text extraction

    # Extract responses (headings) for each page using LLM
    print("---------Responses Extraction---------")
    responses = {}
    memory_inside = {}
    for i in range(len(pages)):
        responses[f"doc {i}"] = []
        memory_inside[f"doc {i}"] = {}
        count = 0
        page_no= 0
        total_pages = len(pages[f"doc {i}"])
        for page in pages[f"doc {i}"][:]:
            
            prompt = f"""
                You are given the following page content:
                {page}

                Your task:
                1. Carefully read and understand the content of this page.
                2. Ignore any irrelevant characters, extra newlines, or formatting artifacts not intended to be part of the document.
                3. Think critically about how this page’s content might differ if the same topic appeared in another, similar document.
                4. Based solely on this page, identify detailed and descriptive **LABELS** representing possible aspects or points of difference to check.

                Output:
                - Provide **only** a list of LABELS (no explanations, reasoning, or additional text).
                """


            # Wrap LLM with structured output
            
            system_message =  SystemMessage(content=sys_msg)
            human_message = HumanMessage(content=prompt)
            response = structured_llm.invoke([system_message, human_message]) #response = set of responses \n sepearate
            print(f"Loading {page_no+1}/{total_pages}.....")
            print(page)
            print('*'*100)
            print(response.response)
            for label in response.response:
              memory_inside[f"doc {i}"].update({label : page_no})
            responses[f"doc {i}"].extend(response.response) # responses[key] = [[],[]] list of lists
            page_no += 1

    # # Combine all responses into a single string per document
    # print("---------Sentences Extraction---------") 
    # senteces = {}
    # for key, value in responses.items():
    #     senteces[key] = ""
        
    #     for i in range(len(value)):
    #         senteces[key] += value[i]

    # # Split the combined responses into a list of sentences per document
    # sentences_list = {}
    # for key, value in senteces.items():
    #     sentences_list[key] = [line for line in value.split("\n") if line.strip()]

    sentences_list = responses

    return sentences_list,memory_inside

def NO_LLM_label_creation_with_memory(all_docs) -> dict:
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

        memory = memory['doc 0']["Confidential Information Use Restrictions"] # file no , label name ...
    """

    # Extract pages from all documents
    print("---------Pages Extraction---------")
    pages = {}
    for i, doc in enumerate(all_docs):
        pages[f"doc {i}"] = []
        for j in doc:
            pages[f"doc {i}"].append(j.page_content)
    #Document obj -> text extraction

    # Extract responses (headings) for each page using LLM
    print("---------Responses Extraction---------")
    responses = {}
    memory_inside = {}
    for i in range(len(pages)):
        responses[f"doc {i}"] = []
        memory_inside[f"doc {i}"] = {}
        count = 0
        page_no= 0
        total_pages = len(pages[f"doc {i}"])
        for page in pages[f"doc {i}"][:]:
            
            response = [page]
            print(f"Loading {page_no+1}/{total_pages}.....")
            print(page)
            print('*'*100)
            print(response)
            for label in response:
              memory_inside[f"doc {i}"].update({label : page_no})
            responses[f"doc {i}"].extend(response) # responses[key] = [[],[]] list of lists
            page_no += 1

    # # Combine all responses into a single string per document
    # print("---------Sentences Extraction---------") 
    # senteces = {}
    # for key, value in responses.items():
    #     senteces[key] = ""
        
    #     for i in range(len(value)):
    #         senteces[key] += value[i]

    # # Split the combined responses into a list of sentences per document
    # sentences_list = {}
    # for key, value in senteces.items():
    #     sentences_list[key] = [line for line in value.split("\n") if line.strip()]

    sentences_list = responses

    return sentences_list,memory_inside
# ----------------------------------------------------------------------------
# 1.4.1 Make Labels -> AGENTIC
# ----------------------------------------------------------------------------
def agentic_label_creation(all_docs_splits,deep_level = "high") -> dict:
    """
    STILL DEVELOPMENT ----------------->
    """

    semantic_labeling_prompt = """
    You are a semantic labeling agent. Your task is to generate 1–3 high-level, generalized labels for the given text that:
    - Capture the key entities or concepts and their roles/actions, without relying on exact names, dates, or numbers.
    - Focus on the semantics: what is happening, what roles exist, and what attributes or outcomes are important.
    - Preserve important relationships or statuses 
    - Ensure the labels are concise, consistent, and abstract enough to match similar content in other contexts even if names, numbers, or dates differ.
    - Do NOT omit critical conceptual information.
    - Output only the labels as a numbered list.
    """

    chunk_summary = """
    You are the steward of a group of chunks, where each chunk represents a document we walkthrough
    Generate a very brief, 1 to 2,3-sentence summary that explains what the chunk  is about.
    The summary should:
    - Capture the main topic or concept of the chunk .
    - Include any clarifying instructions for what to add or consider in the chunk.
    - Be generalizable, so similar content in other contexts can be recognized.
    Only respond with the new chunk summary, nothing else.
    """

    # Create a chat prompt template to compare two contexts
    tagging_prompt = ChatPromptTemplate.from_template("""
    You are given a chunk of  a whole pdf document. Your task is to:
    Logically think about the desired information from the following passage.
    Only think about the properties mentioned in the 'Classification' function.

    Passage:
    Context 1:
    {context1}

    """)

    # Define the structured output schema
    class Classification(BaseModel):
        chunks: List[str] = Field(
            description= semantic_labeling_prompt
        )
        summarization: str = Field(
            description=chunk_summary
        )

    # Wrap LLM with structured output
    structured_llm = llm.with_structured_output(Classification)

    #deep_level = "high"
    sys_msg = build_system_message(deep_level)

    # Extract responses (headings) for each page using LLM
    print("---------Responses Extraction---------")
    responses = {}
    for key,value in all_docs_splits.items():
        responses[key] = []
        for chunk in value[:]:
            prompt = f"""
            You need to understand the content of this page: {chunk}

            Think about what aspects could differ if the same topic appeared in another similar document.
            Based only on this page and your careful understanding, identify the **LABELS** that describe possible differences to check. Make the LABELS more detailed.
            List **only** the LABELS, without any explanations or details.
            """
            system_message =  SystemMessage(content=sys_msg)
            human_message = HumanMessage(content=prompt)
            prompt = tagging_prompt.invoke({"context1": chunk})
            response =  structured_llm.invoke(prompt)
            responses[key].append(response.chunks)

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

# ----------------------------------------------------------------------------
# 1.5 Extract the Mapping labels
# ----------------------------------------------------------------------------
def mapping_labels(labels,threshold=0.5,k=3):
    """
    Input:
        labels: dict  
            Format: {doc 1: [label1, label2, ..., labeln], doc 2: [label1, label2, ..., labelm]}
    Output:
        embeddings: dict  
            Format: {doc 1: [embed1, embed2, ..., embedn], doc 2: [embed1, embed2, ..., embedm]}  
            Note: embed1 -> [num1, num2, ..., numd] (d-dimensional vector)
        comparing_headings: list  
            Format: [[doc0_label1, doc1_label1], [doc0_label2, doc1_label2], ...]
    """

    # Create embeddings for each label
    print("---------Embedding Creation---------")
    embeddings = {}
    for key, value in labels.items():
        embeddings[key] = []
        for i in range(len(value)):
            embeddings[key].append(embedding_model.embed_query(value[i]))

    # Compute similarities between document embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = model.similarity(embeddings["doc 0"], embeddings["doc 1"])  # nxm similarity matrix

    # Get top k matches per row based on similarity
    comparing_headings = []
    t = threshold # similarity threshold
    values, indices = torch.topk(similarities, k=k, dim=1)  
    # values: highest similarity scores, indices: corresponding indices
    indices = indices.tolist()  # [[i00, i01, i02], [i10, i11, i12], ...]
    values = values.tolist()    # [[v00, v01, v02], [v10, v11, v12], ...]

    # NUM_OF_LABELS_SELECTED_FOR_EACH NOT ACTIVELY FUNCTION YET  
    for i in range(len(indices)):
        # Skip if the top similarity score is below threshold
        if values[i][0] < t:
            continue

        # Print matched labels
        print(labels["doc 0"][i])
        print(labels["doc 1"][indices[i][0]])
        comparing_headings.append([labels["doc 0"][i], labels["doc 1"][indices[i][0]]])
        print('-' * 50)

    print(f"---------Comparing Headings Creation {len(comparing_headings)}---------")
    return comparing_headings

# ----------------------------------------------------------------------------
# 1.6 Structured LLM to generate the response
# ----------------------------------------------------------------------------
def labelized_llm(context1, context2):
    """
    Input:
        context1 : str  
            Text from the first source.
        context2 : str  
            Text from the second source.
    Output:
        response : BaseModel  
            Contains structured output:
                - contradiction_result: str  
                    Indicates if there are any contradictions between the two contexts.
                - reason: str  
                    Explains the reason behind the contradiction (only if a contradiction exists).
    """

    # Create a chat prompt template to compare two contexts
    tagging_prompt = ChatPromptTemplate.from_template("""
    You are given two contexts from different sources.
    Logically think about the desired information from the following passage.
    Only think about the properties mentioned in the 'Classification' function.
    
    Passage:
    Context 1:
    {context1}

    Context 2:
    {context2}
    """)

    # Define the structured output schema
    class Classification(BaseModel):
        contradiction_result: str = Field(
            description="Are there any contradictions between the two contexts. Must be 'CONTRADICTION' or 'NO CONTRADICTION'."
        )
        reason: Optional[str] = Field(
            default=None,
            description="Only provide a reason if there is a contradiction."
        )

    # Wrap LLM with structured output
    structured_llm = llm.with_structured_output(Classification)

    # Invoke the prompt with the provided contexts
    prompt = tagging_prompt.invoke({"context1": context1, "context2": context2})
    response = structured_llm.invoke(prompt)

    print(response)
    return response

# Retrieve top-k relevant documents for a pair of labels
def get_topk_docs(pair, k=5,data=None,memory=None,type_is=None,retrievers=None):

   
    # pair = [label_doc1, label_doc2]
    # Restrict type_is to only two allowed values
    allowed_types = ["sementic_search", "label_bind", "both"]

     # Function to format document pages as a single string
    format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

    if type_is.lower() not in allowed_types:
        raise ValueError(f"type_is must be one of {allowed_types}, got '{type_is}'")

    if type_is == "sementic_search":
        docs1 = retrievers["doc 0"].get_relevant_documents(pair[0], k=k)
        docs2 = retrievers["doc 1"].get_relevant_documents(pair[1], k=k)
        docs1_f = format_docs(docs1)
        docs2_f = format_docs(docs2)
    elif type_is == "label_bind":
        # some other logic for type2
        docs1 = data[0][memory['doc 0'][pair[0]]]
        docs2 = data[1][memory['doc 1'][pair[1]]]
        docs1_f = docs1.page_content
        docs2_f = docs2.page_content
    elif type_is == "both":
        # some other logic for type2
        docs1_sementic = retrievers["doc 0"].get_relevant_documents(pair[0], k=k)
        docs2_sementic = retrievers["doc 1"].get_relevant_documents(pair[1], k=k)
        docs1_f_sementic = format_docs(docs1)
        docs2_f_sementic = format_docs(docs2)

        docs1_label_bind = data[0][memory['doc 0'][pair[0]]]
        docs2_label_bind = data[1][memory['doc 1'][pair[1]]]
        docs1_f_label_bind = docs1.page_content
        docs2_f_label_bind = docs2.page_content

        docs1_f = docs1_f_sementic + docs1_f_label_bind
        docs2_f = docs2_f_sementic + docs2_f_label_bind

    return docs1, docs2

def make_alldocs_as_document_object(extracted_file_contents, page_size=500):
    """
        extracted_file_contents = [[fileid,filename,filecontent]]
    """
    print("make_alldocs_as_document_object")
    all_docs = []
    for file_content in extracted_file_contents:
        print("file processing: ",file_content[:2]) # file name and id

    for file_content in extracted_file_contents:
        file_content = file_content[2] # just to extract the content
        doc_pages = [Document(
            page_content=file_content[i:i+page_size],
            metadata={"source": "user_input", "length": page_size}
        ) for i in range(0, len(file_content), page_size)]
        all_docs.append(doc_pages)
    return all_docs

def get_files_from_local_dir_pypdf(folder_path="/home/tharaka/thakshana/doc_comparison/data"):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            all_docs.append(docs)  # append all pages to the main list
    return all_docs

    
# ----------------------------------------------------------------------------
# 1.7 Main comparison function
# ----------------------------------------------------------------------------
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
   
   # labels = label_creation(all_docs)  # sentences_list: dict {doc 1:[sentence1,...], doc 2:[sentence1,...]}

    #labels,memory = label_creation_with_memory(all_docs,DEEP_LEVEL) 

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