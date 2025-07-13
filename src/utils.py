from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import asyncio
import logging

# from src.azure_services import OpenAIService
from src.config import AZURE_OPENAI_MODEL_NAME, AZURE_OPENAI_API_VERSION, CHUNK_OVERLAP, CHUNK_SIZE
from src.oifile import OIFile
from src.prompts import map_prompt, reduce_prompt
from src.states import OverallState


logger = None
llm = None
map_chain = None
reduce_chain = None

# Setup the OpenAIService LLM
# openaiservice = OpenAIService()
# async def get_lg_llm():
#     """Get the LLM for the langgraph agent."""
#     global llm

#     if not llm:
#         llm = AzureChatOpenAI(
#             azure_endpoint=str(openaiservice.client._azure_endpoint),
#             azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
#             openai_api_version=openaiservice.client._api_version,
#             openai_api_key=openaiservice.client.api_key,
#             temperature=0,
#         )
#     return llm

# Setup the AzureChatOpenAI LLM
def get_llm():
    """Get the LLM for the langgraph agent."""
    global llm

    if not llm:
        llm = AzureChatOpenAI(
            model=AZURE_OPENAI_MODEL_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

    return llm

def get_map_chain():
    """Get the map chain for the langgraph agent."""
    global map_chain

    if not map_chain:
        llm = get_llm()
        map_chain = map_prompt | llm | StrOutputParser()

    return map_chain

def get_reduce_chain():
    """Get the reduce chain for the langgraph agent."""
    global reduce_chain

    if not reduce_chain:
        llm = get_llm()
        reduce_chain = reduce_prompt | llm | StrOutputParser()

    return reduce_chain

def get_logger(name: str="summarizer-map-reduce") -> logging.Logger:
    """
    Get a logger with the specified name. If no handlers are set, it will create a default StreamHandler.

    Args:
        name (str): The name of the logger. Defaults to "summarizer-map-reduce".

    Returns:
        logging.Logger: The configured logger instance.
    """
    global logger

    if not logger:
        # Validate the logger name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Logger name must be a non-empty string.")

        # Get or create a logger with the specified name
        logger = logging.getLogger(name)

        # Ensure the logger is not already configured
        if not logger.hasHandlers():
            # If the logger does not have handlers, we will set it up
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG)

    return logger

async def chunk_document(
    document: OIFile,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple:
    """Asynchronously chunk documents in parallel with limited concurrency"""
    logger = get_logger()

    # Create text splitter in a thread to avoid blocking
    def create_splitter_and_split_text(text: str) -> tuple:
        return tuple(RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", "... ", " ", ""],
            is_separator_regex=False
        ).split_text(text))

    # Validate parameters
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        logger.error(f"Invalid chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
        raise ValueError(
            "Chunk size must be positive, overlap must be non-negative, and overlap must be less than chunk size."
        )

    if not document:
        logger.warning("No documents provided for chunking.")
        return ()

    name = document.get_name()
    text = document.get_content()

    if len(text) < chunk_size:
        logger.warning(f"Document '{name}' is shorter than chunk size. No chunking applied.")
        return tuple(text)

    # Initialize the return value
    split_docs = []

    logger.info(f"Chunking document '{name}' with length {len(text)} characters")

    # Process the text
    split_docs = await asyncio.to_thread(create_splitter_and_split_text, text)

    num_chunks = len(split_docs)
    if num_chunks == 0:
        logger.warning(f"No chunks created while splitting '{name}'")
    else:
        logger.info(f"Split '{name}' into {num_chunks} chunks")

    return tuple(split_docs)

# Add this helper function for token counting
def count_tokens_sync(documents: List[Document]) -> int:
    """Synchronous helper for token counting that will run in a thread."""
    llm = get_llm()
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# Replace the existing length_function with this async version
async def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents asynchronously."""
    return await asyncio.to_thread(count_tokens_sync, documents)

# Add this async version of split_list_of_docs
async def split_list_of_docs_async(
    docs: List[Document],
    length_func,
    token_max: int
) -> List[List[Document]]:
    """Async version of split_list_of_docs that works with async length functions."""
    if not docs:
        return []

    docs_queue = list(docs)
    result = []
    _current = []
    _current_tokens = 0

    while docs_queue:
        doc = docs_queue.pop(0)
        # Get tokens for this document asynchronously
        doc_tokens = await length_func([doc])

        # If adding this doc would exceed token limit,
        # save the current group and start a new one
        if _current and _current_tokens + doc_tokens > token_max:
            result.append(_current)
            _current = [doc]
            _current_tokens = doc_tokens
        else:
            _current.append(doc)
            _current_tokens += doc_tokens

    if _current:
        result.append(_current)

    return result

def log_state_detailed(state: OverallState):
    """
    Log detailed information about the OverallState instance with
    specialized handling for each field.

    Args:
        state (OverallState): The state object to log
    """
    logger = get_logger()

    if not state:
        logger.debug("OverallState is empty or None")
        return

    logger.debug("=== STATE CONTENTS ===")

    # Handle documents (List[OIFile])
    if 'documents' in state:
        docs = state.get('documents', [])
        logger.debug(f"  documents: {len(docs)} document(s)")
        for i, doc in enumerate(docs[:3]):
            try:
                # Access common OIFile properties safely
                name = getattr(doc, 'get_name', lambda: str(doc))()
                doc_id = getattr(doc, 'get_id', lambda: 'unknown')()
                content_preview = getattr(doc, 'get_content', lambda: '')()[:50] if hasattr(doc, 'get_content') else '?'
                logger.debug(f"    [{i}] {name} (id: {doc_id}) - Content: '{content_preview}...'")
            except Exception as e:
                logger.debug(f"    [{i}] Error accessing document: {e}")
        if len(docs) > 3:
            logger.debug(f"    ... and {len(docs)-3} more document(s)")

    # Handle chunked documents (Dict[str, Tuple[str]])
    if 'chunked_documents' in state:
        chunk_dict = state.get('chunked_documents', {})
        logger.debug(f"  chunked_documents: {len(chunk_dict)} document ID(s)")
        for i, (doc_id, chunks) in enumerate(list(chunk_dict.items())[:2]):
            if isinstance(chunks, tuple):
                logger.debug(f"    [{i}] ID {doc_id}: {len(chunks)} chunks")
                if chunks and len(chunks) > 0:
                    first_chunk = chunks[0][:50] + "..." if len(chunks[0]) > 50 else chunks[0]
                    logger.debug(f"      First chunk: '{first_chunk}'")
        if len(chunk_dict) > 2:
            logger.debug(f"    ... and {len(chunk_dict)-2} more document(s)")

    # Handle file_ids (List[str])
    if 'file_ids' in state:
        file_ids = state.get('file_ids', [])
        logger.debug(f"  file_ids: {len(file_ids)} ID(s)")
        if file_ids:
            preview = file_ids[:3]
            logger.debug(f"    Preview: {preview}" + ("..." if len(file_ids) > 3 else ""))

    # Handle partial_summaries (List[str])
    if 'partial_summaries' in state:
        summaries = state.get('partial_summaries', [])
        logger.debug(f"  partial_summaries: {len(summaries)} summary(ies)")
        for i, summary in enumerate(summaries[:2]):
            preview = summary[:50] + "..." if len(summary) > 50 else summary
            logger.debug(f"    [{i}] {preview}")
        if len(summaries) > 2:
            logger.debug(f"    ... and {len(summaries)-2} more summary(ies)")

    # Handle partial_summaries_by_id (Dict[str, List[Document]])
    if 'partial_summaries_by_id' in state:
        summaries_by_id = state.get('partial_summaries_by_id', {})
        logger.debug(f"  partial_summaries_by_id: {len(summaries_by_id)} document ID(s)")
        for i, (doc_id, doc_summaries) in enumerate(list(summaries_by_id.items())[:2]):
            logger.debug(f"    [{i}] ID {doc_id}: {len(doc_summaries)} summary(ies)")
            if doc_summaries and len(doc_summaries) > 0:
                first_summary = doc_summaries[0]
                if hasattr(first_summary, 'page_content'):
                    content_preview = first_summary.page_content[:50] + "..." if len(first_summary.page_content) > 50 else first_summary.page_content
                    logger.debug(f"      First summary: '{content_preview}'")
        if len(summaries_by_id) > 2:
            logger.debug(f"    ... and {len(summaries_by_id)-2} more document ID(s)")

    # Check for any unexpected fields
    standard_fields = {
        'documents', 'chunked_documents', 'file_ids',
        'partial_summaries', 'partial_summaries_by_id'
    }

    extra_fields = [f for f in state.keys() if f not in standard_fields]
    if extra_fields:
        logger.debug("  Other fields:")
        for field in extra_fields:
            value = state[field]
            type_name = type(value).__name__
            preview = str(value)[:50] + "..." if isinstance(value, str) and len(value) > 50 else value
            logger.debug(f"    {field} ({type_name}): {preview}")

    logger.debug("=====================")