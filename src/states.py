from langchain_core.documents import Document
from typing import Annotated, Any, Dict, List, Tuple, TypedDict
import operator

from src.oifile import OIFile


class InputState(TypedDict):
    """State for the input node that contains the files to be processed."""
    files: List[Dict[str, str]]

class OverallState(TypedDict):
    """State for the overall process, including all documents and their summaries."""
    documents: Annotated[List[OIFile], operator.add]
    document_chunks: Annotated[Dict[str, Tuple[str]], operator.or_]
    document_ids: Annotated[List[str], operator.add]
    partial_summaries: Annotated[List[str], operator.add]
    document_partial_summaries: Annotated[Dict[str, List[Document]], operator.or_]

class OutputState(TypedDict):
    """State for the output node that contains the final documents, including their summaries."""
    result: Annotated[Dict[str, str], operator.or_]


class LoadState(TypedDict):
    """State for the load node that contains a file to be loaded as an IOFile object."""
    file: Dict[str, Any]

class SplitState(TypedDict):
    """State for the split node that contains an IOFile object whose content will be split into chunks."""
    document: OIFile

class MapSummaryState(TypedDict):
    """State for the map node that contains a document ID and a chunk of text content to be summarized."""
    document_id: str
    content: str

class CollapseState(TypedDict):
    """State for the collapse node that contains a document ID and a list of partial summaries to be collapsed into a final summary."""
    document_id: str
    summaries: List[Document]

class ReduceSummaryState(TypedDict):
    """State for the reduce node that contains a document as an OIFile object and a list with its content's partial summaries."""
    document: OIFile
    summaries: List[Document]