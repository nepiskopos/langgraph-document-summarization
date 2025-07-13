from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import acollapse_docs
from langgraph.types import Send
from typing import Literal

from src.config import TOKEN_MAX
from src.oifile import OIFile
from src.states import InputState, OverallState, OutputState, LoadState, SplitState, MapSummaryState, CollapseState, ReduceSummaryState
from src.utils import split_list_of_docs_async, chunk_document, get_logger, get_map_chain, get_reduce_chain, length_function


logger = get_logger()


async def _map_input(state: InputState) -> LoadState:
    """Map input files to load_document state."""
    sends = []

    # Send each file in the input state to the load_document state in parallel
    for file in state.get('files', []):
        sends.append(
            Send("load_document", {
                "file": file,
            })
        )

    return sends

async def _load_document(state: LoadState) -> OverallState:
    """Load a document from the provided file information dictionary."""
    results = []

    try:
        file = state.get('file', {})

        if file and isinstance(file, dict) and "file" in file:
            file_info = file["file"]
            logger.debug(f"Loading document with ID {file_info.get('id', '')}")

            if file_info.get('data', {}).get('content', ''):
                results.append(OIFile(
                    id=file_info['id'],
                    name=file_info['filename'],
                    type=file_info['meta']['content_type'],
                    content=file_info['data']['content']
                ))
                logger.debug(f"✓ Successfully loaded document: {results[0]}")
            else:
                logger.error(f"✕ ERROR: Missing data or content of document with ID {file_info.get('id', '')}")
        else:
            logger.warning(f"⚠ WARNING: Invalid document format received: {type(file)}")
    except Exception as e:
        logger.error(f"✕ ERROR: Could not load document: {str(e)}")

    return {'documents': results}

async def _map_documents(state: OverallState) -> SplitState:
    """Map loaded documents to split_document state."""
    sends = []

    for doc in state.get('documents', []):
        sends.append(
            Send("split_document", {
                "document": doc,
            })
        )

    return sends

async def _split_document(state: SplitState) -> OverallState:
    """Split a document into chunks."""
    results = {}

    file = state.get('document', None)

    if file:
        logger.debug(f"Splitting document: {file}")

        chunks = await chunk_document(file)

        if chunks:
            # Store chunks in dictionary with file ID as key
            results[file.get_id()] = chunks

            logger.debug(f"✓ Successfully split document {file.get_name()} into {len(chunks)} chunks")
        else:
            logger.warning(f"⚠ WARNING: No chunks generated for document {file.get_name()}")
    else:
        logger.error("✕ ERROR: No document provided to '_split_document'")

    return {'document_chunks': results}

async def _map_chunks(state: OverallState) -> MapSummaryState:
    """Map document chunks to generate_summary state."""
    sends = []

    for fid, chunks in state.get('document_chunks', {}).items():
        for chunk in chunks:
            sends.append(
                Send("generate_summary", {
                    "document_id": fid,
                    "content": chunk,
                })
            )

    return sends

async def _generate_summary(state: MapSummaryState) -> OverallState:
    """Generate a summary for a document chunk."""
    document_ids = []
    partial_summaries = []

    file_id = state.get("document_id", '')
    context = state.get("content", '')

    if file_id and context:
        map_chain = get_map_chain()
        response = await map_chain.ainvoke({'context': context})
        document_ids.append(file_id)
        partial_summaries.append(response)

        logger.debug(f"✓ Successfully generated summary for document with ID {file_id}")
    else:
        logger.error('✕ ERROR: No text content for generating summary on')

    # Return with document and chunk indexes to maintain structure
    return {
        "document_ids": document_ids,
        "partial_summaries": partial_summaries,
    }

async def _group_partial_summaries(state: OverallState) -> OverallState:
    """Group partial summaries by their document IDs."""
    results_by_doc = {}

    # Collect all items with their document indexes
    for fid, partial_summary in zip(state.get("document_ids", []), state.get("partial_summaries", [])):
        results_by_doc.setdefault(fid, []).append(Document(partial_summary))

    logger.debug(f"✓ Successfully grouped {len(results_by_doc)} partial summaries by document ID")

    return {"document_partial_summaries": results_by_doc}

async def _should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    """Decide whether to collapse summaries or generate final summary."""
    sends = []

    # Create mapping of documents by ID for easier lookup
    doc_map = {doc.get_id(): doc for doc in state.get("documents", [])}

    for fid, partial_summaries in state.get('document_partial_summaries', {}).items():
        if fid in doc_map:
            if not doc_map[fid].get_summary():
                # Use async version - add await here
                token_count = await length_function(partial_summaries)

                if token_count > TOKEN_MAX:
                    sends.append(
                        Send("collapse_summaries", {
                            "document_id": fid,
                            "summaries": partial_summaries,
                        })
                    )
                    logger.debug(f"→ Directed flow to 'collapse_summaries' for file with ID {fid}")
                else:
                    sends.append(
                        Send("generate_final_summary", {
                            "document": doc_map[fid],
                            "summaries": partial_summaries,
                        })
                    )
                    logger.debug(f"→ Directed flow to 'generate_final_summary' for file with ID {fid}")

    return sends

async def _collapse_summaries(state: CollapseState) -> OverallState:
    """Collapse summaries for a document."""
    results = {}

    file_id = state.get('document_id', '')
    summaries = state.get('summaries', [])

    if file_id and summaries:
        # Use our async version instead of the synchronous one
        doc_lists = await split_list_of_docs_async(
            summaries,
            length_function,
            TOKEN_MAX
        )

        if doc_lists:
            results[file_id] = []

            reduce_chain = get_reduce_chain()
            for doc_list in doc_lists:
                results[file_id].append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

            logger.debug(f"✓ Successfully collapsed summaries for document ID: {file_id}")

    return {"document_partial_summaries": results}

async def _generate_final_summary(state: ReduceSummaryState) -> OutputState:
    """Generate the final summary for a document."""
    results = {}

    doc = state.get("document", None)
    summaries = state.get("summaries", [])

    if doc and summaries:
        try:
            reduce_chain = get_reduce_chain()
            response = await reduce_chain.ainvoke({'docs': summaries})
            doc.set_summary(response)
            results[doc.get_id()] = doc.to_dict()

            logger.debug(f"✓ Successfully generated final summary for {doc.get_name()}")
        except Exception as e:
            logger.error(f"✕ ERROR: Exception while generating final summary for {doc.get_name()}: {str(e)}")
    else:
        if not doc:
            logger.error("✕ ERROR: No document provided to _generate_final_summary")

        if not summaries:
            logger.warning(f"⚠ WARNING: No summaries provided for {doc.get_name()}, using placeholder")

    return {"result": results}