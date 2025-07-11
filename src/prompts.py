from langchain.prompts import ChatPromptTemplate

system_prompt = """
You are a document and literature analysis assistant specialized in identifying important
information in text documents. Your response should be consise and focused on the most
relevant information, and should not include any personal opinions or interpretations.
Your response should be written in a neutral tone, without any bias or subjective language.
"""

map_template = """
### Instruction:
Your task is to analyze the following chunk of text from a document and generate a summarization of it,
which contains the most important information contained in it. This summarization will be a part of a
larger summarization process, so it should be concise and focused on the key points of the specific chunk.

### Input:
{context}

### Response:
Please provide a concise summary of the input above, focusing on the most relevant information.
Your summary should be clear and easy to understand, highlighting key points and important details.
Your summary should have the form of a single paragraph, with no more than 20 words.
"""

reduce_template = """
### Instruction:
The following is a set of partial summaries generated from chunks of text from the same document,
and contain the most important information of said chunks. Your task is to take these summaries,
abalyze them and distill them into a final, consolidated summary of the main themes of the document.

### Input:
{docs}

### Response:
Please provide a concise summary of the input above, focusing on the most relevant information.
Your summary should be clear and easy to understand, highlighting key points and important details.
Your summary should have the form of a single paragraph, with no more than 250 words.
"""


map_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", map_template)
    ]
)

reduce_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", reduce_template)
    ]
)