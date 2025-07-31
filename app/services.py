# import requests
# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
# from langchain_together import TogetherEmbeddings, ChatTogether
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from pinecone import Pinecone
# from pinecone import ServerlessSpec
# from .config import settings
# import time

# PDF_TEMP_PATH = "temp_policy.pdf"

# embedding_model = TogetherEmbeddings(
#     model="BAAI/bge-large-en-v1.5",
#     together_api_key=settings.together_api_key
# )

# llm = ChatTogether(
#     model="meta-llama/Llama-3-8b-chat-hf",
#     temperature=0.0,
#     max_tokens=1024,
#     together_api_key=settings.together_api_key
# )

# pc = Pinecone(api_key=settings.pinecone_api_key)

# def process_document_and_get_retriever(doc_url: str):
#     """
#     Downloads, chunks, embeds, and indexes a document with rate limiting.
#     """
#     try:
#         response = requests.get(doc_url)
#         response.raise_for_status()
#         with open(PDF_TEMP_PATH, "wb") as f:
#             f.write(response.content)

#         loader = PyPDFLoader(PDF_TEMP_PATH)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         chunks = text_splitter.split_documents(documents)

#         index_name = settings.pinecone_index_name
#         if index_name not in pc.list_indexes().names():
#             pc.create_index(
#                 name=index_name,
#                 dimension=1024,
#                 metric='cosine',
#                 spec=ServerlessSpec(
#                     cloud='aws',
#                     region='us-east-1'
#                 )
#             )

#         # --- NEW BATCHING LOGIC TO HANDLE RATE LIMITS ---
#         batch_size = 20 # Process 20 chunks at a time
#         index = pc.Index(index_name) # Get a handle to the index

#         print("Embedding and indexing in batches to respect rate limits...")
#         for i in range(0, len(chunks), batch_size):
#             # Get the batch of chunks
#             batch_chunks = chunks[i:i + batch_size]

#             # Create embeddings for the batch
#             batch_texts = [chunk.page_content for chunk in batch_chunks]
#             embeddings = embedding_model.embed_documents(batch_texts)

#             # Prepare vectors for upsert
#             vectors_to_upsert = []
#             for j, chunk in enumerate(batch_chunks):
#                 vector_id = f"vec_{i+j}"
#                 vectors_to_upsert.append({
#                     "id": vector_id,
#                     "values": embeddings[j],
#                     "metadata": {'text': chunk.page_content}
#                 })

#             # Upsert the batch to Pinecone
#             index.upsert(vectors=vectors_to_upsert)

#             print(f"Upserted batch {i//batch_size + 1}, waiting to avoid rate limit...")
#             time.sleep(20) # Wait for 20 seconds to reset the rate limit budget

#         print("Indexing complete.")
#         # Create the vector store object from the existing, populated index
#         vector_store = PineconeVectorStore.from_documents(
#             documents=chunks,
#             embedding=embedding_model,
#             index_name=index_name
#         )

#         return vector_store.as_retriever(search_kwargs={'k': 8})

#     finally:
#         if os.path.exists(PDF_TEMP_PATH):
#             os.remove(PDF_TEMP_PATH)

# async def generate_answer(retriever, question: str) -> str:
#     # A more forceful prompt for concise summarization
#     template = """SYSTEM: You are an expert insurance policy analyst. Your task is to provide a clear and concise answer based ONLY on the provided context. Do not use any of your outside knowledge. Summarize the findings into a single, clean paragraph. Do not return raw text chunks.

#     CONTEXT:
#     {context}

#     QUESTION:
#     {question}

#     ANSWER:"""
#     prompt = PromptTemplate.from_template(template)

#     rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     answer = await rag_chain.ainvoke(question)
#     return answer

import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone
from pinecone import ServerlessSpec
from .config import settings
import httpx
import asyncio
PDF_TEMP_PATH = "temp_policy.pdf"

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=settings.google_api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.0,
    google_api_key=settings.google_api_key
)

pc = Pinecone(api_key=settings.pinecone_api_key)

async def process_document_and_get_retriever(doc_url: str):
    """
    Asynchronously downloads, chunks, embeds, and indexes a document.
    """
    try:
        # Use httpx for async download
        async with httpx.AsyncClient() as client:
            response = await client.get(doc_url, follow_redirects=True)
            response.raise_for_status()
            with open(PDF_TEMP_PATH, "wb") as f:
                f.write(response.content)

        # The rest of the processing is CPU-bound, so it's okay to run it like this,
        # but we run it inside an async function.
        loader = PyPDFLoader(PDF_TEMP_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        index_name = settings.pinecone_index_name
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=index_name
        )

        return vector_store.as_retriever(search_kwargs={'k': 7})
    finally:
        if os.path.exists(PDF_TEMP_PATH):
            os.remove(PDF_TEMP_PATH)

async def generate_answer(retriever, question: str) -> str:
    template = """SYSTEM: You are an expert insurance policy analyst. Your task is to provide a clear and concise answer based ONLY on the provided context. Do not use any of your outside knowledge. Summarize the findings into a single, clean paragraph. Do not return raw text chunks.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:"""
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = await rag_chain.ainvoke(question)
    return answer