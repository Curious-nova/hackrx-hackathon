# # app/main.py

# # No longer need asyncio
# from fastapi import FastAPI, Depends, HTTPException, status, Header
# from typing import Annotated, Dict
# from langchain.vectorstores.base import VectorStoreRetriever

# from .models import HackathonRequest, HackathonResponse
# from .services import process_document_and_get_retriever, generate_answer
# from .config import settings

# app = FastAPI(
#     title="Intelligent Query-Retrieval System API",
#     description="API for answering questions about documents using a RAG pipeline."
# )

# retriever_cache: Dict[str, VectorStoreRetriever] = {}

# # This can stay async, it's fine
# async def verify_token(authorization: Annotated[str, Header()]):
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication scheme.")
#     token = authorization.split(" ")[1]
#     if token != settings.hackathon_api_token:
#         raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid token.")


# # --- CHANGE THE ENDPOINT BACK TO A NORMAL 'def' ---
# @app.post("/api/v1/hackrx/run", response_model=HackathonResponse)
# def run_query_retrieval(
#     request: HackathonRequest,
#     _token: None = Depends(verify_token)
# ):
#     doc_url = request.documents
#     try:
#         if doc_url in retriever_cache:
#             retriever = retriever_cache[doc_url]
#         else:
#             retriever = process_document_and_get_retriever(doc_url)
#             if retriever is None:
#                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to process document.")
#             retriever_cache[doc_url] = retriever

#         # ** USE A SIMPLE FOR LOOP INSTEAD OF ASYNCIO **
#         all_answers = []
#         for question in request.questions:
#             answer = generate_answer(retriever, question)
#             all_answers.append(answer)

#         return HackathonResponse(answers=all_answers)

#     except Exception as e:
#         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")


# @app.get("/")
# def health_check():
#     return {"status": "ok", "cache_size": len(retriever_cache)}

import asyncio
from fastapi import FastAPI, Depends, HTTPException, status, Header
from typing import Annotated, Dict
from langchain.vectorstores.base import VectorStoreRetriever
from .models import HackathonRequest, HackathonResponse
from .services import process_document_and_get_retriever, generate_answer
from .config import settings

app = FastAPI(
    title="Intelligent Query-Retrieval System API",
    description="API for answering questions about documents using a RAG pipeline."
)

retriever_cache: Dict[str, VectorStoreRetriever] = {}

async def verify_token(authorization: Annotated[str, Header()]):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication scheme.")
    token = authorization.split(" ")[1]
    if token != settings.hackathon_api_token:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid token.")

@app.post("/api/v1/hackrx/run", response_model=HackathonResponse)
async def run_query_retrieval(
    request: HackathonRequest,
    _token: None = Depends(verify_token)
):
    doc_url = request.documents
    try:
        if doc_url in retriever_cache:
            retriever = retriever_cache[doc_url]
        else:
            # We must now 'await' this function call
            retriever = await process_document_and_get_retriever(doc_url)
            if retriever is None:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to process document.")
            retriever_cache[doc_url] = retriever

        tasks = [generate_answer(retriever, q) for q in request.questions]
        all_answers = await asyncio.gather(*tasks)

        return HackathonResponse(answers=all_answers)

    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "ok", "cache_size": len(retriever_cache)}