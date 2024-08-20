"""
title: Custom RAG Pipeline
author: callum
description: A pipeline for retrieving relevant information from a chroma db using langchain.
requirements: chromadb, sentence_transformers,requests
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio
import requests

class Pipeline:

    def __init__(self):
        #self.db = None
        self.collection = None
        self.client = None
        self.embedding_function = None
        self.reranking_function = None
        self.model = None


    async def on_startup(self):

        #models
        #from langchain_huggingface import HuggingFaceEmbeddings #embedding model
        from sentence_transformers import CrossEncoder  #reranking model
        from sentence_transformers import SentenceTransformer

        #imports for vectordb
        import chromadb
        #from langchain_community.vectorstores import Chroma
        # from langchain_chroma import Chroma

        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        RERANKING_MODEL = "BAAI/bge-reranker-large"

        # This function is called when the server is started.
        # global client, embedding_function, db, reranking_function, model
        #https://docs.trychroma.com/reference/py-client
        self.client = chromadb.HttpClient(host="chroma",port="8000", ssl=False)
        self.collection = self.client.get_or_create_collection(name="research")

        #https://api.python.langchain.com/en/latest/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
        #self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        #https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
        # self.db = Chroma(
        #     client=self.client,
        #     collection_name="research",
        #     embedding_function=self.embedding_function,
        # )

        self.embedding_function = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.reranking_function = CrossEncoder(
            RERANKING_MODEL,
            trust_remote_code=True
        )
        pass


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
        ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        prompt = """
                You are an expert consultant helping financial advisors to get relevant information from market research reports.

                Use the context given below to answer the advisors questions.
        
                The context will be a series of excerpts from market research reports. They may not belong to the same report. 
                Please prioritise the most recent. Please prioritise actual data over forecast where applicable.
                
                Constraints:
                1. Only use the context given to answer.
                2. Do not make any statements which aren't verifiable from this context. 
                2. Try to answer in one or two concise paragraphs
            """

        embedded_query = self.embedding_function.encode([user_message])

        # docs = self.db.similarity_search(user_message,k=30)
        docs = self.collection.query(query_embeddings=embedded_query,include=["documents","distances","metadatas"],n_results=20)

        reranked = self.reranking_function.rank(
            user_message,
            #[doc.page_content for doc in docs],
            docs["documents"][0],
            top_k=5,
            return_documents=True
        )

        context ="\n\n NEW CONTEXT".join(doc["text"] for doc in reranked)

        return context.replace('\n','')


        payload = {
            "model": "qwen2:1.5b",
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": f"CONTEXT: {context}\nQUERY: {user_message}"},
            ],
            "stream": body["stream"],
        }

        #https://github.com/ollama/ollama/blob/main/docs/api.md
        try:
            r = requests.post(
                url=f"http://ollama:11434/v1/chat/completions",
                json=payload,
                stream=True
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"


