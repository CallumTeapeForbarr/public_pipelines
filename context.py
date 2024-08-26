"""
title: Custom RAG Pipeline
author: callum
description: A pipeline for retrieving relevant information from a chroma db using langchain.
requirements: chromadb, sentence_transformers,requests,numpy
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio
import requests
import numpy as np

class Pipeline:

    def __init__(self):
        self.collection = None
        self.client = None
        self.embedding_function = None
        self.reranking_function = None
        self.model = None


    async def on_startup(self):

        #models
        from sentence_transformers import CrossEncoder  #reranking model
        from sentence_transformers import SentenceTransformer

        #imports for vectordb
        import chromadb

        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        RERANKING_MODEL = "BAAI/bge-reranker-large"

        # This function is called when the server is started.
        # global client, embedding_function, db, reranking_function, model
        #https://docs.trychroma.com/reference/py-client
        self.client = chromadb.HttpClient(host="chroma",port="8000", ssl=False)
        self.collection = self.client.get_or_create_collection(name="research")


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


    def get_company(user_message):
        company = user_message.split(";")[0]
        return company



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
        
        company_list = np.array(['OCA', 'IKE', 'IPL', 'TWR', 'MHM', 'FRW', 'SML', 'CVT', 'KPG', 'NPH', 'AIR', 'AFT', 'HLG', 'ARV', 'VSL', 'RAD', 'WIN', 'SKO', 'SCL', 'PEB', 'ATM', 'VHP', 'NZM', 'PCT', 'RAK', 'SCT', 'TRA'])

        company, query = user_message.split(';')

        embedded_query=self.embedding_function.encode([query])

        docs = self.collection.query(
            query_embeddings=embedded_query,
            include=["documents","distances","metadatas"],
            where = {'company': company},
            n_results=15)

        reranked = self.reranking_function.rank(
            user_message,
            #[doc.page_content for doc in docs],
            docs["documents"][0],
            top_k=5,
            return_documents=True
        )

        context = ""
        for doc in reranked:
            context += doc["text"].replace('\n', ' ')
            context += '\n'

        return context


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


