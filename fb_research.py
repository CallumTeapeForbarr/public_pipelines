"""
title: no pipeline?
author: callum
description: no pipeline?
requirements: chromadb, sentence_transformers,requests,numpy
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio
import requests

class Pipeline:

    def __init__(self):
        self.research_collection = None
        self.data_collection = None
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
        # self.research_collection = self.client.get_or_create_collection(name="research")
        self.data_collection = self.client.get_or_create_collection(name="data")

        print(self.data_collection.count())


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
        
        company, query = user_message.split(';')

        # embedded_query=self.embedding_function.encode([query])

        # docs = self.research_collection.query(
        #     query_embeddings=embedded_query,
        #     include=["documents","distances","metadatas"],
        #     where = {'company': company},
        #     n_results=15)
        
        data = self.data_collection.query(
            query_embeddings=[0],
            include=["documents"],
            where = {'company': company},
            n_results=1
        )

        return data["documents"][0]

