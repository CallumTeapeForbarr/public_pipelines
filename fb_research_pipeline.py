"""
title: Custom RAG Pipeline
author: callum
description: A pipeline for retrieving relevant information from a chroma db using langchain.
requirements: langchain_text_splitters,langchain_community,langchain_huggingface,chromadb,langchain_chroma,flask,transformers,pypdf,tiktoken
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio


class Pipeline:

    def __init__(self):
        self.db = None
        self.client = None
        self.embedding_function = None
        self.reranking_function = None
        self.model = None


    async def on_startup(self):

        #models
        from langchain_huggingface import HuggingFaceEmbeddings #embedding model
        from sentence_transformers import CrossEncoder  #reranking model

        #imports for vectordb
        import chromadb
        #from langchain_community.vectorstores import Chroma
        from langchain_chroma import Chroma

        #imports for LLM querying
        from langchain_community.chat_models import ChatOllama

        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        RERANKING_MODEL = "BAAI/bge-reranker-large"

        # This function is called when the server is started.
        # global client, embedding_function, db, reranking_function, model
        #https://docs.trychroma.com/reference/py-client
        self.client = chromadb.HttpClient(host="chroma",port="8000", ssl=False)

        #https://api.python.langchain.com/en/latest/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        #https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
        self.db = Chroma(
            client=self.client,
            collection_name="research",
            embedding_function=self.embedding_function,
        )

        self.reranking_function = CrossEncoder(
            RERANKING_MODEL,
            trust_remote_code=True
        )

        self.model = ChatOllama(
            base_url="ollama", 
            model="gemma2:2b",
            num_ctx=4096,
            num_gpus=0,
            verbose=True,
            keep_alive=-1   #keep the model loaded in memory
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

        docs = self.db.similarity_search(user_message,k=30)



        reranked = self.reranking_function.rank(
            user_message,
            [doc.page_content for doc in docs],
            top_k=5,
            return_documents=True
        )

        context =''.join(doc["text"]+"\n" for doc in reranked)

        return context

        messages =  [
                ("system", f"""
                        You are an expert consultant helping financial advisors to get relevant information from market research reports.

                        Use the context given below to answer the advisors questions.
                
                        The context will be a series of excerpts from market research reports. They may not belong to the same report. 
                        Please prioritise the most recent. Please prioritise actual data over forecast where applicable.
                        
                        Constraints:
                        1. Only use the context given to answer.
                        2. Do not make any statements which aren't verifiable from this context. 
                        2. Try to answer in one or two concise paragraphs
                        """
                ),
                ("user", f"CONTEXT: {context}\nQUERY: {user_message}")
            ]
        
        return self.model.invoke(messages).content

