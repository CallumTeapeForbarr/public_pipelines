"""
title: fb_research  
author: callum
description: fb research pipeline
requirements: chromadb, sentence_transformers,requests,numpy
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio
import requests
import json
import copy
import math
from datetime import datetime

class Pipeline:

    def __init__(self):
        self.research_collection = None
        self.data_collection = None
        self.time_collection = None
        self.client = None
        self.embedding_function = None
        self.reranking_function = None
        self.model = None
        self.conversation = None


    async def on_startup(self):

        import numpy as np

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
        self.research_collection = self.client.get_or_create_collection(name="research")
        self.data_collection = self.client.get_or_create_collection(name="data")
        self.time_collection = self.client.get_or_create_collection(name='time')

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


        date = '2024-09-01'
        company = None
        query = None



        if ';' not in user_message:
            query = user_message

        else:
            date, company, query = user_message.split(';')

        print(date)

        date = datetime.strptime(date.split('\n')[-1], '%Y-%m-%d')
        the_big_bang = datetime(2020,1,1)
        r = 1
        z = (date - the_big_bang).days
        compression = math.sqrt((16/3))
        revs = z/365
        vec = [r*math.cos(2*math.pi*revs), r*math.sin(2*math.pi*revs),  compression*revs]


        embedded_query=self.embedding_function.encode([query])

        if not company is None:
            text_rankings = self.research_collection.query(
                query_embeddings=embedded_query,
                include=["documents","distances","metadatas"],
                where = {'company': company},
                n_results=int(self.research_collection.count()/2)
            )

            time_rankings = self.time_collection.query(
                query_embeddings=[vec],
                include=["documents","distances","metadatas"],
                where = {'company': company},
                n_results=self.time_collection.count()
            )
        
        else:
            docs = self.research_collection.query(
                query_embeddings=embedded_query,
                include=["documents","distances","metadatas"],
                n_results=int(self.research_collection.count()/2)
            )

            time_rankings = self.time_collection.query(
                query_embeddings=[vec],
                include=["documents","distances","metadatas"],
                n_results=self.time_collection.count()
            )


        time_distances = np.array(time_rankings['distances'][0])
        time_distances_normalised = time_distances/np.max(time_distances)
        time_dict = dict(zip(time_rankings['ids'][0], time_distances_normalised))

        text_distances = np.array(text_rankings['distances'][0])
        text_distances_normalised = text_distances/np.max(text_distances)
        text_dict = dict(zip(text_rankings['ids'][0],text_distances_normalised))

        combined = {}
        for key in text_dict.keys():
              combined[key] = text_dict[key] + time_dict[key]

        ranking = dict(sorted(combined.items(), key=lambda item: item[1]))

        docs = [self.research_collection.get(ids=[key]) for key in ranking.keys()[:30]]
        
        reranked = self.reranking_function.rank(
            query,
            # docs["documents"][0],
            docs,
            top_k=10,
            return_documents=True
        )

        
        try:
            data = self.data_collection.query(
                query_embeddings=[[0]],
                include=["documents"],
                where = {'company': company},
                n_results=1
            )

            facts = data["documents"][0][0]
        except:
            facts = "no data found"

        context = ''
        sources = []

        for ranking in reranked:
            context += docs['metadatas'][0][ranking['corpus_id']]['date']
            context += '\n'
            context += ranking['text']
            context += '\n\n'

            sources.append(docs['metadatas'][0][ranking['corpus_id']]['source'].split('/')[-1].split('.')[0])


        deduped_sources = list(set(sources))
        source_text = '\n'.join(deduped_sources)



        prompt = """
                You are an expert consultant helping financial advisors to get relevant information from market research reports.

                Use the context given below to answer the advisors questions.

                The context will consist of a series of data in json format, and a series of excerpts from reports. 
                Use the data to find values, statistics and facts, use the excerpts to find explanations, descriptions and speculation.
                In cases where numbers reported in excerpts differ from numbers in the data section, assume that data section contains the true values.

                Constraints:
                1. Only use the context given to answer.
                2. Do not make any statements which aren't verifiable from this context. 
                3. Answer in one or two concise paragraphs
                4. Do not make any references to the context which was given. Write as though you are explaining without the documents at hand.
            """


        convo = copy.deepcopy(messages)

        convo = convo[:-1]

        convo.insert(0,
                     {
                         'role': 'system',
                         'content': prompt
                     })
        
        convo.append(
                {
                    "role": "user", 
                    "content": f"DATA: {facts}\EXCERPTS: {context}\nQUERY: {user_message}"
                }
        )


        payload = {
            "model": "qwen2:1.5b",
            "options": {
                "num_ctx": 4096
            },
            "messages": convo,
            "stream": body["stream"]
        }

        print(payload['messages'][1]['content'])


        api_url = 'http://ollama:11434/api/chat'

        #https://github.com/ollama/ollama/blob/main/docs/api.md
        try:
            r = requests.post(
                url=api_url,
                json=payload,
                stream=True
            )

            r.raise_for_status()

            if body["stream"]:
                for line in r.iter_lines():
                    if line:
                        # Convert the line to a dictionary
                        data = json.loads(line.decode('utf-8'))

                        # Extract the content from the message
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']

                        # Stop if the "done" flag is True
                        if data.get('done', False):
                            yield(f"\n\n\n\n{source_text}")
                            break
                    else:
                        return r.json()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"


