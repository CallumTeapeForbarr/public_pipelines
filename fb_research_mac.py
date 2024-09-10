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
import numpy as np

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

        #models
        from sentence_transformers import CrossEncoder  #reranking model
        from sentence_transformers import SentenceTransformer

        #imports for vectordb
        import chromadb

        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        # EMBEDDING_MODEL = "dunzhang/stella_en_400M_v5"
        RERANKING_MODEL = "BAAI/bge-reranker-large"

        # This function is called when the server is started.
        # global client, embedding_function, db, reranking_function, model
        #https://docs.trychroma.com/reference/py-client
        self.client = chromadb.HttpClient(host="chroma",port="8000", ssl=False)
        self.research_collection = self.client.get_or_create_collection(name="research")
        self.data_collection = self.client.get_or_create_collection(name="data")
        self.time_collection = self.client.get_or_create_collection(name='time')

        self.embedding_function = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True
        )

        # self.reranking_function = CrossEncoder(
        #     RERANKING_MODEL,
        #     trust_remote_code=True
        # )

        pass


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
        ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        date_string = '2024-09-01'
        company = None
        query = None



        if ';' not in user_message:
            query = user_message

        else:
            date_string, company, query = user_message.split(';')


        print(date_string)

        """
        Method to give an embedding for dates into a helical space.

        I choose a helix because of the following characteristics:
            1.  Continuous: reflects continuity of time.
            2.  Cyclical: the shape repeats every revolution, meaning relationships within each revolution (year) are comporable
            3.  Spacing: Can control how tightly wound the helix is to make points between years more or less similar

        """
        #parsing time input and choosing starting date as the first of 2020
        date = datetime.strptime(date_string.strip(), '%Y-%m-%d')
        the_big_bang = datetime(2020,1,1)

        #choosing constants (r being radius, compression being the amount the curve travels in the z axis per revolution/how tightly wound it is)
        r = 1
        compression = math.sqrt((16/3))

        #figuring out the number of revolutions (as a function of years since the starting date)
        z = (date - the_big_bang).days
        revs = z/365


        # Computing the 3D vector:
        # - x and y are on the circle, determined by the angle (2 * π * revs)
        # - z is determined by the number of revolutions multiplied by the compression factor
        vec = [[r*math.cos(2*math.pi*revs), r*math.sin(2*math.pi*revs),  compression*revs]]


        embedded_query=self.embedding_function.encode([query])

            #searching the space of sentence transformer embedded text
        # research_collection = self.client.get_or_create_collection(name="research")
        text_rankings = self.research_collection.query(
            query_embeddings=embedded_query,
            include=["documents","distances","metadatas"],
            where = {'company': company},
            n_results=int(self.research_collection.count()/5))   #trying to keep this relatively small while retaining enough as to not throw away the time relevant points
        
        #searching the space of helix embedded dates
        # time_collection = client.get_or_create_collection(name='time')
        time_rankings = self.time_collection.query(
            query_embeddings=vec,
            include=["documents","distances","metadatas"],
            where = {'company': company},
            n_results=self.time_collection.count())  #everything (assuming this search won't take too long as it's a 3d space)


        #creating normalised dictionary of time collection rankings
        time_distances = np.array(time_rankings['distances'][0])
        time_distances_normalised = time_distances/np.max(time_distances)
        time_dict = dict(zip(time_rankings['ids'][0], time_distances_normalised))

        #creating normalised dictionary of text collection rankings
        text_distances = np.array(text_rankings['distances'][0])
        text_distances_normalised = text_distances/np.max(text_distances)
        text_dict = dict(zip(text_rankings['ids'][0],text_distances_normalised))


        #combining the two rankings together
        combined = {}
        #iterate over text_dict keys, note that time_dict has every id in the database, so this should never throw a key error
        for key in text_dict.keys():
                combined[key] = text_dict[key] + time_dict[key]   #equal parts each


        #reordering the combined dcitionary to sort by the lowest distance
        ranking = dict(sorted(combined.items(), key=lambda item: item[1]))   
        # print(ranking)

        sources = []
        # docs = []
        # dates = []

        context = ''

        #now taking the top ten of the combined ranking (source, publication date, and text itself)
        for id in list(ranking.keys())[:10]:
            document = self.research_collection.get(ids=[id])

            source_str = document['metadatas'][0]['source']
            date_str = document['metadatas'][0]['date']
            document_str = document['documents'][0]

            context += f'{date_str}\n{document_str}\n'

            sources.append(source_str)
            # dates.append(date_str)
            # docs.append(document_str)

        deduped_sources = list(set(sources))
        source_text = '\n'.join(deduped_sources)

        
        # reranked = self.reranking_function.rank(
        #     query,
        #     docs["documents"][0],
        #     top_k=5,
        #     return_documents=True
        # )

        # context = ''
        # sources = []

        # for ranking in reranked:
        #     context += docs['metadatas'][0][ranking['corpus_id']]['date']
        #     context += '\n'
        #     context += ranking['text']
        #     context += '\n\n'

        #     sources.append(docs['metadatas'][0][ranking['corpus_id']]['source'].split('/')[-1].split('.')[0])


        # deduped_sources = list(set(sources))
        # source_text = '\n'.join(deduped_sources)
        
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



        prompt = f"""
                You are an expert consultant helping financial advisors to get relevant information from market research reports.

                Use the context given below to answer the advisors questions.

                The context will consist of a series of data in json format, and a series of excerpts from reports. 
                Use the data to find values, statistics and facts, use the excerpts to find explanations, descriptions and speculation.

                The user has provided a date: {date_string}. The excerpts provided will have been published close to this date. 
                The exact publication date will be provided above each excerpt. Please prioritise excerpts which were published near the provided date {date_string}.
                When using data from this period, please clearly indicate that the data is historic, and cite the period it pertains to.
        
                The data in the data section will always be the most current. Always use this data if the user gives no indication that they want historic data.

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
                    "content": f"DATA: {facts}\EXCERPTS: {context}\nQUERY: {query}"
                }
        )


        payload = {
            "model": "llama3.1",
            "options": {
                "num_ctx": 4096
            },
            "messages": convo,
            "stream": body["stream"]
        }

        print(payload)


        api_url = 'http://host.docker.internal:11434/api/chat'

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
                            yield(f"\n{source_text}")
                            break
                    else:
                        return r.json()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"


