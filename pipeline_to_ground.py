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
                You are a helpful assistant
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
                    "content": user_message
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
                            break
                    else:
                        return r.json()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"


