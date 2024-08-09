"""
title: Custom RAG Pipeline
author: callum
description: A pipeline for retrieving relevant information from a chroma db using langchain.
requirements: aiohappyeyeballs==2.3.4, aiohttp==3.10.0, aiosignal==1.3.1, annotated-types==0.7.0, anyio==4.4.0, asgiref==3.8.1, attrs==23.2.0, backoff==2.2.1, bcrypt==4.2.0, beautifulsoup4==4.12.3, blinker==1.8.2, build==1.2.1, cachetools==5.4.0, certifi==2024.7.4, chardet==5.2.0, charset-normalizer==3.3.2, chroma-hnswlib==0.7.6, chromadb==0.5.5, click==8.1.7, coloredlogs==15.0.1, dataclasses-json==0.6.7, deepdiff==7.0.1, deprecated==1.2.14, dirtyjson==1.0.8, distro==1.9.0, dnspython==2.6.1, email-validator==2.2.0, emoji==2.12.1, fastapi==0.111.1, fastapi-cli==0.0.4, filelock==3.15.4, filetype==1.2.0, flask==3.0.3, flatbuffers==24.3.25, frozenlist==1.4.1, fsspec==2024.6.1, google-auth==2.32.0, googleapis-common-protos==1.63.2, greenlet==3.0.3, grpcio==1.65.2, h11==0.14.0, httpcore==1.0.5, httptools==0.6.1, httpx==0.27.0, huggingface-hub==0.24.5, humanfriendly==10.0, idna==3.7, importlib-metadata==8.0.0, importlib-resources==6.4.0, itsdangerous==2.2.0, jinja2==3.1.4, joblib==1.4.2, jsonpatch==1.33, jsonpath-python==1.0.6, jsonpointer==3.0.0, kubernetes==30.1.0, langchain==0.2.11, langchain-chroma==0.1.2, langchain-community==0.2.10, langchain-core==0.2.26, langchain-huggingface==0.0.3, langchain-text-splitters==0.2.2, langdetect==1.0.9, langsmith==0.1.95, llama-cloud==0.0.11, llama-index==0.10.59, llama-index-agent-openai==0.2.9, llama-index-cli==0.1.13, llama-index-core==0.10.59, llama-index-embeddings-openai==0.1.11, llama-index-indices-managed-llama-cloud==0.2.7, llama-index-legacy==0.9.48, llama-index-llms-openai==0.1.27, llama-index-multi-modal-llms-openai==0.1.8, llama-index-program-openai==0.1.7, llama-index-question-gen-openai==0.1.3, llama-index-readers-file==0.1.32, llama-index-readers-llama-parse==0.1.6, llama-parse==0.4.9, lxml==5.2.2, markdown==3.6, markdown-it-py==3.0.0, markupsafe==2.1.5, marshmallow==3.21.3, mdurl==0.1.2, mmh3==4.1.0, monotonic==1.6, mpmath==1.3.0, multidict==6.0.5, mypy-extensions==1.0.0, nest-asyncio==1.6.0, networkx==3.3, nltk==3.8.1, numpy==1.26.4, nvidia-cublas-cu12==12.1.3.1, nvidia-cuda-cupti-cu12==12.1.105, nvidia-cuda-nvrtc-cu12==12.1.105, nvidia-cuda-runtime-cu12==12.1.105, nvidia-cudnn-cu12==9.1.0.70, nvidia-cufft-cu12==11.0.2.54, nvidia-curand-cu12==10.3.2.106, nvidia-cusolver-cu12==11.4.5.107, nvidia-cusparse-cu12==12.1.0.106, nvidia-nccl-cu12==2.20.5, nvidia-nvjitlink-cu12==12.6.20, nvidia-nvtx-cu12==12.1.105, oauthlib==3.2.2, onnxruntime==1.18.1, openai==1.37.1, opentelemetry-api==1.26.0, opentelemetry-exporter-otlp-proto-common==1.26.0, opentelemetry-exporter-otlp-proto-grpc==1.26.0, opentelemetry-instrumentation==0.47b0, opentelemetry-instrumentation-asgi==0.47b0, opentelemetry-instrumentation-fastapi==0.47b0, opentelemetry-proto==1.26.0, opentelemetry-sdk==1.26.0, opentelemetry-semantic-conventions==0.47b0, opentelemetry-util-http==0.47b0, ordered-set==4.1.0, orjson==3.10.6, overrides==7.7.0, packaging==24.1, pandas==2.2.2, pillow==10.4.0, posthog==3.5.0, protobuf==4.25.4, psutil==6.0.0, pyasn1==0.6.0, pyasn1-modules==0.4.0, pydantic==2.8.2, pydantic-core==2.20.1, pygments==2.18.0, pypdf==4.3.1, pypika==0.48.9, pyproject-hooks==1.1.0, python-dateutil==2.9.0.post0, python-dotenv==1.0.1, python-iso639==2024.4.27, python-magic==0.4.27, python-multipart==0.0.9, pytz==2024.1, pyyaml==6.0.1, rapidfuzz==3.9.5, regex==2024.7.24, requests==2.32.3, requests-oauthlib==2.0.0, requests-toolbelt==1.0.0, rich==13.7.1, rsa==4.9, safetensors==0.4.3, scikit-learn==1.5.1, scipy==1.14.0, sentence-transformers==3.0.1, shellingham==1.5.4, six==1.16.0, sniffio==1.3.1, soupsieve==2.5, sqlalchemy[asyncio]==2.0.31, starlette==0.37.2, striprtf==0.0.26, sympy==1.13.1, tabulate==0.9.0, tenacity==8.5.0, threadpoolctl==3.5.0, tiktoken==0.7.0, timedatectl==0.8.0, titlecase==2.3.2, tokenizers==0.14.1, torch==2.1.0, tornado==6.3.3, tqdm==4.66.1, transformers==4.34.0, triton==2.1.1, typer==0.9.2, typing-extensions==4.8.1, typing-inspect==1.2.0, uc-messages==1.0.3, unstructured==0.7.14, unstructured-client==0.5.0, urllib3==1.27.0, uvicorn==0.23.1, uvloop==0.17.3, watchfiles==0.21.0, web-socket==1.2.7, websockets==11.0.3, werkzeug==3.0.0, wrapt==1.15.0, xformers==0.0.21, xmltodict==0.13.1, yarl==1.9.4, zstandard==0.21.0, zlib==1.3.0, zxcvbn==4.5.0
"""

from typing import List, Union, Generator, Iterator
import os
import asyncio

#models
from langchain_huggingface import HuggingFaceEmbeddings #embedding model
from sentence_transformers import CrossEncoder  #reranking model

#imports for vectordb
import chromadb
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

#imports for LLM querying
from langchain_community.chat_models import ChatOllama

class Pipeline:

    def __init__(self):
        self.db = None
        self.client = None
        self.embedding_function = None
        self.reranking_function = None
        self.model = None


    async def on_startup(self):

        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        RERANKING_MODEL = "BAAI/bge-reranker-large"

        # This function is called when the server is started.
        # global client, embedding_function, db, reranking_function, model
        #https://docs.trychroma.com/reference/py-client
        self.client = chromadb.HttpClient(port=8000)

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
            base_url="http://127.0.0.1:11434", 
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

