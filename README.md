# Chatbot: RAG with Speech

This project provides a **Retrieval Augmented Generation (RAG)** API that acts as an intelligent chatbot, named *Meera*, capable of answering user queries based on your custom data. It combines **LlamaIndex** for efficient retrieval, **ChromaDB** for vector storage, and **Edge TTS** for generating lifelike speech responses.

---

## Features

* **RAG Chatbot (Meera)**: Answers user queries using vector-based retrieval and large language models (LLMs).
* **Vector Store Indexing**: Uses **HuggingFace embeddings** and **ChromaDB** for storing document vectors and enabling fast similarity searches.
* **Chat Engine**: Supports conversational memory, enabling context-aware responses.
* **Text-to-Speech (TTS)**: Converts responses into natural-sounding audio with Microsoft Edge TTS.
* **Gradio Demo**: Interactive UI for voice input and response playback.

## How to Run
1. **Run the API Server**: The API provides endpoints to interact with the chat engine and generate speech.
  ```bash
  uvicorn app:app --port 8000 --reload
  ```
  Endpoints:

* `POST /query` – Retrieve query response.

* `POST /tts` – Generate speech from text.

2. **Run the Gradio Voice Assistant Demo**: Launch the Gradio demo to test the end-to-end functionality
  ```bash
  python main_demo.py
  Or,
  uvicorn main_demo:app --port 8000 --reload
  ```

  * Accepts **voice input** from the user via microphone
  * **Transcribes** audio using speech recognition
  * **Retrieves** query response from query / chat engine
  * **Speaks** response in the selected AI generated voice 

3. **Run Speech Recognition Test**: Try out speech recognition functionality using Gradio.

  ```bash
  python gradio_sr.py
  Or,
  uvicorn gradio_sr:app --port 8000 --reload
  ```

## How It Works

1. **Indexing Data**

   * Documents in `./data` are embedded using `BAAI/bge-base-en-v1.5`.
   * Vectors are stored in **ChromaDB** (`./indexes/chroma`).
   * Collections allow **fast retrieval** of similar content during queries.

2. **Retrieval and Chat**

   * Queries use `VectorIndexRetriever` for fetching relevant chunks.
   * Responses are generated via **LLMs** (ChatGPT, Mistral AI, Hugging Face models, etc.).
   * The system supports conversational memory for natural dialogue.

3. **Text-to-Speech**

   * Responses can be synthesized into audio using **Edge TTS** with customizable voices.


## Tech Stack

* **FastAPI** – REST API framework
* **LlamaIndex** – Data indexing, retrieval & chat engine
* **ChromaDB** – Vector store for efficient similarity search
* **HuggingFace Embeddings** – To generate dense vector representations
* **Edge TTS** – Text-to-speech synthesis
* **Gradio** – For building interactive demos

## Highlights
* Quickly index and query your own document collections.
* Swap between LLMs (ChatGPT, HuggingFace, Mistral) with minimal changes.
* Supports multilingual TTS with lifelike voices.

