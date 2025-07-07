# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings  #, set_global_tokenizer
# from transformers import AutoTokenizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
load_dotenv()

# from huggingface_hub import login

# hf_token = os.getenv("HF_TOKEN")
# login(token=hf_token)
# print("Successfully logged in to Hugging Face Hub")

# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "HuggingFaceH4/zephyr-7b-alpha"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

# set_global_tokenizer(
#     AutoTokenizer.from_pretrained(MODEL_NAME).encode
# )

# print(os.getenv("HF_TOKEN"))
# hf_llm = HuggingFaceInferenceAPI(
#     model_name=MODEL_NAME, token=os.getenv("HF_TOKEN"))

# print(os.getenv("MISTRAL_API_KEY"))
mistral_llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
# resp = mistral_llm.complete("Paul Graham is ")
# print(resp)

Settings.llm = mistral_llm  # Set as global LLM

def create_chat_engine(index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=mistral_llm,
        system_prompt=(
            "You are a chatbot named 'Meera'. Help users solve their queries in a friendly conversational manner."
            "be concise but if question repeats again elaborate response in simple english"
            "If three consecutive questions are similar, prompt the user to raise a support ticket instead"
        ),
    )

    return chat_engine