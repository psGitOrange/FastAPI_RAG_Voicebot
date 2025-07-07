from huggingface_hub import login, whoami
import os

from dotenv import load_dotenv
load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
print("Successfully logged in to Hugging Face Hub")

# Get user info
user_info = whoami(token=hf_token)
print(f"Logged in as: {user_info['name']}")
print(f"Email: {user_info.get('email', 'Not provided')}")
print(f"Organization: {user_info.get('orgs', 'None')}")