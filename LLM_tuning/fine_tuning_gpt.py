import json
import openai
import os
import pandas as pd
from pprint import pprint
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
env_path = Path('C:/Users/hanna/OneDrive - Universität des Saarlandes/Dokumente/Hiwi 09.2024/task generation/Mareikes_key.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("API Key not found")
else:
    print("API Key loaded successfully")

client = OpenAI(api_key=api_key)



training_file_name = "train_openai.jsonl"
validation_file_name = "val_openai.jsonl"

def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id

training_file_id = upload_file(training_file_name, "fine-tune")
validation_file_id = upload_file(validation_file_name, "fine-tune")

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

MODEL = "gpt-4o-mini-2024-07-18"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=MODEL,
    suffix="health_insurance_churn",
)

job_id = response.id

print("Job ID:", response.id)
print("Status:", response.status)

response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

response = client.fine_tuning.jobs.list_events(job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)
    
response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model

if fine_tuned_model_id is None:
    raise RuntimeError(
        "Fine-tuned model ID not found. Your job has likely not been completed yet."
    )

print("Fine-tuned model ID:", fine_tuned_model_id)