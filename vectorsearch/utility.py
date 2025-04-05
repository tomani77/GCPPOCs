import json
from google.cloud import storage
from google import genai
import tqdm
import time

def load_jsonl_data_from_bucket(bucket_name, file_name, project_id):
    """Loads restaurant data from a JSONL file in GCS."""
    print(f"Loading data from bucket: {bucket_name}, file: {file_name}, project: {project_id}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    jsonl_data = blob.download_as_text()
    # Parse JSONL into an array of objects
    data = []
    for line in jsonl_data.splitlines():
        if line:  # Skip empty lines
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}. Error: {e}")

    return data    

def load_data_from_bucket(bucket_name, file_name, project_id):
    """Loads restaurant data from a JSONL file in GCS."""
    print(f"Loading data from bucket: {bucket_name}, file: {file_name}, project: {project_id}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    jsonl_data = blob.download_as_text()
    # Parse JSON
    data = json.loads(jsonl_data)
    return data
    # Parse JSONL into an array of objects
    data = []
    # for line in jsonl_data.splitlines():
    #     if line:  # Skip empty lines
    #         try:
    #             data.append(json.loads(line))
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON line: {line}. Error: {e}")

    return data

def upload_embeddings_bucket(embeddings_data, bucket_name, project_id, filename="synthetic_restaurants_embedded.json"):
    """Uploads embeddings data to a GCS bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)

    # Convert list of JSON strings to newline-separated JSONL format
    jsonl_content = "\n".join(embeddings_data)
    blob.upload_from_string(jsonl_content, content_type="application/json")
    print(f"Embeddings uploaded to gs://{bucket_name}/{filename}")

def get_embeddings_wrapper(texts: list[str]) -> list[list[float]]:
    
    PROJECT_ID = "agentic-ai-poc"  
    LOCATION = "us-central1"
    BATCH_SIZE = 5
    EMBEDDING_MODEL_NAME = "text-embedding-005" 

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    embeddings: list[list[float]] = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME, contents=texts[i : i + BATCH_SIZE]
        )
        embeddings = embeddings + [e.values for e in response.embeddings]
    return embeddings
