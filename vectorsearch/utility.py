import json
from google.cloud import storage

def load_jsonl_data_from_bucket(bucket_name, file_name,project_id):
    """Loads restaurant data from a JSONL file in GCS."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    jsonl_string = blob.download_as_string().decode("utf-8")
    data = []
    for line in jsonl_string.splitlines():
        data.append(json.loads(line))
    return data

def load_data_from_bucket(bucket_name, file_name, project_id):
    """Loads restaurant data from a JSONL file in GCS."""
    print(f"Loading data from bucket: {bucket_name}, file: {file_name}, project: {project_id}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    json_data = blob.download_as_text()
    # Parse JSON
    data = json.loads(json_data)
    return data

def upload_embeddings_bucket(embeddings_data, bucket_name, project_id, filename="synthetic_restaurants_embedded.json"):
    """Uploads embeddings data to a GCS bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(json.dumps(embeddings_data), content_type="application/json")
    print(f"Embeddings uploaded to gs://{bucket_name}/{filename}")
