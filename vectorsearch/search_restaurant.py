import json
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from utility import load_data_from_bucket,upload_embeddings_bucket, get_embeddings_wrapper
from google import genai
import tqdm

# GCP Configuration
PROJECT_ID = "agentic-ai-poc"  # Replace with your GCP project ID
LOCATION = "us-central1"
BUCKET_NAME = "manish-synthetic" # Ensure bucket exists
EMBEDDING_MODEL_NAME = "text-embedding-005" #"textembedding-gecko@001" # Or your preferred model
EMBEDDED_FILE_NAME = "synthetic_restaurants.json"

# Index and Endpoint details
DEPLOYED_INDEX_ID = "rest_deployed_1743845886277" # Replace with your Index ID
INDEX_ENDPOINT_ID = "2001564161342963712" # Replace with your Endpoint ID

vertexai.init(project=PROJECT_ID, location=LOCATION)
restaurants_data = load_data_from_bucket(BUCKET_NAME, EMBEDDED_FILE_NAME, PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)


def query_vector_search(query):
    """
    Performs vector search using a query and a deployed index.

    Args:
        query: The query string.

    Returns:
        A list of matching restaurant results.
    """
    # embeddings = embedding_model.get_embeddings([query])
    # query_embedding = [embedding.values for embedding in embeddings][0]
    # #print('query embedding ' + str(query_embedding))

    query_embeddings = get_embeddings_wrapper([query])
    
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}",
            project=PROJECT_ID,
            location=LOCATION,
        )

    # response = index_endpoint.find_neighbors(
    #     deployed_index_id=DEPLOYED_INDEX_ID,
    #     queries=[query_embedding],
    #     num_neighbors=5,  # Adjust as needed
    # )

    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=query_embeddings,
        num_neighbors=5,  # Adjust as needed
    )

    # Extract the list of MatchNeighbor objects directly
    neighbors = response[0]  # Access the first (and only) inner list

    results = []

    # Iterate through the MatchNeighbor objects and extract the data
    for neighbor in neighbors:
        result_index = int(neighbor.id) -1 
        print (neighbor.id + " "  + str(neighbor.distance))
        # print("Neighbor ID:", neighbor.distance)
        results.append(restaurants_data[result_index])

    return results

query = "find  veg Indian restaurant which dont serve non veg food in area 60601"
search_results = query_vector_search(query)

print("Search Results:")
for restaurant in search_results:
    print({
        "id": restaurant["id"],
        "name": restaurant["name"],
        "zip_code": restaurant["zip_code"],
        "cuisine": restaurant["cuisine"],
        "veg": restaurant["veg"],
        "non_veg": restaurant["non_veg"],
        "kids_friendly": restaurant["kids_friendly"],
        "wheelchair_accessible": restaurant["wheelchair_accessible"],
        "average_rating": restaurant["average_rating"],
        # "credit_card_benefits": restaurant.get("credit_card_benefits", [])
    })

