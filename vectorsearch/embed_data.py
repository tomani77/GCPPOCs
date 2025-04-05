import json
from google import genai
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from utility import load_data_from_bucket,upload_embeddings_bucket, get_embeddings_wrapper
import tqdm
import time

# GCP Configuration
PROJECT_ID = "agentic-ai-poc"  # Replace with your GCP project ID
LOCATION = "us-central1"
JSONL_FILE_NAME = "synthetic_restaurants.json" #replace with your JSONL file path
BUCKET_NAME = "manish-synthetic" # Ensure bucket exists
EMBEDDING_MODEL_NAME = "text-embedding-005" #"textembedding-gecko@001" # Or your preferred model
EMBEDDED_FILE_NAME = "synthetic_restaurants_embedded.json"


# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)
vertexai.init(project=PROJECT_ID, location=LOCATION)




def generate_embeddings(text_list, model_name=EMBEDDING_MODEL_NAME):
    """Generates embeddings for a list of texts using Vertex AI."""
    print(f"Generating embeddings for {len(text_list)} texts.")
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings = model.get_embeddings(text_list)
    return [embedding.values for embedding in embeddings]

def prepare_text_for_embedding(restaurant):
   # Process credit card benefits
    # card_benefits = ", ".join(
    #     [f"{benefit['bank']} ({benefit['card_type']})" for benefit in restaurant.get("credit_card_benefits", [])]
    # )
    # card_text = f"Accepted Cards: {card_benefits}" if card_benefits else "No card benefits available"
    
    # Create text representation for embedding
    # text_representation = f"{restaurant['name']}, Cuisine: {restaurant['cuisine']}, "
    # text_representation += f"Veg: {restaurant['veg']}, Non-Veg: {restaurant['non_veg']}, "
    # text_representation += f"Kids Friendly: {restaurant['kids_friendly']}, "
    # text_representation += f"Wheelchair Accessible: {restaurant['wheelchair_accessible']}, "
    # text_representation += card_text

    # text_representation = f"{restaurant['name']} is a  {restaurant['cuisine']} restaurant, "
    # text_representation += f"{'Vegetarian options available.' if restaurant['veg'] else ''} "
    # text_representation += f"{'Non-vegetarian options available.' if restaurant['non_veg'] else ''} "
    # text_representation += f"{'Kids-friendly.' if restaurant['kids_friendly'] else ''} "
    # text_representation += f"{'Wheelchair accessible.' if restaurant['wheelchair_accessible'] else ''}"
    # text_representation += f"{restaurant['name']} is a located in zip code  {restaurant['zip_code']}, "
    # text_representation += card_text

    # Create a rich text description for better embeddings
    # text_representation  = f"""
    # Restaurant {restaurant['name']} serves {restaurant['cuisine']} cuisine in area {restaurant['zip_code']}.
    # {'' if restaurant['veg'] else 'Not '}Vegetarian friendly. 
    # {'' if restaurant['non_veg'] else 'Not '}Non-vegetarian friendly.
    # {'Kids friendly' if restaurant['kids_friendly'] else 'Not kids friendly'}.
    # {'Wheelchair accessible' if restaurant['wheelchair_accessible'] else 'Not wheelchair accessible'}.
    # """

    text_representation  = f"{restaurant['name']}  {restaurant['cuisine']} restaurant in zip code {restaurant['zip_code']},  " \
    f"{'Vegetarian Only available ' if restaurant.get('veg', False) and not restaurant.get('non_veg', False) else 'Vegetarian & Non-Vegetarian available' if restaurant.get('veg', False) and restaurant.get('non_veg', False) else 'Non-Vegetarian Only'} , Rating: {restaurant.get('average_rating', 'N/A')}, " \
    f"{'Wheelchair Friendly' if restaurant.get('wheelchair_accessible', False) else 'No Wheelchair Access'}"

    print (text_representation)

    return text_representation


# def create_embeddings(text_list):
#     """Creates embeddings and uploads them to GCS."""

#     embeddings = generate_embeddings(text_list)

#     embeddings_data = []
#     for i, restaurant in enumerate(restaurant_data):
#         embeddings_data.append(json.dumps({
#         "id": restaurant["id"],
#         # "name": restaurant["name"],
#         # "zip_code": restaurant["zip_code"],
#         # "cuisine": restaurant["cuisine"],
#         # "veg": restaurant["veg"],
#         # "non_veg": restaurant["non_veg"],
#         # "kids_friendly": restaurant["kids_friendly"],
#         # "wheelchair_accessible": restaurant["wheelchair_accessible"],
#         # "average_rating": restaurant["average_rating"],
#         # "credit_card_benefits": restaurant.get("credit_card_benefits", []),
#         "embedding": embeddings[i]
#     }))

#     return embeddings_data

def merge_restaurant_data_with_embedding(restaurant_data, embeddings):
    """Creates embeddings and uploads them to GCS."""

    embeddings_data = []
    for i, restaurant in enumerate(restaurant_data):
        embeddings_data.append(json.dumps({
        "id": restaurant["id"],
        # "name": restaurant["name"],
        # "zip_code": restaurant["zip_code"],
        #"cuisine": restaurant["cuisine"],
        # "veg": restaurant["veg"],
        # "non_veg": restaurant["non_veg"],
        # "kids_friendly": restaurant["kids_friendly"],
        # "wheelchair_accessible": restaurant["wheelchair_accessible"],
        # "average_rating": restaurant["average_rating"],
        # "credit_card_benefits": restaurant.get("credit_card_benefits", []),
        "embedding": embeddings[i]
    }))

    #print(embeddings_data)
    return embeddings_data

# Example usage (replace with your file path)
restaurant_data = load_data_from_bucket(BUCKET_NAME, JSONL_FILE_NAME, PROJECT_ID)  
text_list = [prepare_text_for_embedding(restaurant) for restaurant in restaurant_data]
embeddings = get_embeddings_wrapper(text_list)
embedded_data_list = merge_restaurant_data_with_embedding(restaurant_data, embeddings)

#embedded_data_list = create_embeddings(text_list)
upload_embeddings_bucket(embedded_data_list,BUCKET_NAME, EMBEDDED_FILE_NAME)
