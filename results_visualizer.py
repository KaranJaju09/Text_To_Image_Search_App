"""
This script provides a Streamlit web interface for a text-to-image search application.

It allows users to enter a text query and view the most relevant images
from a Milvus database populated by the `text_to_image_search.py` script.
"""

import streamlit as st
from pymilvus import MilvusClient
import clip
from PIL import Image
import torch
import os

# --- Configuration ---
MILVUS_URI = "./milvus.db"
COLLECTION_NAME = "image_embeddings"
MODEL_NAME = "ViT-B/32"

# --- Milvus and CLIP Model Initialization ---

@st.cache_resource
def get_milvus_client():
    """
    Initializes and returns a Milvus client.
    The `@st.cache_resource` decorator ensures that the client is created only once.
    """
    print(f"Initializing Milvus client, connecting to '{MILVUS_URI}'...")
    return MilvusClient(uri=MILVUS_URI)

milvus_client = get_milvus_client()

@st.cache_resource
def get_clip_model():
    """
    Loads and returns the CLIP model and preprocessor.
    The `@st.cache_resource` decorator ensures that the model is loaded only once.
    """
    print("Loading CLIP model...")
    device = "cpu" 
    model, preprocess = clip.load(MODEL_NAME, device=device)
    print(f"Model '{MODEL_NAME}' loaded on device: {device}")
    return model, preprocess, device

model, preprocess, device = get_clip_model()

# --- Text Encoding ---

def encode_text(text):
    """
    Encodes the given text query into a vector embedding using the CLIP model.
    
    Args:
        text (str): The text query to encode.
        
    Returns:
        list: The encoded text embedding.
    """
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().squeeze().tolist()

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Text-to-Image Search")

st.title("Text-to-Image Search with CLIP and Milvus")

# Check if the Milvus collection exists
has_collection = milvus_client.has_collection(collection_name=COLLECTION_NAME)
if not has_collection:
    st.error(f"Milvus collection '{COLLECTION_NAME}' not found. Please run `text_to_image_search.py` first to populate the database.")
    st.stop()

# --- Search Interface ---

query_text = st.text_input("Enter your search query :", placeholder="e.g., 'banana'")

num_results = st.slider("Number of results to display:", min_value=1, max_value=10, value=5)

if st.button("Search Images"):
    if query_text:
        st.write(f"Searching for: **{query_text}**")
        with st.spinner("Encoding text and searching Milvus..."):
            try:
                # Encode the text query
                query_embedding = encode_text(query_text)

                # Search Milvus for similar images
                search_results = milvus_client.search(
                    collection_name=COLLECTION_NAME,
                    data=[query_embedding],
                    limit=num_results,
                    output_fields=["image_path"],
                )

                # Display the search results
                if search_results and search_results[0]:
                    st.subheader("Search Results:")
                    cols = st.columns(5)
                    col_idx = 0
                    for hit in search_results[0]:
                        image_path = hit["entity"]["image_path"]
                        distance = hit["distance"]

                        if os.path.exists(image_path):
                            try:
                                img = Image.open(image_path)
                                with cols[col_idx % 5]:
                                    st.image(img, caption=f"Score: {distance:.4f}", use_container_width=True)
                                col_idx += 1
                            except Exception as e:
                                st.warning(f"Could not load image {image_path}: {e}")
                        else:
                            st.warning(f"Image file not found: {image_path}")
                else:
                    st.info("No results found for your query. Try a different one or ensure images are indexed.")
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
    else:
        st.warning("Please enter a search query.")

# --- Sidebar ---

st.sidebar.header("About")
st.sidebar.info("This is a simple text-to-image search application using CLIP and Milvus.")
st.sidebar.info("Enter a text query to find relevant images from the indexed dataset.")
st.sidebar.info(
    "The app is trained on following objects: bannister , kit_fox , barn_spider , bottlecap , goblet , chain_mail , killer_whale , capuchin , Airedale , triceratops , comic_book , orange , china_cabinet , bullet_train , oxcart , black_widow , wombat , toilet_seat , tub , basset , steam_locomotive , banana , Rhodesian_ridgeback , American_egret , conch , car_mirror , carousel , ruffed_grouse , cuirass , dugong , standard_poodle , electric_locomotive , castle , scabbard , cornet , cliff_dwelling , harmonica , red-backed_sandpiper , Lakeland_terrier , soft-coated_wheaten_terrier , grocery_store , mixing_bowl , lynx , African_crocodile , holster , lion , meat_loaf , turnstile , Afghan_hound , folding_chair , magpie , vine_snake , dishwasher , goldfish , half_track , remote_control , bicycle-built-for-two , junco , typewriter_keyboard , measuring_cup , white_stork , can_opener , warplane , flamingo , pinwheel , traffic_light , strainer , ambulance , toyshop , Doberman , theater_curtain , steel_arch_bridge , parachute , mongoose , hermit_crab , minibus , Bouvier_des_Flandres , jean , trilobite , rocking_chair , tiger_cat , ram , police_van , flatworm , cocktail_shaker , ski_mask , screen , pizza , brain_coral , combination_lock , malamute , loudspeaker , horizontal_bar , apiary , yawl , basketball , guillotine , knee_pad , safety_pin , hen-of-the-woods"
)