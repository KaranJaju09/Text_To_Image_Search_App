"""
This script indexes a directory of images, creating embeddings for each image using the CLIP model.
These embeddings are then stored in a Milvus collection for efficient similarity search.
"""
import os
import torch
import clip
from PIL import Image
from pymilvus import MilvusClient, DataType

def initialize_database():
    # Configuration 
    MODEL_NAME = "ViT-B/32"
    COLLECTION_NAME = "image_embeddings"
    IMAGES_FOLDER = "images_folder/train"
    MILVUS_URI = "./milvus.db"

    # Load CLIP Model
    print("Loading CLIP model...")
    device = "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    print(f"Model '{MODEL_NAME}' loaded on device: {device}")

    # Initialize Milvus Client
    print(f"Initializing Milvus client, connecting to '{MILVUS_URI}'...")
    milvus_client = MilvusClient(uri=MILVUS_URI)

    # Create Milvus Collection (if it doesn't exist)
    has_collection = milvus_client.has_collection(collection_name=COLLECTION_NAME)

    if not has_collection:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating now...")
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        
        # Define schema fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=2048)

        # Define index parameters
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128},
        )

        # Create collection
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    else:
        print(f"Using existing collection: '{COLLECTION_NAME}'")

    # Image Indexing
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Folder not found:'{IMAGES_FOLDER}'.")

    # Recursively find all image files in the specified folder
    image_paths = []
    print(f"Recursively scanning for images in '{IMAGES_FOLDER}'...")
    for root, _, files in os.walk(IMAGES_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    # Process and insert image embeddings into Milvus
    if not image_paths:
        print(f"No images found in '{IMAGES_FOLDER}'. Please add images and run the script again.")
    else:
        data_to_insert = []
        print(f"Found {len(image_paths)} images. Processing and inserting into Milvus...")
        for image_path in image_paths:
            try:
                # Preprocess image and generate embedding
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # Prepare data for insertion
                data_to_insert.append({
                    "vector": image_features.cpu().numpy().flatten().tolist(),
                    "image_path": image_path
                })
            except Exception as e:
                print(f"Could not process image {image_path}: {e}")

        # Insert data into Milvus
        if data_to_insert:
            res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
            print(f"Successfully inserted {len(res['ids'])} image embeddings into Milvus.")
            milvus_client.flush(collection_name=COLLECTION_NAME)
            print("Flushed data to disk.")

    print("Script finished.")

    # Close Milvus Connection
    milvus_client.close()
    print("Milvus client connection closed.")