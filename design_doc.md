
# Design Document: Text-to-Image Search using CLIP + Milvus

## 📁 Project Structure

```
text_to_image_search/
│
├── README.md                # Project overview and usage instructions
├── requirements.txt         # Python dependencies
├── images_folder/           # Contains training images for indexing
│   └── train/               # Subfolder with image files
├── text_to_image_search.py  # Script to index images into Milvus using CLIP
├── results_visualizer.py    # Streamlit web app to search images using text
├── milvus.db                # Milvus local vector database (auto-created)
```

## 🎯 Objective

This project builds a **semantic image search engine** that:
- Uses OpenAI’s **CLIP model** to extract image and text embeddings.
- Stores image embeddings in a **Milvus vector database**.
- Provides a **Streamlit UI** to search images using natural language.

## 🤖 CLIP Model Overview

CLIP (Contrastive Language–Image Pretraining), developed by **OpenAI**, is a vision-language model that:
- Encodes images and text into the **same embedding space**.
- Learns to match images with natural language descriptions.
- Enables **zero-shot** tasks like “find images that match this text”.

**Model used:** `ViT-B/32`
- `ViT`: Vision Transformer
- `B`: Base size (86M parameters)
- `32`: Image patch size
- Embedding dimension: **512**

## 🏗️ System Architecture

```
+--------------------------+
|  images_folder/train/    |
|  (Raw Images)            |
+------------+-------------+
             |
             v
+------------+-------------+
| text_to_image_search.py  |
|  - Uses CLIP to embed    |
|    images (512-d vectors)|
|  - Stores embeddings in  |
|    Milvus (milvus.db)    |
+------------+-------------+
             |
             v
+------------+-------------+
| Milvus Vector DB         |
|  - Stores image vectors  |
|  - Performs similarity   |
|    search on query vector|
+------------+-------------+
             ^
             |
+------------+-------------+
| results_visualizer.py    |
|  - Streamlit UI          |
|  - Encodes user text     |
|    query with CLIP       |
|  - Queries Milvus        |
|  - Displays similar      |
|    images                |
+--------------------------+
```

## ⚙️ Configuration & Parameters

| Parameter           | File                   | Description |
|--------------------|------------------------|-------------|
| `MODEL_NAME`       | Both                   | CLIP model name: `ViT-B/32` |
| `IMAGES_FOLDER`    | `text_to_image_search` | Directory path to indexed images |
| `COLLECTION_NAME`  | Both                   | Milvus collection: `image_embeddings` |
| `MILVUS_URI`       | Both                   | Milvus local DB path: `./milvus.db` |
| `device`           | Both                   | Inference device (`"cpu"`) |

## 📄 Script Breakdown

### 📌 `text_to_image_search.py`

This script:
1. Loads the CLIP model.
2. Initializes a Milvus client.
3. Creates a collection (if not exists) with fields:
   - `id` (INT64)
   - `vector` (FLOAT_VECTOR, dim=512)
   - `image_path` (VARCHAR)
4. Recursively scans images in `images_folder/train/`
5. For each image:
   - Preprocesses it using CLIP
   - Encodes it into a 512-d vector
   - Normalizes the embedding
6. Inserts image vectors + metadata into Milvus.

> When run, this script prepares the vector DB for semantic search.

### 📌 `results_visualizer.py`

This Streamlit UI:
1. Loads the same CLIP model (`ViT-B/32`).
2. Connects to the local Milvus database.
3. Provides an input box for text queries.
4. Encodes the text using CLIP → 512-d vector.
5. Sends query to Milvus:
   - Finds top-N similar image vectors using **L2 distance**
6. Displays matching image results in a grid with distance scores.

Other features:
- Uses `@st.cache_resource` to **avoid reloading model & DB** on rerun.
- Sidebar shows supported search categories.
- Handles errors for missing collection or missing images.

## 💡 Design Choices

| Component        | Choice                  | Rationale |
|------------------|--------------------------|-----------|
| Embedding Model | CLIP (ViT-B/32)          | Lightweight + accurate image-text pairing |
| DB Engine       | Milvus (local)           | Fast, scalable vector search |
| Index Type      | IVF_FLAT + L2            | Balanced speed and accuracy |
| UI Framework    | Streamlit                | Simple interactive prototyping |
