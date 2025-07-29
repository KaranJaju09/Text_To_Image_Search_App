# Text-to-Image Search

This project implements a text-to-image search engine using CLIP and Milvus.

## Description

The `text_to_image_search.py` script indexes a directory of images, creating embeddings for each image using the CLIP model. These embeddings are then stored in a Milvus collection.

The `results_visualizer.py` script provides a Streamlit web interface to search the indexed images. Users can enter a text query, and the application will return the most relevant images from the Milvus collection.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download the training data:**

    Download the training data and extract it into the `images_folder/train` directory.

## Usage

1.  **Index the images:**

    Run the `text_to_image_search.py` script to index the images in the `images_folder/train` directory:

    ```bash
    python text_to_image_search.py
    ```

2.  **Run the web interface:**

    Run the `results_visualizer.py` script to start the Streamlit web interface:

    ```bash
    streamlit run results_visualizer.py
    ```

    The web interface will be available at `http://localhost:8501`.
