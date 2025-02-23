# AI-Powered File QA Bot

This Streamlit application allows you to ask questions based on the content of a local text file. It utilizes Pinecone for efficient semantic search and a question-answering pipeline from Hugging Face Transformers to generate answers.

## Features

-   **File Input:** Users can provide the path to a local text file.
-   **Content Storage:** The file content is processed, embedded, and stored in a Pinecone vector database.
-   **Question Answering:** Users can ask questions related to the file content, and the application retrieves relevant information and generates answers.
-   **Semantic Search:** Pinecone's vector database enables efficient semantic search to find the most relevant information for each query.

## Prerequisites

-   Python 3.7+
-   Streamlit
-   Pinecone Python client
-   Sentence Transformers
-   Transformers
-   Requests
-   Beautiful Soup 4
-   Python-dotenv

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv env
    # For Windows:
    env\Scripts\activate
    # For macOS/Linux:
    source env/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install streamlit pinecone-client sentence-transformers transformers requests beautifulsoup4 python-dotenv
    ```

4.  **Set up environment variables:**

    -   Create a `.env` file in the same directory as your script.
    -   Add your Pinecone and Hugging Face API keys and your Pinecone environment to the `.env` file:

        ```
        PINECONE_API_KEY=your_pinecone_api_key
        HUGGINGFACE_API_KEY=your_huggingface_api_key
        PINECONE_ENVIRONMENT=your_pinecone_environment
        ```

    -   Replace `your_pinecone_api_key`, `your_huggingface_api_key`, and `your_pinecone_environment` with your actual API keys and environment.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run project.py
    ```

   

2.  **Enter the path to your text file:**

    -   In the Streamlit app, enter the path to the text file you want to use.

3.  **Ask questions:**

    -   After the file content is processed, you can enter questions related to the file content.
    -   The application will retrieve relevant information and generate answers.

## Code Explanation

-   The application uses Pinecone to store and query vector embeddings of the file's content.
-   Sentence Transformers are used to generate embeddings.
-   The Hugging Face Transformers library provides a question-answering pipeline.
-   Streamlit is used to create the user interface.
-   The dotenv library is used to load environment variables.

## Notes

-   Ensure that your Pinecone index has the correct dimensions (384) and metric (cosine).
-   Replace placeholder values (API keys, file paths) with your actual values.
-   The application processes text files line by line.
-   Error handling is included for file reading and web scraping.
-   The index will be deleted and recreated every time the script is run.