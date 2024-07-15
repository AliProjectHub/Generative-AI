# LLM Analytics Banker

## Overview

LLM Analytics Banker is a Streamlit-based web application designed to provide insights and answers related to the financial performance of a company. Utilizing the powerful capabilities of the Hugging Face transformers and the Llama Index, this app reads and processes financial documents to generate meaningful responses to user queries.

## Features

- **Interactive User Interface**: Simple and intuitive UI built with Streamlit.
- **Natural Language Processing**: Uses advanced NLP models from Hugging Face for understanding and generating text.
- **Document Embeddings**: Employs embeddings to represent document chunks for efficient querying.
- **PDF Document Loading**: Supports loading financial documents in PDF format.

## Installation

To get started with LLM Analytics Banker, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/Generative-AI.git
    cd Generative-AI
    ```

2. **Install Dependencies**:
    Ensure you have Python installed (version 3.7 or higher), then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Hugging Face Authentication**:
    Replace `"your_auth_token"` with your actual Hugging Face API token in the script.

## Usage

1. **Prepare the Environment**:
    Ensure your environment is set up correctly with all dependencies installed.

2. **Run the Streamlit App**:
    Start the Streamlit server to launch the app:
    ```bash
    streamlit run LLM_Analytics.py
    ```

3. **Load PDF Document**:
    Place the financial document you want to analyze in the appropriate directory and update the file path in the script if necessary.

4. **Interact with the App**:
    Open the local URL provided by Streamlit, input your prompt in the text box, and hit enter to receive responses based on the content of the loaded financial document.

## Code Explanation

### Imports

- **Streamlit**: For building the web application.
- **Transformers**: For using Hugging Face's pre-trained models.
- **Torch**: For handling tensor data types.
- **Llama Index**: For generating prompts and wrapping Hugging Face models.
- **Langchain**: For managing embeddings.
- **Pathlib**: For handling file paths.

### Model and Tokenizer Initialization

The script initializes a tokenizer and a language model from the Hugging Face model hub, specifically the "meta-llama/Llama-2-7b-chat-hf" model.

### Prompts

- **System Prompt**: Defines the behavior and guidelines for the assistant.
- **Query Wrapper Prompt**: Wraps the user query to format it correctly for the model.

### LLM and Embeddings

- **LLM**: Configures the language model with specific parameters.
- **Embeddings**: Creates an instance for document embeddings using Langchain and Hugging Face models.

### Service Context

Sets up the service context, which includes chunk size, language model, and embeddings. This context is then set globally.

### Document Loading

Loads the financial document (PDF) using `PyMuPDFLoader` and processes it to extract text.

### Indexing and Query Engine

Creates a vector store index from the loaded documents and initializes a query engine to handle user queries.

### Streamlit Interface

- **Title**: Displays the main title of the app.
- **Input Box**: Provides a text input box for user queries.
- **Response Handling**: Processes the user's input, queries the document, and displays the response.

## Example Usage

1. **Launch the App**:
    ```bash
    streamlit run LLM_Analytics.py
    ```

2. **Enter a Prompt**:
    Type your query into the text input box, such as "What was the net income for the year 2023?"

3. **View the Response**:
    The app will display the response along with the source text used to generate it.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
