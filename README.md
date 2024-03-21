# Testing Framework 

Welcome to the testing framework repository! This project is a comprehensive testing solution built on top of the RAGAS framework, utilizing FastAPI for the server-side and Streamlit for the client-side interface.

## Installation

Before proceeding with the installation, ensure you have Python installed on your system. Follow these steps to set up the testing framework:

1. Install all dependencies listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to set up and run the testing framework:

### 1. Server-Side (FastAPI)

Run the server-side code using the following command:

```bash
uvicorn main:app --reload
```

This command starts the FastAPI server, allowing the client-side to interact with it.

### 2. Client-Side (Streamlit)

Run the client-side code using the following command:

```bash
streamlit run class.py
```

This command launches the Streamlit application, providing a user-friendly interface for interacting with the testing framework.

## Environmental Variables

Before running the commands mentioned above, ensure that you have set up your OpenAI API key as an environmental variable. This key is necessary for accessing the OpenAI API functionalities within the framework.
