# pinecone-openai-sample
Test the simple application that saves and loads vector data made with OpenAI and Pinecone.

# What you can do
- generate_vectors.py：create vectors into your Pinecone index.
- delete_vectors.py：delete vectors on your Pinecone index.
- main.py: Search the most relevant vector index from your Pinecone index, then create prompt and get response from GPT. Finally store response vector data.

# Setup
First of all, install packages with requirements.txt
```
pip install -r app/requirement.txt
```

Create .env file, then fullfil the items below
```
OPENAI_API_KEY = ""
PINECONE_API_KEY = ""
PINECONE_ENVIRONMENT = ""
PINECONE_INDEX_NAME = ""
```

Before execute main.py, recommend to create vectors with trec and OpenAI by running generate_vectors.py
```
python app/generate_vectors.py
```
