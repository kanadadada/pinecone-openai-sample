import os
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone and connect to your index
pinecone.init(
  api_key=os.getenv("PINECONE_API_KEY"),
  environment=os.getenv("PINECONE_ENVIRONMENT"),
)
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pinecone.Index(index_name)

# Delete all the vectors in the index
index.delete(
  delete_all=True
)