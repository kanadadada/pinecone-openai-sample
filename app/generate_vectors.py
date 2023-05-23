import os
import openai
import pinecone
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

def embedding_from_text(text: str):
  # Create an embeddings of message
  embedding_response = openai.Embedding.create(
    model = "text-embedding-ada-002",
    input = text,
  )
  message_embedding = embedding_response["data"][0]["embedding"]
  return message_embedding

def main():
  # Load environment variables
  load_dotenv()
  
  # Set your API key
  openai.api_key = os.getenv("OPENAI_API_KEY")
    
  # Initialize Pinecone and connect to your index
  pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
  )
  index_name = "sample-app"
  index = pinecone.Index(index_name)

  # Create a TREC dataset to connect OpenAI to Pinecone
  trec = load_dataset("trec", split="train[:1000]")
    
  # upsert the ID, vector embedding, and original text for each phrase to Pinecone.
  batch_size = 32
  for i in tqdm(range(0, len(trec["text"]), batch_size)):
    i_end = min(i + batch_size, len(trec["text"]))
    lines_batch = trec["text"][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
      
    res = openai.Embedding.create(input=lines_batch, model="text-embedding-ada-002")
    embeds = [record["embedding"] for record in res["data"]]
      
    # prepare metadata
    meta = [{"text": line} for line in lines_batch]
      
    # prepare to_upsert list with dictionaries
    to_upsert = zip(ids_batch, embeds, meta)
      
    # upsert to Pinecone
    index.upsert(vectors=to_upsert, namespace="openai_trec")

if __name__ == "__main__":
  main()