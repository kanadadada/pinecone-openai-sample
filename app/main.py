import argparse
import os
import openai
import pinecone
import uuid
from dotenv import load_dotenv
from generate_vectors import embedding_from_text

def main():
  # Load environment variables
  load_dotenv()
  
  # Get CLI text input
  parser = argparse.ArgumentParser()
  parser.add_argument("text", help="Input message to be user as prompt")
  args = parser.parse_args()
  
  # Set your API key
  openai.api_key = os.getenv("OPENAI_API_KEY")
    
  # Initialize Pinecone
  pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
  )
  index_name = os.getenv("PINECONE_INDEX_NAME")
  index = pinecone.Index(index_name)
  
  # Set message from CLI argument
  user_message = args.text
  
  # Create an embeddings of message
  user_message_embedding = embedding_from_text(user_message)
  
  # fetch the most similar conversation from Pinecone
  query_result = index.query(queries=[user_message_embedding], top_k=1)
  if query_result["results"][0]["matches"]:
    most_similar_id = query_result["results"][0]["matches"][0]["id"]
    fetch_response = index.fetch(ids=[most_similar_id])
    if fetch_response["vectors"][most_similar_id]["metadata"]["text"]:
      most_similar_conversation = fetch_response["vectors"][most_similar_id]["metadata"]["text"]
    else:
      most_similar_conversation = ""
      print("No text on metadata found in Pinecone.")
  else:
    most_similar_conversation = ""
    print("No similar conversation found in Pinecone.")
  
  # Append the user's message to the conversation
  prompt = f'''
  You are an AI dialogue assistant. Please be considerate of your dialogue partner and continue the dialogue.
  Please keep in mind the context of the dialogue you have just had. The following is the context information of the dialogue.
  {most_similar_conversation}
  '''
  
  conversation = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user_message},
  ]
  
  # Generate a response with the OpenAI Chat API
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=conversation,
  )
  
  # Print the assistant's message
  response_text = response["choices"][0]["message"]["content"]
  print (response_text)
  
  # Create an embeddings of the assistant's message
  response_embedding = embedding_from_text(response_text)
  
  # Create an ID for the response_embedding
  response_id = str(uuid.uuid4())
  
  # Upsert the conversation to Pinecone
  index.upsert(vectors=[{"id": response_id, "values": response_embedding}], namespace="response_conversations")

if __name__ == "__main__":
  main()