from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Connect to your specific index
index = pc.Index(os.environ["INDEX_NAME"])

# Delete all vectors in the default namespace
index.delete(delete_all=True)
