
import os  
from openai import AzureOpenAI  

endpoint = os.getenv("ENDPOINT_URL", "https://openai-charter.openai.azure.com/")  
# deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-technical-task")  
deployment = os.getenv("EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-small")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")  

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)

response = client.chat.completions.create(
    model="gpt-4o-technical-task",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(response.choices[0].message.content)

endpoint = "https://openai-charter.openai.azure.com/"
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=subscription_key
)

response = client.embeddings.create(
    model=model_name,
    input="Hello, world!"
)

print(response)
