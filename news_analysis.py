import pandas as pd
import chromadb as cd 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


news = pd.read_csv("labelled_newscatcher_dataset.csv", sep=";")

#fields assigned to the news document
max_news = 1000
document = 'title'
topic = 'topic'
subset_news = news.head(max_news)

#import API
chroma_client = cd.PersistentClient()

collection_name = "news_collection"

#checking if the news_collection is unique or not delete the collection
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)

collection.add(documents=subset_news[document].tolist(), metadatas=[{topic: t} for t in subset_news[topic].tolist()], ids=[f"id{x}" for x in range(len(subset_news))],)

res = collection.query(query_texts=["laptop"], n_results=10)

#picking minillama as a huggingface model that is directly from the alpha LLM (from llama)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

pipe = pipeline(
    "text_generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device_map=auto,
)

#constructing text prompt
question = "Can This Laptop be toshiba?"
context = "".join([f"#{str(i)}" for i in results["documents"][0]])

#context = context[0:5120]
prompt_template = f""
relevant_context = {context}

#obtaining the response
lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])


#obtaining the collections by simply using their names
res_2 = collection.query(query_texts=["laptop"], n_results=10)
#print(res_2) quick and easy IF the query is in the server already


#connecting to chromadb's server
cl = chromadb.HttpClient(host="localhost", port=8000)
collection_local = cl.get_collection(name="local_news_collection")
res_3 = collection_local.get_query(query_textx=["laptop"], n_results=10)
print(res_3)
