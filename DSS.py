from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import os
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS 
import fitz 
 
from dotenv import load_dotenv 
load_dotenv() 
 
API_KEY = os.getenv('API_KEY') 
strings = "" 
with fitz.open(r"IPC_186045.pdf") as pdf: 
    for page_num in range(15, len(pdf)): 
        page = pdf[page_num] 
        strings += page.get_text("text") 

text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size=1000, chunk_overlap=200 
) 

split_docs = text_splitter.split_text("".join(strings)) 
split_docs = [Document(page_content=t) for t in split_docs] 

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=API_KEY, 
    task_type="retrieval_document"
) 

if not os.path.exists("faiss_index_ipc"):
    vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
    vector_store.save_local("faiss_index_ipc")

db = FAISS.load_local("faiss_index_ipc", embeddings, allow_dangerous_deserialization=True)
 
# Take query input from the user
query = input("Enter your query: ")

docs = db.similarity_search_with_score(query)
lst = []

print("Using FAISS direct search")

for doc, score in docs:
    lst.append((doc.page_content, "Probability: ", 1 - score))
    for _ in lst:
        print(_[0])
        print(_[1], _[2])
        print("______________")
