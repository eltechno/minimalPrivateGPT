from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from chromadb.config import Settings


persist_directory = "db"
source_directory = "./source/source.pdf"
embeddings_model_name = "all-MiniLM-L6-v2"
chunk_size = 500
chunk_overlap = 50


CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory="db", anonymized_telemetry=False
)

#this is the ingestion 

loader = PDFMinerLoader(source_directory)
documents = loader.load()

# Split text in chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Create and store locally vectorstore
print("Loading PDF documents and creating a new vectorstore")

print(f"Creating embeddings. May take some minutes...")
# db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

db = Chroma.from_documents(
    texts,
    embeddings,
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
)

db.persist()
db = None #clear variable to free memory

# The above code loads PDF documents from a source directory using the PDFMinerLoader class. 
#It then splits the text in the documents into chunks using RecursiveCharacterTextSplitter. 
#Next, it creates embeddings for each chunk of text using a pre-trained Hugging Face model 
#specified by `embeddings_model_name`. Finally, it creates a Chroma vector store from the 
#texts and embeddings using `Chroma.from_documents()`, which returns an instance of 
#`Chroma` that can be used to perform similarity search operations on the vectors. 
#The resulting vector store is stored locally in `persist_directory`.


# ====================================================
# The following code initializes a Chroma vector store using HuggingFaceEmbeddings as the embedding function.
# It uses a pre-trained model located at "models/ggml-gpt4all-j.bin" to generate embeddings.
# The resulting Chroma vector store is saved in the "db" directory and uses client settings defined in CHROMA_SETTINGS. 

#this is the QA

# model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
model_path = "models/ggml-gpt4all-j.bin"
embedding2 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
db = Chroma(
    persist_directory="db",
    embedding_function=embedding2,
    client_settings=CHROMA_SETTINGS,
)

retriever = db.as_retriever()
llm = GPT4All(model=model_path, n_ctx=1000, backend="gptj")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
while True:
    query = input("Ask your Document 💬")
    if query == "exit":
        break
    answer = qa(query)["result"]
    print(f"\nQuestion: {query}")
    print(f"\nAnswer 💬 : {answer}")
