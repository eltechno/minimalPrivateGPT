
from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from chromadb.config import Settings



persist_directory="db"
source_directory="./source/source.pdf"
embeddings_model_name="all-MiniLM-L6-v2"
chunk_size = 500
chunk_overlap = 50


CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory="db", anonymized_telemetry=False)

loader = PDFMinerLoader(source_directory)
documents = loader.load()

# Split text in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Create and store locally vectorstore
print("Loading PDF documents and creating a new vectorstore")

print(f"Creating embeddings. May take some minutes...")
#db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

db = Chroma.from_documents(texts, embeddings, chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)

db.persist()
db = None


#====================================================

#model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
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
    print(f"\nAnswer 💬: {answer}")
