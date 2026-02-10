import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

loader = PyPDFLoader("Bikash_AMEX.pdf")
docs = loader.load()
# print(f"Number of documents loaded: {len(docs)}")
# print("First document content:")
# print(docs[0].page_content[:300]) #outputs the first 300 characters
# print("Metadata of the first document:")
# print(docs[0].metadata)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)
# print(f"Number of chunks created: {len(chunks)}")
# print("First chunk content:")
# print(chunks[0].page_content[:300]) #outputs the first 300 characters

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Using a local LLM model (no API key required)
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 256, "temperature": 0.1}
)

print("Setup complete. You can now ask questions about the document.")

# Format retrieved documents into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("Answer the following question based on the context:\n{context}\n\nQuestion: {input}")

# Create retrieval chain using LCEL
retrieval_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) 

while True:
    query = input("Enter your Question(or 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break
    result = retrieval_chain.invoke(query)
    print("Answer:", result)
