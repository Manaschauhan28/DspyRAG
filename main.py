import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import dspy

load_dotenv()


# === DSPy Signature & Bedrock Module ===
class CustomBedrockSignature(dspy.Signature):
    context = dspy.InputField(desc="Context from documents")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Concise answer")


class BedrockWrapper(dspy.Module):
    def __init__(self):
        super().__init__()
        self.llm = BedrockLLM(
            credentials_profile_name="default",
            model_id="amazon.titan-text-express-v1",
            model_kwargs={"temperature": 0.3}
        )

    def forward(self, context, question):
        prompt = (
            f"Use the given context to answer the question.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"If you don't know the answer, say you don't know. Be concise."
        )
        response = self.llm.invoke(prompt)
        return dspy.Prediction(answer=response)


# === RAG Pipeline using FAISS ===
class RAGFaissDSPyPipeline:
    def __init__(self, doc_path, faiss_path="./faiss_index"):
        self.doc_path = doc_path
        self.faiss_path = faiss_path

        self.llm = BedrockLLM(
            credentials_profile_name="default",
            model_id="amazon.titan-text-express-v1",
            model_kwargs={"temperature": 0.3}
        )

        self.embeddings = BedrockEmbeddings(
            credentials_profile_name="default",
            model_id="amazon.titan-embed-text-v2:0"
        )

    def load_documents(self):
        loader = UnstructuredWordDocumentLoader(self.doc_path)
        return loader.load()

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return splitter.split_documents(documents)

    def create_vector_store(self, docs):
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(self.faiss_path)
        return vector_store

    def build_chain(self, retriever):
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Keep the answer concise.\nContext: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, qa_chain)

    def run_query(self, query):
        print("Document loaded")
        documents = self.load_documents()
        docs = self.split_documents(documents)
        print("Addinng in Vector DB")
        vector_store = self.create_vector_store(docs)
        retriever = vector_store.as_retriever()
        print("Building Chain")
        chain = self.build_chain(retriever)

        # Run LangChain RAG
        response = chain.invoke({"input": query})
        context_text = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])

        # Use DSPy optimized Bedrock answer
        bedrock_model = BedrockWrapper()
        final_answer = bedrock_model(context=context_text, question=query).answer

        return final_answer


# === MAIN ===
if __name__ == "__main__":


    rag = RAGFaissDSPyPipeline(doc_path="data/GST_Smart_Guide.docx")
    query = input("Please enter your query: ")
    answer = rag.run_query(query)
    print("\nFinal Answer (via Bedrock + DSPy):", answer)
