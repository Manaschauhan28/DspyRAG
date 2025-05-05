# RAG Pipeline with FAISS, DSPy, and Amazon Bedrock

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:

- 🧠 **Amazon Bedrock** (Titan LLM + Embeddings)
- 🔍 **LangChain** for document parsing, splitting, retrieval, and chains
- ⚡ **DSPy** for LLM orchestration and inference
- 🗂️ **FAISS** for efficient vector search
- 📄 DOCX file ingestion for contextual Q&A

---

## 📁 Project Structure
```bash
├── data/
│ └── GST_Smart_Guide.docx # Sample document for RAG
├── main.py # Main pipeline implementation
├── .env # Environment variables (AWS profile)
└── README.md 

```

## 🚀 Features

- Loads `.docx` file using `UnstructuredWordDocumentLoader`
- Splits document into chunks
- Embeds chunks using **Amazon Titan Embeddings**
- Stores & searches chunks using **FAISS**
- Answers user queries using **Amazon Titan LLM** via **DSPy**
- Ensures fallback safety: "I don't know" for uncertain responses

---

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ManasChauhan28/DspyRAG.git
cd DspyRAG
```
2. **Create a Virtual Environment**:
```bash
python -m venv venv
venv\Scripts\activate
```
3. **Install Required Packages**:
```bash
pip install -r requirements.txt
```
4. **Configure `aws` profile**:
```bash
aws configure
```
4. **Run `main.py` file**
```bash
python main.py
```


