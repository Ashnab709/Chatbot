# 📄 PDF Question Answering Chatbot 🤖

An interactive **Streamlit-based chatbot** that lets you upload a PDF and ask natural language questions about its content.  
It uses **LangChain**, **Hugging Face Sentence Transformers**, **FAISS**, and a **local Mistral-7B-Instruct** model to answer queries with full offline processing.

---

## 🚀 Features

✅ **Upload & Parse PDFs** – Extracts text from all pages  
✅ **Smart Chunking** – Splits large text into manageable chunks  
✅ **Semantic Search** – Embeddings via `all-MiniLM-L6-v2`  
✅ **Vector Store** – FAISS for fast similarity search  
✅ **Local LLM Answering** – Mistral-7B-Instruct for contextual Q&A  
✅ **Privacy Friendly** – All processing is local, no external API calls  

---

## 🖼 Demo Preview
![Demo Screenshot](questions.png)

---

## 🛠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **Backend**: Python  
- **NLP Framework**: [LangChain](https://www.langchain.com/)  
- **Embedding Model**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **Vector Database**: [FAISS](https://faiss.ai/)  
- **LLM**: [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)  

---

## 📦 Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run chatbot.py
🎯 Usage
Run the app with streamlit run chatbot.py

Upload your PDF using the sidebar

Ask a question about the content in the input box

Get an answer from the local Mistral-7B-Instruct model

📜 Requirements
Python 3.9+

Sufficient RAM & GPU (for local model)

Models downloaded from Hugging Face (first run may take time)

🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you'd like to change.

