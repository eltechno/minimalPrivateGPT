# 🌐 minimalPrivateGPT 📚

Hello and welcome to `minimalPrivateGPT`! 😃 This is a trim and streamlined variant of the `privateGPT` project, specifically designed for processing PDF files. This tool enables you to extract information from your PDF documents offline, leveraging the power of Large Language Models (LLMs). Be assured, your data remains 💯% private, as no data is transmitted outside your local execution environment.

This project utilizes the robust technologies of LangChain, GPT4All, LlamaCpp, Chroma.

## 🛠️ Installation

To commence, clone the repository to your local machine:

```bash
git clone https://github.com/eltechno/minimalPrivateGPT.git
```

Then, navigate into the cloned repository:

```bash
cd minimalPrivateGPT
```
For Linux

apt-get install build-essential -y



## 🚀 Usage

On launching the application, put the PDF file into source directory and it forms a `db` folder to host the local vectorstore. Processing a PDF document typically takes around 20-30 seconds, dependent on the document size. You're not restricted on the number of documents you can ingest - they all aggregate in the local embeddings database.

To initiate a clean start, just delete the `db` folder.

*Note: Your data stays local during the ingestion process, but the first time you run the ingest script, an internet connection will be required to download the embeddings model.*

## 🔍 Query your PDF documents

Post the ingestion process, you can start querying your PDF documents locally. This ensures the privacy and security of your data, whilst providing you with immediate, offline access to your information.

## 🏞️ Environment Setup

To establish your environment, first install all the necessary prerequisites:

```bash
pip3 install -r requirements.txt
```

Then, download the LLM model and place it in a directory of your choice:

- LLM: The default model is `ggml-gpt4all-j-v1.3-groovy.bin`. If you have a preference for a different GPT4All-J compatible model, simply download it and reference it in your main file.

## 📜 License

This project is licensed under the GNU General Public License. For more information, refer to the [LICENSE.md](https://github.com/yourusername/minimalPrivateGPT/blob/main/LICENSE.md) file.
