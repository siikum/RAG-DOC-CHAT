# RAG-DocsChatBot
A chatbot built using **LangChain** and **Gemini** that answers user queries from PDFs. This implementation focuses on document-based question answering and can respond to queries about text and tables in a PDF.

## Functionalities



https://github.com/user-attachments/assets/7bf2ee22-3d15-4775-87cc-0b6b0014eb97


### Document QA part

This is an implementation of a **document retrieval type chatbot** that can answer questions based on information available in a PDF file.  

Currently, the sample PDF used is [2022 Q3 AAPL Report](https://github.com/docugami/KG-RAG-datasets/blob/main/sec-10-q/data/v1/docs/2022%20Q3%20AAPL.pdf).


## Installation

#### Clone the repository into a folder:

    git clone https://github.com/<your-username>/RAG-DocsChat.git

#### Navigate into the directory:

    cd RAG-DocsChat

#### Install dependencies from the requirements.txt file:

    pip install -r requirements.txt
You also need a Gemini API key to run the generative AI model.
* You can create a free Gemini API key for testing.
* Copy your API key and save it in a file called API_KEY.txt in the project root:

    echo Your_API_KEY > API_KEY.txt

## Execution
This project uses Streamlit for an interactive chatbot interface. Run the frontend with:

    streamlit run main.py
<img width="571" height="98" alt="image" src="https://github.com/user-attachments/assets/b4c4a13a-b364-4455-820c-f7b9063acbfe" />

* Type a question about the PDF in the input box
* The system returns the answer based on the PDF content

### Programmatic access
You can also use the query() function in Python:

    from DocsChat.DocsChat import query

    answer = query("What was Apple’s net income for the three months ended June 25, 2022?")
    print(answer)
