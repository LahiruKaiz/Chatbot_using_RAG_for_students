import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriver, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from a .env file
load_dotenv()

# Define the current directory and paths for text files and the Chroma database
current_dir = current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, "text_files")
db_dir = os.path.join(current_dir, "chroma_db")


text_files = []

# Iterate through each text file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            text_files.append(Document(page_content=text, metadata={"source": filename.split('.')[0], "source_path": file_path}))

# Initialize the text splitter to break documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=300)
# Split the documents into smaller chunks
docs = text_splitter.split_documents(text_files)

Embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Store the documents and their embeddings in the Chroma database
Chroma.from_documents(docs, Embeddings, persist_directory=db_dir)

# Initialize the Chroma database for retrieval
db = Chroma(persist_directory=db_dir, embedding_function=Embeddings)
# Create a retriever that fetches similar documents based on user queries
retriever = db.as_retriever(SearchType="similarity", SearchKwargs={"k": 2})

llm = ChatOpenAI(model_name="gpt-4o")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history,"
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for the contextualization process
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever that uses the LLM and the contextualization prompt
history_aware_retriever = create_history_aware_retriver(llm, retriever, contextualize_q_prompt)


qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know."
    "\n\n"
    "{context}"
)

# Create a prompt template for the question-answering process
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain for answering questions using the retrieved context
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine the retrieval and question-answering chains into a single RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Load a prompt for the React agent from the Langchain hub
react_docstore_prompt = hub.pull("hwchase17/react")

# Define tools for the agent, including the question-answering function
tools = [
    Tool(
        name = "Answer Question",
        func = lambda input, **kwargs: rag_chain.invoke({"input": input, "chat_history":kwargs.get("chat_history", [])}),
        description = "Useful for when you need to answering questions about the context."
    )
]

# Create a React agent that utilizes the defined tools
agent = create_react_agent(
    llm= llm,
    tools= tools,
    prompt= react_docstore_prompt,
)

# Initialize the agent executor to manage the interaction with the user
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    handle_parsing_errors = True,
)

# Main interaction loop for the user to ask questions

if __name__ == "__main__":
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit": # Check for exit command
            break
        
        # Invoke the agent executor with the user input and chat history
        response = agent_executor.invoke({"input": query, "chat_history": chat_history})
        print(f"Assistant: {response['output']}")
    
         # Update the chat history with the latest interaction
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response['output']))
    