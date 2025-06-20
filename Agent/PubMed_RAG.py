from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek 
from langchain_huggingface import HuggingFaceEmbeddings
from Bio import Entrez
import os

load_dotenv()

class PubmedRAGState(TypedDict):
    query: str
    context: List[Document]
    answer: str
    chat_history: List[AnyMessage]


def process_pubmed_articles(query: str, max_papers=20) -> List[Document]:
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_papers)
    pmids = Entrez.read(search_handle)["IdList"]
    search_handle.close()

    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    docs = []
    for record in records["PubmedArticle"]:
        article = record["MedlineCitation"]["Article"]
        title = article.get("ArticleTitle", "No title")
        abstract_data = article.get("Abstract", {})
        abstracts = abstract_data.get("AbstractText", [])
        if isinstance(abstracts, list):
            abstract_text = "\n".join([a if isinstance(a, str) else a.get("_", "") for a in abstracts])
        else:
            abstract_text = abstracts if isinstance(abstracts, str) else "No abstract available."

        docs.append(Document(
            page_content=f"Title: {title}\n\n{abstract_text}",
            metadata={"title": title}
        ))
    return docs


class PubmedRAGTool:
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.rag_pipeline = self.create_pubmed_rag_pipeline()

    def create_pubmed_rag_pipeline(self):
        llm = ChatDeepSeek(model=self.model)

        def retrieve(state: PubmedRAGState):
            return {"context": state["context"]}

        def generate(state: PubmedRAGState):
            context_content = "\n\n".join(doc.page_content for doc in state["context"])
            prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder("chat_history"),
                ("user", "You are a biomedical research assistant. Based on the provided PubMed context, generate an academic answer."),
                ("user", "CONTEXT:\n{context_content}"),
                ("user", "USER QUESTION:\n{query}"),
                ("user", "Please provide a detailed and specific analysis, especially focusing on the functional impact of the variants.")
            ])
            messages = prompt.invoke({
                "query": state["query"],
                "chat_history": state["chat_history"],
                "context_content": context_content
            })
            response = llm.invoke(messages)
            return {"answer": response.content}

        builder = StateGraph(PubmedRAGState)
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def run(self, query: str, chat_history: List[AnyMessage] = None):
        if chat_history is None:
            chat_history = []

        full_query = f"TP53 AND {query}"
        print(f"Searching PubMed for: {full_query}")
        context_docs = process_pubmed_articles(full_query)

        response = self.rag_pipeline.invoke({
            "query": query,
            "chat_history": chat_history,
            "context": context_docs,
            "answer": ""
        })
        return response["answer"]

pubmed_rag = PubmedRAGTool()

@tool
def pubmed_rag_agent(state: Annotated[Dict, InjectedState], query: str) -> str:
    """Tool that provides literature-based answers from PubMed papers using RAG and DeepSeek.
    
    Args:
        query: Biomedical or scientific question related to TP53 variants

    Returns:
        str: Answer based on retrieved PubMed content
    """
    chat_history = []  # or: state["messages"][:-1]
    return pubmed_rag.run(query, chat_history)
