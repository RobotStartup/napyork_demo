"""
MEWSFLOW AI — CUSTOMER SUPPORT AGENT (with RAG + demo mode)
"""
import os, logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from rag.pipeline import RAGPipeline

logger = logging.getLogger("mewsflow.agents.support")

PROMPT = """You are the NapYork Guest Support Agent.

About NapYork: Nap York partners with landlords, hotels, and property owners
to convert spaces into Sleep Stations. We serve aviation crew (pilots, flight
attendants, ground crew) and travelers who need quality rest between shifts
and flights. What began as a small, hourly sleep station in New York has now
evolved into a global rest network for those constantly on the move.

KNOWLEDGE BASE CONTEXT (from RAG vector search):
{rag_context}

Rules:
- Answer based on the knowledge base context above — it contains real NapYork policies
- Be warm, professional, and concise — you represent the NapYork brand
- For booking modifications, collect the reservation ID and note it for staff
- If the knowledge base doesn't cover it, give a helpful answer based on
  the NapYork context you know, and offer to connect with staff for specifics
- Never make up specific prices unless they appear in the knowledge base"""

class CustomerSupportAgent:
    def __init__(self, llm: ChatOpenAI, rag_pipeline: RAGPipeline):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            | llm
            | StrOutputParser()
        )

    async def run(self, message: str, chat_history: list = None) -> dict:
        rag_results = await self.rag_pipeline.search(message, top_k=3)
        if rag_results:
            rag_context = "\n\n".join(
                f"[{r['source']}]: {r['content']}" for r in rag_results
            )
        else:
            rag_context = "No specific documents found for this query."

        response = await self.chain.ainvoke({
            "input": message,
            "chat_history": chat_history or [],
            "rag_context": rag_context,
        })
        return {"output": response}

    def as_tool(self):
        return None