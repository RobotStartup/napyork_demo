"""
MEWSFLOW AI — ORCHESTRATOR (orchestrator.py)

Multi-agent orchestration for NapYork hotel operations.
Routes messages to specialized agents: Reservation, Support, OTA Sync, Reporting.
"""

import os
import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from agents.reservation_agent import ReservationAgent
from agents.support_agent import CustomerSupportAgent
from agents.ota_sync_agent import OTASyncAgent
from agents.reporting_agent import ReportingAgent
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mewsflow.orchestrator")


class AgentOrchestrator:
    """
    Multi-agent orchestrator with keyword-based routing.
    Each child agent is a specialized LLM chain.
    """

    def __init__(self):
        self.llm = None
        self.agents = {}
        self.rag_pipeline = None
        self.conversations = {}

    async def initialize(self):
        """Initialize LLM, RAG pipeline, and all child agents."""

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Initialize RAG pipeline (vector embeddings for knowledge search)
        self.rag_pipeline = RAGPipeline()
        await self.rag_pipeline.initialize()
        logger.info("RAG pipeline initialized")

        # Create specialized agents
        self.agents["reservation"] = ReservationAgent(llm=self.llm)
        self.agents["support"] = CustomerSupportAgent(
            llm=self.llm,
            rag_pipeline=self.rag_pipeline,
        )
        self.agents["ota_sync"] = OTASyncAgent(llm=self.llm)
        self.agents["reporting"] = ReportingAgent(llm=self.llm)

        logger.info(f"Orchestrator ready with {len(self.agents)} agents")

    async def process_message(
        self,
        message: str,
        conversation_id: str = None,
        source: str = "dashboard",
    ) -> dict:
        """Route a message to the right agent and return the response."""

        run_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        history = self.conversations[conversation_id]

        # ── Route to the right agent ──
        lower = message.lower()
        if source == "chatbot":
            agent_name = "support"
        elif any(w in lower for w in [
            "policy", "faq", "refund", "check-in", "what is", "how do",
            "cancel policy", "amenities", "services", "book", "pricing",
            "cancellation", "checkout", "check out", "wifi", "shower",
        ]):
            agent_name = "support"
        elif any(w in lower for w in [
            "report", "arrival", "departure", "revenue", "occupancy",
            "summary", "analytics", "performance", "how many",
        ]):
            agent_name = "reporting"
        elif any(w in lower for w in [
            "sync", "ota", "expedia", "booking.com", "hostelworld",
            "availability", "overbooking",
        ]):
            agent_name = "ota_sync"
        else:
            agent_name = "reservation"

        # ── Call the agent ──
        try:
            agent = self.agents[agent_name]
            result = await agent.run(message=message, chat_history=history)
            response = result.get("output", "No response generated.")
            agents_used = [agent_name + "_agent"]
        except Exception as e:
            logger.error(f"Agent {agent_name} error: {e}")
            response = f"Error from {agent_name} agent: {e}"
            agents_used = [agent_name + "_agent"]

        # ── Update conversation memory ──
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response))
        if len(history) > 50:
            self.conversations[conversation_id] = history[-40:]

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "response": response,
            "agents_used": agents_used,
            "intermediate_steps": [],
            "elapsed_seconds": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def shutdown(self):
        self.conversations.clear()
        logger.info("Orchestrator shut down")


async def main():
    """Interactive test mode."""
    print("\n═══ MewsFlow AI — Multi-Agent Orchestrator ═══\n")
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    print("\n✅ All agents online. Type 'quit' to exit.\n")

    conversation_id = str(uuid.uuid4())
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = await orchestrator.process_message(
            message=user_input, conversation_id=conversation_id,
        )
        print(f"\nAgent: {result['agents_used']} ({result['elapsed_seconds']:.1f}s)")
        print(f"MewsFlow AI: {result['response']}\n")

    await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())