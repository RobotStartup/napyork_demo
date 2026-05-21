"""
MEWSFLOW AI — ORCHESTRATOR (orchestrator.py)
Multi-agent orchestration with visible routing for NapYork.
"""

import os
import asyncio
import logging
import uuid
from datetime import datetime

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

# Routing rules — keywords mapped to agents
ROUTING_RULES = {
    "reporting": {
        "keywords": ["report", "arrival", "departure", "revenue", "occupancy",
                      "summary", "analytics", "performance", "how many", "stats",
                      "today", "weekly", "monthly", "dashboard", "booking source",
                      "generate", "numbers"],
        "description": "Reports, analytics, arrivals, revenue, occupancy stats",
    },
    "ota_sync": {
        "keywords": ["sync", "ota", "expedia", "booking.com", "hostelworld",
                      "availability", "overbooking", "channel"],
        "description": "OTA sync across Expedia, Booking.com, Hostelworld",
    },
    "support": {
        "keywords": ["policy", "faq", "refund", "check-in", "what is", "how do",
                      "cancel policy", "amenities", "services", "pricing",
                      "cancellation", "checkout", "check out", "wifi", "shower",
                      "hours", "location", "where"],
        "description": "Guest support, policies, FAQ, knowledge base search",
    },
    "reservation": {
        "keywords": [],
        "description": "Reservation management, payments, card charging, cancellations",
    },
}

class AgentOrchestrator:
    def __init__(self):
        self.llm = None
        self.agents = {}
        self.rag_pipeline = None
        self.conversations = {}

    async def initialize(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.rag_pipeline = RAGPipeline()
        await self.rag_pipeline.initialize()
        self.agents["reservation"] = ReservationAgent(llm=self.llm)
        self.agents["support"] = CustomerSupportAgent(llm=self.llm, rag_pipeline=self.rag_pipeline)
        self.agents["ota_sync"] = OTASyncAgent(llm=self.llm)
        self.agents["reporting"] = ReportingAgent(llm=self.llm)
        logger.info(f"Orchestrator ready with {len(self.agents)} agents")

    def route(self, message: str, source: str = "dashboard") -> dict:
        """Determine which agent should handle this message. Returns routing info."""
        if source == "chatbot":
            return {
                "agent": "support",
                "reason": "Guest chatbot messages always route to Support Agent",
                "matched_keywords": [],
            }

        lower = message.lower()
        for agent_name in ["reporting", "ota_sync", "support"]:
            rules = ROUTING_RULES[agent_name]
            matched = [kw for kw in rules["keywords"] if kw in lower]
            if matched:
                return {
                    "agent": agent_name,
                    "reason": rules["description"],
                    "matched_keywords": matched,
                }

        return {
            "agent": "reservation",
            "reason": ROUTING_RULES["reservation"]["description"],
            "matched_keywords": ["(default — no other agent matched)"],
        }

    async def process_message(
        self,
        message: str,
        conversation_id: str = None,
        source: str = "dashboard",
    ) -> dict:
        run_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        history = self.conversations[conversation_id]

        # ── Route the message ──
        routing = self.route(message, source)
        agent_name = routing["agent"]

        # ── Call the agent ──
        try:
            agent = self.agents[agent_name]
            result = await agent.run(message=message, chat_history=history)
            response = result.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"Agent {agent_name} error: {e}")
            response = f"Error from {agent_name} agent: {e}"

        # ── Update memory ──
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response))
        if len(history) > 50:
            self.conversations[conversation_id] = history[-40:]

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "response": response,
            "agents_used": [agent_name + "_agent"],
            "routing": routing,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def shutdown(self):
        self.conversations.clear()
        logger.info("Orchestrator shut down")


async def main():
    print("\n═══ MewsFlow AI — Multi-Agent Orchestrator ═══\n")
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    print("\n✅ All agents online. Type 'quit' to exit.\n")
    conversation_id = str(uuid.uuid4())
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = await orchestrator.process_message(message=user_input, conversation_id=conversation_id)
        print(f"\nRouting: {result['routing']}")
        print(f"Agent: {result['agents_used']} ({result['elapsed_seconds']:.1f}s)")
        print(f"MewsFlow AI: {result['response']}\n")
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())