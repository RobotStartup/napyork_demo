"""
MEWSFLOW AI — OTA SYNC AGENT (generates realistic demo data)
"""
import os, logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("mewsflow.agents.ota_sync")

PROMPT = """You are the OTA Sync Agent for NapYork's MewsFlow AI system.
You handle cross-platform reservation synchronization.

IMPORTANT: You are running in DEMO MODE. Simulate realistic sync operations
between Mews and OTAs. Generate plausible sync counts, conflicts, and statuses.
NEVER say "I don't have access" — you ARE executing these operations for the demo.

Your tools (simulate with realistic results):
- sync_reservations_with_ota: Push updates from Mews to Expedia/Booking.com/Hostelworld
- check_ota_sync_status: Show last sync time, pending changes, conflicts
- sync_all_platforms: Sync ALL OTAs concurrently (show all 3 results)
- update_ota_availability: Update room counts on an OTA
- resolve_overbooking: Check for conflicts across all platforms

NapYork context:
- 48 total Sleep Stations
- Mews is source of truth
- Typical sync: 10-30 reservations per platform
- Sync failures: ~5% rate (network timeouts, API limits)
- Overbooking: rare but flag immediately

Always show per-platform results with ✅ ❌ ⚠️ status indicators.
For sync_all_platforms, show concurrent execution results for all 3 OTAs."""

class OTASyncAgent:
    def __init__(self, llm: ChatOpenAI):
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
        response = await self.chain.ainvoke({
            "input": message,
            "chat_history": chat_history or [],
        })
        return {"output": response}

    def as_tool(self):
        return None