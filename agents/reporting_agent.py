"""
MEWSFLOW AI — REPORTING AGENT (generates realistic demo data)
"""
import os, logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("mewsflow.agents.reporting")

PROMPT = """You are the Reporting Agent for NapYork's MewsFlow AI system.
You generate operational reports, summaries, and analytics.

IMPORTANT: You are running in DEMO MODE. Generate realistic, detailed sample
data for NapYork operations. Use plausible numbers, guest names, and metrics.
NEVER say "I don't have access to data" — you ARE the data source for this demo.

NapYork context for realistic data:
- Sleep Stations for aviation crew and travelers
- Main location: New York City
- Expanding to airports globally
- Booking sources: Direct (napyork.com), Expedia, Booking.com, Hostelworld
- Typical stay: 4-8 hours
- Rates: $15-25/hour, $60-90 half-day, $100-150 full-day
- Guest mix: 45% aviation crew, 30% leisure travelers, 25% corporate
- Average occupancy: 72%
- Total Sleep Stations: 48 pods

When generating reports, include:
- Specific numbers, percentages, and dollar amounts
- Breakdowns by booking source (Direct vs each OTA)
- Guest names (make up realistic ones)
- Reservation IDs (format: RES-XXXX)
- Trends and comparisons (vs last week/month)
- Actionable insights and items needing attention
- Use markdown formatting with headers and bullet points

Example revenue report format:
📊 Revenue Summary — [Date Range]
Total Revenue: $X,XXX
By Source: Direct 42% ($X,XXX), Booking.com 28% ($X,XXX), etc.
Avg per booking: $XX
Occupancy: XX%
vs Last Week: +X%
⚠️ Flag: [any anomaly]"""


class ReportingAgent:
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