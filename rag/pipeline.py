"""
RAG Pipeline — Lightweight version for Streamlit Cloud deployment.
Uses simple text matching instead of Chroma + sentence-transformers.
Same concept (search docs before answering), just no heavy dependencies.
"""
import logging
from difflib import SequenceMatcher

logger = logging.getLogger("mewsflow.rag")


class RAGPipeline:
    def __init__(self):
        self.documents = []

    async def initialize(self):
        self.documents = [
            {
                "content": "NapYork Cancellation Policy: Free cancellation up to 24 hours before check-in. Cancellations within 24 hours: first night charge applies. No-shows: full stay charged. Group bookings (5+ rooms): free cancellation up to 7 days before check-in, 50% charge 3-7 days before, full charge less than 3 days before. OTA bookings follow the OTA's policy unless NapYork's is more favorable.",
                "source": "cancellation_policy.md",
                "type": "policy",
            },
            {
                "content": "NapYork Check-in/Check-out: Check-in available 24/7 via self-service kiosk or front desk. Guest must present valid ID matching reservation. Payment verification required. Sleep Station assignment is automated. Check-out: simply leave. Automatic checkout at end of booked period. Extensions via app or front desk, subject to availability. Late checkout: $15/hour after 15-minute grace period. Aviation crew: priority check-in with airline ID, flexible checkout for delayed flights.",
                "source": "checkin_procedures.md",
                "type": "sop",
            },
            {
                "content": "NapYork Payment Processing SOP: All reservations require valid card on file. Pre-auth hold: $1.00 verification released in 24h. Direct bookings charged at check-in. OTA bookings follow OTA payment terms. Payment Recovery: 1) Retry card charge. 2) OTA booking failed: mark card invalid on OTA, wait 24h for new card, cancel if none. 3) Direct booking failed: send payment request link, wait 24h, cancel if unpaid. 4) All actions logged. Refunds: 5-10 business days, original payment method, manager approval over $200.",
                "source": "payment_sop.md",
                "type": "sop",
            },
            {
                "content": "NapYork Services: Private soundproofed Sleep Stations with premium bedding, climate control, blackout curtains, white noise. Hourly, half-day, and full-day rates. All locations: free WiFi, USB/wireless charging, fresh linens, showers, luggage storage, 24/7 security. Aviation Crew Packages: discounted rates with airline ID, quick turnaround cleaning, wake-up calls, quiet zone guarantees, FAA/EASA rest compliance support. Locations: NYC original, expanding to airports globally.",
                "source": "services_amenities.md",
                "type": "info",
            },
            {
                "content": "NapYork FAQ: NapYork is a global rest network providing Sleep Stations for travelers, aviation crew, and anyone needing quality rest. Book at napyork.com, the app, or via Booking.com, Expedia, Hostelworld. Rates vary by location, hourly from $15. Extensions subject to availability. Corporate and airline rates available — contact partnerships@napyork.com. Minimum stay: 1 hour.",
                "source": "faq.md",
                "type": "faq",
            },
        ]
        logger.info(f"RAG pipeline ready — {len(self.documents)} documents loaded")

    async def search(self, query: str, top_k: int = 3) -> list:
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            content_lower = doc["content"].lower()
            # Score by keyword overlap + sequence matching
            keywords = query_lower.split()
            keyword_hits = sum(1 for kw in keywords if kw in content_lower)
            seq_score = SequenceMatcher(None, query_lower, content_lower[:200]).ratio()
            score = keyword_hits + seq_score
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": doc["content"],
                "source": doc["source"],
                "type": doc["type"],
                "score": score,
            }
            for score, doc in scored[:top_k]
        ]

    async def add_document(self, content: str, metadata: dict) -> int:
        self.documents.append({"content": content, **metadata})
        return 1