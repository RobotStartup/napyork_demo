"""
═══════════════════════════════════════════════════════════════════════════════
MEWSFLOW AI — RAG PIPELINE (rag/pipeline.py)
═══════════════════════════════════════════════════════════════════════════════

RAG (Retrieval-Augmented Generation) SECTION — FULL EXPLANATION:

THE PROBLEM:
  LLMs don't know NapYork's specific policies. If you ask "what's the
  cancellation policy for group bookings?", the LLM will make something up.
  That's called hallucination, and it's dangerous for a business.

THE SOLUTION — RAG:
  Store your real documents in a searchable database. Before the LLM answers,
  RETRIEVE the relevant documents and FEED them to the LLM as context.
  Now it answers from YOUR data, not its imagination.

VECTOR EMBEDDINGS SECTION — HOW THE SEARCH WORKS:

  Traditional search (keyword matching):
    Query: "cancel group reservation"
    Matches: documents containing the exact words "cancel", "group", "reservation"
    Problem: misses documents that say "cancellation policy for block bookings"

  Semantic search (vector embeddings):
    Query: "cancel group reservation" → [0.12, -0.34, 0.56, ...] (384 numbers)
    Document: "cancellation policy for block bookings" → [0.11, -0.33, 0.55, ...]
    These vectors are SIMILAR even though the words are different!
    The search finds documents by MEANING, not keywords.

  How vectors are created:
    1. An embedding model (all-MiniLM-L6-v2) reads the text
    2. It outputs a list of 384 numbers that represent the meaning
    3. Similar meanings → similar numbers (measured by cosine similarity)

  Vector Database (Chroma for dev, Pinecone for production):
    - Stores thousands of document vectors
    - When you search, it converts your query to a vector
    - Finds the k nearest vectors (most similar documents)
    - Returns those documents as context for the LLM

THE FULL RAG FLOW:
  1. INGEST: Load NapYork SOPs, FAQs, policies → split into chunks → embed → store
  2. SEARCH: User asks a question → embed the question → find similar chunks
  3. AUGMENT: Feed the matching chunks to the LLM in the prompt
  4. GENERATE: LLM answers using the real documents as evidence

TEXT SPLITTING — WHY WE CHUNK DOCUMENTS:
  A full SOP document might be 5 pages long. If we embed the whole thing
  as one vector, it's too broad — the meaning gets diluted.

  Instead, we split it into overlapping chunks of ~1000 characters.
  Each chunk gets its own vector. When searching, we find the SPECIFIC
  paragraph that answers the question, not the entire document.

  chunk_overlap=200 means consecutive chunks share 200 characters.
  This prevents cutting a sentence in half at the boundary.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger("mewsflow.rag")

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_store")


class RAGPipeline:
    """
    RAG pipeline for NapYork's knowledge base.

    Uses:
      - HuggingFace embeddings (free, runs locally, no API key needed)
      - Chroma vector store (local for development)

    For production, swap to:
      - OpenAI embeddings (better quality, costs ~$0.0001 per 1K tokens)
      - Pinecone vector store (managed, scalable, fast)
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store = None

        # ── Text Splitter Configuration ──
        # chunk_size=1000: each chunk is ~1000 characters (~200 words)
        # chunk_overlap=200: chunks share 200 chars to avoid broken sentences
        # separators: tries to split at paragraphs first, then sentences, etc.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def initialize(self):
        """
        Initialize the embedding model and vector store.
        Called once when the server starts.
        """

        # ── Create the Embedding Model ──
        # This model converts text → vectors (384-dimensional arrays)
        # all-MiniLM-L6-v2 is free, runs on CPU, and is surprisingly good
        # For production: use text-embedding-3-small from OpenAI ($0.02/1M tokens)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # ── Create the Vector Store ──
        # Chroma stores vectors locally in a directory
        # For production: use Pinecone (cloud-hosted, faster at scale)
        #
        # To swap to Pinecone, replace this block with:
        #   from langchain_pinecone import PineconeVectorStore
        #   self.vector_store = PineconeVectorStore(
        #       index_name="napyork-knowledge",
        #       embedding=self.embeddings,
        #       pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        #   )
        self.vector_store = Chroma(
            collection_name="napyork_knowledge",
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_DIR,
        )

        # Load default knowledge base if the vector store is empty
        collection = self.vector_store._collection
        if collection.count() == 0:
            await self._load_napyork_knowledge()
            logger.info("Loaded NapYork knowledge base into vector store")

        logger.info(f"RAG pipeline ready — {collection.count()} document chunks in store")

    # ───────────────────────────────────────────────────────────────────────
    # DEFAULT NAPYORK KNOWLEDGE BASE
    # ───────────────────────────────────────────────────────────────────────
    #
    # These are the documents the support agent searches.
    # In production, you'd load these from files:
    #   await self.ingest_directory("./knowledge_base/")
    #
    # Each Document has:
    #   - page_content: the actual text
    #   - metadata: source file, document type (for filtering and citation)
    # ───────────────────────────────────────────────────────────────────────

    async def _load_napyork_knowledge(self):
        """Load NapYork's default knowledge documents."""
        documents = [
            Document(
                page_content="""
NapYork Cancellation Policy

Free cancellation up to 24 hours before check-in time.
Cancellations within 24 hours of check-in: First night charge applies.
No-shows: Full stay amount will be charged.

For group bookings (5+ rooms):
- Free cancellation up to 7 days before check-in
- 50% charge for cancellations 3-7 days before check-in
- Full charge for cancellations less than 3 days before check-in

OTA bookings follow the respective OTA's cancellation policy unless
NapYork's policy is more favorable to the guest.
                """.strip(),
                metadata={"source": "cancellation_policy.md", "type": "policy"},
            ),

            Document(
                page_content="""
NapYork Check-in and Check-out Procedures

Check-in: Available 24/7 via self-service kiosk or front desk.
Guest must present valid ID matching the reservation name.
Payment verification required at check-in.
Sleep Station assignment is automated based on availability.

Check-out: Simply leave the Sleep Station.
Automatic checkout at the end of the booked period.
Extensions can be requested via the NapYork app or front desk.
Late checkout fee: $15 per hour after the 15-minute grace period.

For aviation crew members:
Priority check-in available with airline crew ID.
Flexible checkout for delayed flights (notify front desk).
Crew rest compliance documentation available on request.
                """.strip(),
                metadata={"source": "checkin_procedures.md", "type": "sop"},
            ),

            Document(
                page_content="""
NapYork Payment Processing Standard Operating Procedure

1. Card on File Verification
   All reservations require a valid card on file.
   Cards are verified (not charged) at time of booking.
   Pre-authorization hold: $1.00 verification charge, released within 24 hours.

2. Payment Collection
   Direct bookings: Charged at check-in or as specified in rate plan.
   OTA bookings: Follow OTA payment terms (virtual card or guest card).
   Failed payments: See Payment Recovery Workflow.

3. Payment Recovery Workflow
   a. First attempt: Retry card charge.
   b. If failed and OTA booking:
      Mark card as invalid on OTA portal.
      Wait 24 hours for OTA to provide new card.
      If no new card: Cancel reservation on both OTA and Mews.
   c. If failed and direct booking:
      Send payment request link to guest email.
      Wait 24 hours for payment.
      If no payment: Cancel reservation on Mews.
   d. All actions logged in Mews task system.

4. Refund Processing
   Refunds processed within 5-10 business days.
   Refunds go back to original payment method.
   Manager approval required for refunds over $200.
                """.strip(),
                metadata={"source": "payment_sop.md", "type": "sop"},
            ),

            Document(
                page_content="""
NapYork Services and Amenities

Sleep Stations:
Private, soundproofed rest pods with premium bedding and climate control.
Hourly, half-day, and full-day rates available.
Blackout curtains and white noise machines included.

Available at all locations:
Free high-speed WiFi, USB and wireless charging.
Fresh linens and towels, shower facilities, luggage storage.
24/7 security.

Aviation Crew Packages:
Discounted crew rates with valid airline ID.
Quick turnaround cleaning between shifts.
Wake-up call service, quiet zone guarantees during rest periods.
FAA and EASA rest compliance support.

Locations:
New York City (original location).
Expanding to airports and city centers globally.
Partners with landlords, hotels, and property owners to convert spaces.
                """.strip(),
                metadata={"source": "services_amenities.md", "type": "info"},
            ),

            Document(
                page_content="""
NapYork Frequently Asked Questions

What is NapYork?
NapYork is a global rest network providing Sleep Stations for travelers,
aviation crew, and anyone who needs quality rest. We partner with landlords,
hotels, and property owners to convert existing spaces into Sleep Stations.

How do I book a Sleep Station?
Book directly at napyork.com, through the NapYork app, or via OTAs like
Booking.com, Expedia, and Hostelworld.

What are your rates?
Rates vary by location and duration. Hourly rates start from $15 per hour.
Check napyork.com for current pricing at your preferred location.

Can I extend my stay?
Yes. Request an extension via the NapYork app or at the front desk.
Extensions are subject to availability.

Do you offer corporate or airline rates?
Yes. We have special rates for airline crew and corporate partners.
Contact partnerships@napyork.com for details.

What is the minimum stay?
Minimum stay is 1 hour at most locations.
                """.strip(),
                metadata={"source": "faq.md", "type": "faq"},
            ),
        ]

        # ── Split documents into chunks and add to vector store ──
        # Each document gets split into ~1000 char chunks
        # Each chunk gets embedded (converted to a 384-dim vector)
        # Each vector gets stored in Chroma for fast similarity search
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        logger.info(f"Embedded and stored {len(chunks)} knowledge chunks")

    # ───────────────────────────────────────────────────────────────────────
    # SEARCH — THE CORE RAG OPERATION
    # ───────────────────────────────────────────────────────────────────────
    #
    # When the support agent calls search_knowledge_base("cancellation policy"),
    # it ends up here. This method:
    #   1. Takes the query string
    #   2. The vector store embeds it (converts to vector)
    #   3. Finds the top_k most similar document chunks
    #   4. Returns them with similarity scores
    #
    # The similarity score ranges from 0 (identical) to 2 (opposite).
    # Lower = more relevant.
    # ───────────────────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over the NapYork knowledge base.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            List of dicts with: content, source, type, score
        """
        results = self.vector_store.similarity_search_with_score(
            query, k=top_k,
        )

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "score": float(score),
            }
            for doc, score in results
        ]

    async def add_document(self, content: str, metadata: dict) -> int:
        """
        Add a new document to the knowledge base.

        Use this to add new SOPs, policy updates, etc. without restarting.
        The document gets chunked, embedded, and stored immediately.
        """
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        self.vector_store.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks from new document")
        return len(chunks)
