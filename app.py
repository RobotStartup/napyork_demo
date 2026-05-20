"""
MewsFlow AI — Streamlit MVP Dashboard
"""

import streamlit as st
import asyncio
import uuid
import time
import os
import traceback
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="MewsFlow AI — NapYork Operations",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0c10; }
    [data-testid="stSidebar"] { background-color: #12151c; border-right: 1px solid #1e2330; }
    [data-testid="stChatMessage"] { background-color: #12151c; border: 1px solid #1e2330; border-radius: 12px; padding: 12px; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .cost-bar { height: 32px; border-radius: 8px; display: flex; align-items: center; padding-left: 12px; font-weight: 600; font-size: 14px; }
    .cost-human { background: rgba(239,68,68,0.15); color: #ef4444; width: 100%; }
    .cost-ai { background: rgba(34,197,94,0.15); color: #22c55e; width: 8%; min-width: 140px; }
</style>
""", unsafe_allow_html=True)


# ── Initialize Orchestrator ──
@st.cache_resource(show_spinner="Initializing MewsFlow AI agents...")
def get_orchestrator():
    from agents.orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(orchestrator.initialize())
    loop.close()
    return orchestrator


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to **MewsFlow AI** — NapYork's intelligent operations assistant.\n\n"
                "I coordinate a team of specialized AI agents to handle hotel operations. "
                "I can help with:\n"
                "- 💳 **Reservation & payments** — scan tasks, charge cards, handle failures\n"
                "- 💬 **Guest support** — answer questions using our knowledge base\n"
                "- 🔄 **OTA sync** — keep Mews, Expedia, Booking.com, Hostelworld in sync\n"
                "- 📊 **Reports** — arrivals, revenue, occupancy, agent performance\n\n"
                "What would you like to do?"
            ),
            "agents": [],
            "elapsed": None,
        }
    ]

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "action_logs" not in st.session_state:
    st.session_state.action_logs = []
if "agent_stats" not in st.session_state:
    st.session_state.agent_stats = {
        "Reservation": {"runs": 0, "total_time": 0, "errors": 0},
        "Support": {"runs": 0, "total_time": 0, "errors": 0},
        "OTA Sync": {"runs": 0, "total_time": 0, "errors": 0},
        "Reporting": {"runs": 0, "total_time": 0, "errors": 0},
    }
if "total_tokens_estimate" not in st.session_state:
    st.session_state.total_tokens_estimate = 0


# ── Sidebar ──
agents_config = [
    ("Reservation", "💳", "Payment recovery, card charging, cancellations"),
    ("Support", "💬", "Guest FAQ, policies, RAG knowledge search"),
    ("OTA Sync", "🔄", "Expedia, Booking.com, Hostelworld sync"),
    ("Reporting", "📊", "Arrivals, revenue, occupancy reports"),
]

with st.sidebar:
    st.image("https://napyork.com/cdn/shop/files/Untitled_design_36.png?v=1717701226", width=180)
    st.markdown("### MewsFlow AI")
    st.caption("Multi-Agent Operations Dashboard")
    st.divider()

    st.markdown("#### 🤖 Agent Status")
    for name, icon, desc in agents_config:
        stats = st.session_state.agent_stats[name]
        runs = stats["runs"]
        avg_time = (stats["total_time"] / runs) if runs > 0 else 0
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{icon} **{name}**")
            st.caption(f"{runs} runs · {avg_time:.1f}s avg")
        with col2:
            st.markdown("🟢")

    st.divider()
    st.markdown("#### ⚡ Quick Actions")
    if st.button("📋 Scan Open Tasks", use_container_width=True):
        st.session_state.quick_action = "Scan all open tasks and check for pending balances"
    if st.button("📊 Today's Arrivals", use_container_width=True):
        st.session_state.quick_action = "Generate today's arrivals report"
    if st.button("🔄 Sync All OTAs", use_container_width=True):
        st.session_state.quick_action = "Sync all reservations across Expedia, Booking.com, and Hostelworld"
    if st.button("💰 Agent Performance", use_container_width=True):
        st.session_state.quick_action = "Show agent performance metrics and API costs"

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.conversation_id[:8]}...`")
    est_cost = st.session_state.total_tokens_estimate * 0.0000015
    st.caption(f"Est. cost this session: ${est_cost:.4f}")


# ── Tabs ──
tab_chat, tab_logs, tab_analytics = st.tabs(["💬 Chat", "📋 Action Logs", "📊 Analytics"])


# ── Chat Tab ──
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏨" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("agents"):
                st.caption(f"Agents: {' '.join(f'`{a}`' for a in msg['agents'])}")
            if msg.get("elapsed"):
                st.caption(f"⏱️ {msg['elapsed']:.1f}s")

    if len(st.session_state.messages) <= 2:
        st.markdown("---")
        st.markdown("**Try asking:**")
        suggested = [
            "Show all VIP guests checking in tomorrow and send housekeeping prep requests",
            "Find reservations with failed payments and start recovery",
            "What is the refund policy for group bookings?",
            "Generate this week's revenue summary by booking source",
            "Scan all open tasks and charge pending balances",
        ]
        cols = st.columns(2)
        for i, prompt in enumerate(suggested):
            with cols[i % 2]:
                if st.button(f"💡 {prompt}", key=f"suggest_{i}", use_container_width=True):
                    st.session_state.quick_action = prompt

    user_input = st.chat_input("Ask MewsFlow AI anything about hotel operations...")

    if "quick_action" in st.session_state:
        user_input = st.session_state.quick_action
        del st.session_state.quick_action

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🏨"):
            status = st.status("Processing your request...", expanded=True)

            lower = user_input.lower()
            if any(w in lower for w in ["policy", "faq", "refund", "check-in", "what is", "how do", "cancel policy"]):
                likely_agent = "Support"
            elif any(w in lower for w in ["report", "arrival", "departure", "revenue", "occupancy", "summary", "analytics", "performance"]):
                likely_agent = "Reporting"
            elif any(w in lower for w in ["sync", "ota", "expedia", "booking.com", "hostelworld", "availability"]):
                likely_agent = "OTA Sync"
            else:
                likely_agent = "Reservation"

            status.update(label=f"Routing to {likely_agent} Agent...", state="running")
            start_time = time.time()

            try:
                orchestrator = get_orchestrator()
                result = run_async(
                    orchestrator.process_message(
                        message=user_input,
                        conversation_id=st.session_state.conversation_id,
                        source="dashboard",
                    )
                )

                elapsed = time.time() - start_time
                response_text = result.get("response", "No response generated.")
                agents_used = result.get("agents_used", [likely_agent.lower() + "_agent"])
                token_estimate = (len(user_input) + len(response_text)) // 4
                st.session_state.total_tokens_estimate += token_estimate
                status.update(label="✅ Complete", state="complete")

            except Exception as e:
                # ══════════════════════════════════════════════════
                # DEBUGGING: Print full traceback to terminal
                # Check your PowerShell window for the exact error
                # ══════════════════════════════════════════════════
                print("\n" + "=" * 60)
                print("MEWSFLOW ERROR — Full traceback:")
                print("=" * 60)
                traceback.print_exc()
                print("=" * 60 + "\n")

                elapsed = time.time() - start_time
                response_text = f"I encountered an error processing your request: `{str(e)}`\n\nPlease try again or rephrase your question."
                agents_used = [likely_agent.lower() + "_agent"]
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")

            st.markdown(response_text)
            if agents_used:
                st.caption(f"Agents: {' '.join(f'`{a}`' for a in agents_used)} · ⏱️ {elapsed:.1f}s")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "agents": agents_used,
                "elapsed": elapsed,
            })

            stats = st.session_state.agent_stats[likely_agent]
            stats["runs"] += 1
            stats["total_time"] += elapsed

            st.session_state.action_logs.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "agent": likely_agent,
                "action": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                "status": "error" if "error" in response_text.lower() else "success",
                "elapsed": f"{elapsed:.1f}s",
            })

        st.rerun()


# ── Logs Tab ──
with tab_logs:
    st.markdown("### Action Logs")
    st.caption("Every agent action is logged for full auditability.")
    if not st.session_state.action_logs:
        st.info("No actions yet. Start a conversation in the Chat tab to see logs here.")
    else:
        filter_agent = st.selectbox("Filter by agent", ["All", "Reservation", "Support", "OTA Sync", "Reporting"])
        for log in st.session_state.action_logs:
            if filter_agent != "All" and log["agent"] != filter_agent:
                continue
            status_emoji = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(log["status"], "⚪")
            col1, col2, col3, col4 = st.columns([1, 1, 4, 1])
            with col1:
                st.code(log["time"], language=None)
            with col2:
                st.markdown(f"**{log['agent']}**")
            with col3:
                st.markdown(f"{status_emoji} {log['action']}")
            with col4:
                st.caption(log["elapsed"])


# ── Analytics Tab ──
with tab_analytics:
    st.markdown("### LLM Observability & Analytics")

    total_runs = sum(s["runs"] for s in st.session_state.agent_stats.values())
    total_time = sum(s["total_time"] for s in st.session_state.agent_stats.values())
    avg_time = (total_time / total_runs) if total_runs > 0 else 0
    est_cost = st.session_state.total_tokens_estimate * 0.0000015

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Agent Runs", f"{total_runs}")
    col2.metric("Avg Response Time", f"{avg_time:.1f}s")
    col3.metric("Est. Tokens Used", f"{st.session_state.total_tokens_estimate:,}")
    col4.metric("Est. API Cost", f"${est_cost:.4f}")

    st.divider()
    st.markdown("### 💰 Cost Comparison: AI vs Human Reservation Managers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Human cost (3 managers)**")
        st.markdown('<div class="cost-bar cost-human">$195,000 / year</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**MewsFlow AI cost**")
        st.markdown('<div class="cost-bar cost-ai">~$1,950 / year</div>', unsafe_allow_html=True)
    st.markdown("### **99% cost reduction — saving NapYork $193,050/year**")

    st.divider()
    st.markdown("### Agent Performance Breakdown")
    for name, icon, desc in agents_config:
        stats = st.session_state.agent_stats[name]
        runs = stats["runs"]
        avg = (stats["total_time"] / runs) if runs > 0 else 0
        errors = stats["errors"]
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"{icon} **{name} Agent**")
            st.caption(desc)
        with col2:
            st.metric("Runs", runs)
        with col3:
            st.metric("Avg Time", f"{avg:.1f}s")
        with col4:
            st.metric("Errors", errors)
        if runs > 0:
            st.progress(min(runs / max(total_runs, 1), 1.0))
        st.markdown("")

    st.divider()
    st.markdown("### 🏗️ System Architecture")
    st.code("""
    User (Dashboard / Chatbot)
            ↓
    Streamlit App (this MVP)
            ↓
    LangChain Supervisor Agent (orchestrator.py)
            ↓ routes to
    ┌───────────────┬─────────────┬──────────────┬────────────────┐
    │  Reservation   │   Support   │   OTA Sync   │   Reporting    │
    │  Agent         │   Agent     │   Agent      │   Agent        │
    │  (6 tools)     │  (2 tools   │  (5 tools)   │  (5 tools)     │
    │                │   + RAG)    │              │                │
    └───────┬───────┴──────┬──────┴──────┬───────┴───────┬────────┘
            ↓              ↓             ↓               ↓
        Mews API     Vector DB     Spring Boot      Mews API
                     (Chroma)      OTA Service
    """, language=None)
    st.caption(
        "Production architecture adds: Node.js gateway (auth + WebSocket), "
        "FastAPI server, Spring Boot OTA service, Flask admin, "
        "React dashboard, Amazon EC2 + Docker + CI/CD."
    )