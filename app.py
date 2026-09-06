"""
ResearchMind — Multi-Agent Research Platform

Four specialized agents collaborate — searching, reading, writing, and
critiquing — to produce a polished, evidence-based research report on
any topic the user provides.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_TITLE = "ResearchMind"
STEP_ORDER = ["search", "reader", "writer", "critic"]
EXAMPLE_TOPICS = ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]


@dataclass(frozen=True)
class Step:
    key: str
    number: str
    title: str
    description: str
    spinner_label: str
    icon: str


PIPELINE: list[Step] = [
    Step("search", "01", "Search Agent", "Gathers recent web information", "Scanning the web for fresh sources…", "◈"),
    Step("reader", "02", "Reader Agent", "Scrapes & extracts deep content", "Extracting deep context from top source…", "◉"),
    Step("writer", "03", "Writer Chain", "Drafts the full research report", "Synthesizing the research report…", "◆"),
    Step("critic", "04", "Critic Chain", "Reviews & scores the report", "Auditing report quality…", "◎"),
]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

def configure_page() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · AI Research Agent",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: #eceaf7; }

        .stApp {
            background: #060714;
            background-image:
                radial-gradient(ellipse 70% 45% at 85% -8%, rgba(124,58,237,0.22) 0%, transparent 55%),
                radial-gradient(ellipse 55% 40% at 5% 100%, rgba(20,184,166,0.16) 0%, transparent 55%),
                linear-gradient(180deg, #06071a 0%, #060714 100%);
        }

        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding: 2.2rem 3rem 4rem; max-width: 1240px; }

        /* ── Hero ── */
        .hero { padding: 2.8rem 0 2rem; }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 0.5rem;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase; color: #2dd4bf;
            background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.25);
            padding: 0.4rem 0.9rem; border-radius: 999px; margin-bottom: 1.4rem;
        }
        .hero-badge::before { content: "●"; color: #2dd4bf; font-size: 0.55rem; }
        .hero h1 {
            font-size: clamp(2.4rem, 4.6vw, 3.6rem); font-weight: 700; line-height: 1.05;
            letter-spacing: -0.03em; margin: 0 0 0.8rem;
            background: linear-gradient(120deg, #f4f3ff 20%, #a78bfa 55%, #2dd4bf 85%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-sub { font-size: 1rem; font-weight: 400; color: #9490b3; max-width: 620px; line-height: 1.7; }

        .divider {
            height: 1px; margin: 2rem 0;
            background: linear-gradient(90deg, rgba(124,58,237,0.4), rgba(45,212,191,0.3), transparent);
        }

        /* ── Glass card ── */
        .glass-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(167,139,250,0.18);
            border-radius: 18px; padding: 1.9rem 2.2rem;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px -12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
            margin-bottom: 1.6rem;
        }

        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(167,139,250,0.3) !important;
            border-radius: 10px !important; color: #f4f3ff !important;
            font-family: 'Space Grotesk', sans-serif !important; font-size: 1rem !important;
            padding: 0.75rem 1rem !important;
            transition: border-color 0.2s, box-shadow 0.2s !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #a78bfa !important;
            box-shadow: 0 0 0 3px rgba(167,139,250,0.18) !important;
        }
        .stTextInput > label {
            font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important;
            letter-spacing: 0.16em !important; text-transform: uppercase !important;
            color: #a78bfa !important; font-weight: 500 !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #7c3aed 0%, #2dd4bf 130%) !important;
            color: #05060f !important; font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important; font-size: 0.95rem !important;
            letter-spacing: 0.02em !important; border: none !important; border-radius: 10px !important;
            padding: 0.75rem 2.2rem !important; width: 100%;
            box-shadow: 0 4px 24px rgba(124,58,237,0.4) !important;
            transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 32px rgba(124,58,237,0.5) !important; opacity: 0.96 !important;
        }
        .stButton > button:disabled { opacity: 0.35 !important; transform: none !important; }

        .example-chip {
            background: rgba(167,139,250,0.06); border: 1px solid rgba(167,139,250,0.2);
            border-radius: 999px; padding: 0.3rem 0.85rem; font-size: 0.74rem; color: #b7b3d6;
        }

        /* ── Pipeline ── */
        .step-card {
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px; padding: 1.3rem 1.6rem; margin-bottom: 1rem;
            position: relative; overflow: hidden; transition: all 0.3s ease;
        }
        .step-card.active {
            border-color: rgba(45,212,191,0.5); background: rgba(45,212,191,0.05);
            box-shadow: 0 0 0 1px rgba(45,212,191,0.15), 0 8px 24px -8px rgba(45,212,191,0.25);
        }
        .step-card.done { border-color: rgba(124,58,237,0.35); background: rgba(124,58,237,0.05); }
        .step-card::before {
            content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
            background: rgba(255,255,255,0.06); transition: background 0.3s;
        }
        .step-card.active::before { background: linear-gradient(180deg, #2dd4bf, #7c3aed); }
        .step-card.done::before { background: #7c3aed; }

        .step-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.35rem; }
        .step-icon { font-size: 0.95rem; color: #7c3aed; }
        .step-card.active .step-icon { color: #2dd4bf; }
        .step-num {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 500;
            letter-spacing: 0.14em; color: #746fa0;
        }
        .step-title { font-size: 0.95rem; font-weight: 600; color: #f4f3ff; }
        .step-status {
            margin-left: auto; font-family: 'IBM Plex Mono', monospace;
            font-size: 0.66rem; letter-spacing: 0.1em;
        }
        .status-waiting { color: #4d4a6b; }
        .status-running { color: #2dd4bf; }
        .status-done { color: #a78bfa; }
        .step-desc { font-size: 0.8rem; color: #746fa0; margin-top: 0.25rem; }

        /* ── Result panels ── */
        .result-panel {
            background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px; padding: 1.8rem 2rem; margin-top: 1rem; margin-bottom: 1.5rem;
        }
        .result-panel-title {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 500;
            letter-spacing: 0.18em; text-transform: uppercase; color: #a78bfa;
            margin-bottom: 1rem; padding-bottom: 0.7rem; border-bottom: 1px solid rgba(167,139,250,0.15);
        }
        .result-content {
            font-size: 0.92rem; line-height: 1.8; color: #cbc8e0;
            white-space: pre-wrap; font-family: 'Space Grotesk', sans-serif;
        }

        .report-panel {
            background: rgba(255,255,255,0.025); border: 1px solid rgba(124,58,237,0.3);
            border-radius: 18px; padding: 2.1rem 2.5rem; margin-top: 1rem;
            box-shadow: 0 12px 40px -20px rgba(124,58,237,0.35);
        }
        .feedback-panel {
            background: rgba(255,255,255,0.025); border: 1px solid rgba(45,212,191,0.3);
            border-radius: 18px; padding: 2.1rem 2.5rem; margin-top: 1rem;
            box-shadow: 0 12px 40px -20px rgba(45,212,191,0.3);
        }
        .panel-label {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.18em;
            text-transform: uppercase; margin-bottom: 1.2rem; padding-bottom: 0.7rem;
        }
        .panel-label.violet { color: #a78bfa; border-bottom: 1px solid rgba(167,139,250,0.2); }
        .panel-label.teal { color: #2dd4bf; border-bottom: 1px solid rgba(45,212,191,0.2); }

        /* ── Metrics ── */
        .metric-row { display: flex; gap: 0.9rem; margin: 1.4rem 0 0.4rem; }
        .metric-chip {
            flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 0.8rem 1rem;
        }
        .metric-chip-label {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #746fa0;
            letter-spacing: 0.1em; text-transform: uppercase;
        }
        .metric-chip-value { font-size: 1.15rem; font-weight: 700; color: #f4f3ff; margin-top: 0.15rem; }

        .stSpinner > div { color: #2dd4bf !important; }
        details summary {
            font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important;
            color: #9490b3 !important; letter-spacing: 0.08em !important; cursor: pointer;
        }

        .section-heading {
            font-size: 1.25rem; font-weight: 700; color: #f4f3ff; margin: 2rem 0 1rem;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .section-heading::before { content: "//"; color: #7c3aed; }

        .notice {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4d4a6b;
            text-align: center; margin-top: 3rem; letter-spacing: 0.08em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("done", False)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("topic", "")
    st.session_state.setdefault("run_started_at", None)
    st.session_state.setdefault("elapsed", None)


def reset_for_new_run(topic: str) -> None:
    st.session_state.results = {}
    st.session_state.topic = topic
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.error = None
    st.session_state.elapsed = None
    st.session_state.run_started_at = time.time()


def step_state(key: str, results: dict, running: bool) -> str:
    if key in results:
        return "done"
    if running:
        remaining = [k for k in STEP_ORDER if k not in results]
        if remaining and key == remaining[0]:
            return "running"
    return "waiting"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Multi-Agent AI System</div>
            <h1>ResearchMind</h1>
            <p class="hero-sub">
                Four specialized agents work in sequence — searching, reading, writing,
                and critiquing — to turn any topic into a sourced, reviewed research report.
            </p>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_step_card(step: Step, state: str) -> None:
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("RUNNING", "status-running"),
        "done": ("DONE", "status-done"),
    }
    label, status_cls = status_map[state]
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(
        f"""
        <div class="step-card {card_cls}">
            <div class="step-header">
                <span class="step-icon">{step.icon}</span>
                <span class="step-num">{step.number}</span>
                <span class="step-title">{step.title}</span>
                <span class="step-status {status_cls}">{label}</span>
            </div>
            <div class="step-desc">{step.description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel() -> tuple[str, bool]:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Quantum computing breakthroughs in 2025",
            key="topic_input",
        )
        run_clicked = st.button(
            "▶  Run Research Pipeline",
            use_container_width=True,
            disabled=st.session_state.running,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        chips = "".join(f'<span class="example-chip">{ex}</span>' for ex in EXAMPLE_TOPICS)
        st.markdown(
            f"""
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;margin-bottom:1.5rem;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#4d4a6b;letter-spacing:0.1em;">TRY →</span>
                {chips}
            </div>
            """,
            unsafe_allow_html=True,
        )
    return topic, run_clicked


def render_pipeline_panel(results: dict, running: bool) -> None:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    for step in PIPELINE:
        render_step_card(step, step_state(step.key, results, running))

    completed = len([k for k in STEP_ORDER if k in results])
    pct = int((completed / len(STEP_ORDER)) * 100)
    st.markdown(
        f"""
        <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;margin-top:0.4rem;">
            <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#7c3aed,#2dd4bf);
                border-radius:999px;transition:width 0.5s ease;"></div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#746fa0;margin-top:0.4rem;text-align:right;">
            {pct}% complete
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_raw_result(title: str, content: str) -> None:
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-panel-title">{title}</div>
            <div class="result-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(results: dict) -> None:
    word_count = len(results.get("writer", "").split())
    reading_time = max(1, round(word_count / 200))
    elapsed = st.session_state.elapsed
    elapsed_str = f"{elapsed:.1f}s" if elapsed else "—"

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-label">Stages</div>
                <div class="metric-chip-value">{len(results)}/4</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Words</div>
                <div class="metric-chip-value">{word_count}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Read Time</div>
                <div class="metric-chip-value">{reading_time} min</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Run Time</div>
                <div class="metric-chip-value">{elapsed_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(results: dict) -> None:
    if not results:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
    render_metrics(results)

    if "search" in results:
        with st.expander("Search Results — raw"):
            render_raw_result("Search Agent Output", results["search"])

    if "reader" in results:
        with st.expander("Scraped Content — raw"):
            render_raw_result("Reader Agent Output", results["reader"])

    if "writer" in results:
        st.markdown(
            '<div class="report-panel"><div class="panel-label violet">Final Research Report</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "↓  Download Report (.md)",
            data=results["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in results:
        st.markdown(
            '<div class="feedback-panel"><div class="panel-label teal">Critic Feedback</div>',
            unsafe_allow_html=True,
        )
        st.markdown(results["critic"])
        st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="notice">
            ResearchMind · Powered by a LangChain multi-agent pipeline · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(topic: str) -> None:
    """Runs all four agent stages, persisting partial results after each
    stage so the UI stays accurate even if a later stage fails."""
    results: dict[str, str] = {}

    try:
        with st.spinner(PIPELINE[0].spinner_label):
            search_agent = build_search_agent()
            search_response = search_agent.invoke(
                {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
            )
            results["search"] = search_response["messages"][-1].content
            st.session_state.results = dict(results)

        with st.spinner(PIPELINE[1].spinner_label):
            reader_agent = build_reader_agent()
            reader_response = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Based on the following search results about '{topic}', "
                            "pick the most relevant URL and scrape it for deeper content.\n\n"
                            f"Search Results:\n{results['search'][:800]}",
                        )
                    ]
                }
            )
            results["reader"] = reader_response["messages"][-1].content
            st.session_state.results = dict(results)

        with st.spinner(PIPELINE[2].spinner_label):
            combined_research = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = writer_chain.invoke({"topic": topic, "research": combined_research})
            st.session_state.results = dict(results)

        with st.spinner(PIPELINE[3].spinner_label):
            results["critic"] = critic_chain.invoke({"report": results["writer"]})
            st.session_state.results = dict(results)

    except Exception as exc:  # surfaced in the UI rather than crashing the app
        st.session_state.error = str(exc)

    finally:
        if st.session_state.run_started_at:
            st.session_state.elapsed = time.time() - st.session_state.run_started_at
        st.session_state.running = False
        st.session_state.done = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_page()
    inject_css()
    init_state()

    render_hero()

    col_input, _, col_pipeline = st.columns([5, 0.5, 4])
    with col_input:
        topic, run_clicked = render_input_panel()
    with col_pipeline:
        render_pipeline_panel(st.session_state.results, st.session_state.running)

    if run_clicked:
        if not topic.strip():
            st.warning("Please enter a research topic first.")
        else:
            reset_for_new_run(topic.strip())
            st.rerun()

    if st.session_state.running and not st.session_state.done:
        run_pipeline(st.session_state.topic)
        st.rerun()

    if st.session_state.error:
        st.error(f"Pipeline failed: {st.session_state.error}")

    render_results(st.session_state.results)
    render_footer()


if __name__ == "__main__":
    main()
