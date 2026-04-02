import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from dotenv import load_dotenv

from utils.document_processor import extract_text_from_pdf, extract_text_from_txt, chunk_text
from utils.vector_store import load_embedding_model, get_chroma_client, create_or_get_collection, add_chunks_to_vectorstore, retrieve_relevant_chunks
from utils.llm_client import get_groq_client, generate_response
from utils.analytics import load_sentiment_model, analyze_sentiment, get_sentiment_summary, cluster_unanswered_questions

load_dotenv()

st.set_page_config(
    page_title="AI Support Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%); color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #111827 100%); border-right: 1px solid #1e293b; }
    .main-header { background: linear-gradient(135deg, #1e3a5f 0%, #1a237e 50%, #0d47a1 100%); border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #2d4a7a; box-shadow: 0 8px 32px rgba(13,71,161,0.3); }
    .main-header h1 { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0; }
    .main-header p { color: #90caf9; margin: 0.5rem 0 0 0; }
    .user-message { background: linear-gradient(135deg, #1e3a5f, #1a237e); border-radius: 12px 12px 4px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; margin-left: 20%; border: 1px solid #2d4a7a; color: #e2e8f0; }
    .bot-message { background: #161b22; border-radius: 12px 12px 12px 4px; padding: 0.75rem 1rem; margin: 0.5rem 0; margin-right: 20%; border: 1px solid #1e293b; color: #e2e8f0; }
    .metric-card { background: #161b22; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { color: #64748b; font-size: 0.85rem; margin-top: 0.25rem; }
    .status-ready { background: #064e3b; color: #6ee7b7; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; border: 1px solid #059669; display: inline-block; }
    .status-pending { background: #1c1917; color: #fbbf24; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; border: 1px solid #d97706; display: inline-block; }
    .section-title { font-size: 1.25rem; font-weight: 600; color: #93c5fd; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
    .stButton > button { background: linear-gradient(135deg, #1d4ed8, #1e40af) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; transform: translateY(-1px) !important; }
    .stTextInput > div > div > input { background: #161b22 !important; border: 1px solid #1e293b !important; color: #e2e8f0 !important; border-radius: 8px !important; }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; border: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; border-radius: 6px !important; }
    .stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }
    hr { border-color: #1e293b !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "chat_history": [], "conversations_log": [],
        "unanswered_questions": [], "docs_loaded": False,
        "chunks_count": 0, "bot_name": "Support Agent",
        "company_name": "My Company", "total_messages": 0,
        "groq_api_key": os.getenv("GROQ_API_KEY", "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"),
        "collection_name": "support_docs_v1",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

embedding_model = load_embedding_model()
chroma_client = get_chroma_client()
sentiment_model = load_sentiment_model()

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Support Builder")
    st.markdown("---")
    st.markdown("**🔑 Groq API Key**")
    api_key = st.text_input("API Key", value=st.session_state.groq_api_key,
                            type="password", placeholder="gsk_...", label_visibility="collapsed")
    if api_key:
        st.session_state.groq_api_key = api_key
        st.markdown('<span class="status-ready">✓ API Key Set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pending">⚠ API Key Required</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Bot Configuration**")
    st.session_state.company_name = st.text_input("Company Name", value=st.session_state.company_name)
    st.session_state.bot_name = st.text_input("Bot Name", value=st.session_state.bot_name)
    st.markdown("---")
    st.markdown("**📊 Quick Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Chunks", st.session_state.chunks_count)
    c2.metric("Messages", st.session_state.total_messages)
    if st.session_state.docs_loaded:
        st.markdown('<span class="status-ready">✓ Knowledge Base Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pending">⚠ Upload Docs First</span>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversations_log = []
        st.session_state.unanswered_questions = []
        st.session_state.docs_loaded = False
        st.session_state.chunks_count = 0
        st.session_state.total_messages = 0
        st.rerun()
    st.markdown("---")
    st.markdown("**📖 How to Use**")
    st.markdown("1. Enter Groq API key\n2. Set company & bot name\n3. Upload your FAQ/docs\n4. Chat!\n5. View analytics")
    st.markdown("[Get Free Groq Key →](https://console.groq.com)")

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Customer Support Agent Builder</h1>
    <p>Upload your documents → Build intelligent chatbot → Analyze conversations with ML & NLP</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📁 Setup & Knowledge Base", "💬 Chat Interface", "📊 Analytics Dashboard"])

# ── TAB 1: SETUP ─────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Upload Your Knowledge Base Documents</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files (FAQs, policies, product docs)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("🚀 Process & Build Knowledge Base", use_container_width=True):
                collection = create_or_get_collection(chroma_client, st.session_state.collection_name)
                all_chunks = []
                progress = st.progress(0)
                status = st.empty()
                for i, f in enumerate(uploaded_files):
                    status.text(f"Processing: {f.name}...")
                    fb = f.read()
                    text = extract_text_from_pdf(fb) if f.name.endswith(".pdf") else extract_text_from_txt(fb)
                    all_chunks.extend(chunk_text(text))
                    progress.progress((i + 1) / len(uploaded_files))
                if all_chunks:
                    status.text("Generating embeddings and storing in vector database...")
                    count = add_chunks_to_vectorstore(all_chunks, collection, embedding_model)
                    st.session_state.chunks_count = count
                    st.session_state.docs_loaded = True
                    progress.progress(1.0)
                    status.empty()
                    st.success(f"✅ Done! {count} chunks from {len(uploaded_files)} file(s) stored in vector DB.")
                else:
                    st.error("No text extracted. Please check your files.")
    with col_r:
        st.info("**Good documents to upload:**\n- FAQ documents\n- Product manuals\n- Return/refund policies\n- Pricing info\n- Company info\n- Support guides")
        st.info("**Tips for best results:**\n- Use clear structured text\n- Avoid scanned image PDFs\n- Include common questions\n- Keep docs relevant to support")
        st.markdown("**📎 Sample file included:** `data/sample_faq.txt`")

# ── TAB 2: CHAT ──────────────────────────────────────────────────
with tab2:
    st.markdown(f'<p class="section-title">💬 Chat with {st.session_state.bot_name} — {st.session_state.company_name}</p>', unsafe_allow_html=True)
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
    elif not st.session_state.docs_loaded:
        st.warning("⚠️ Please upload and process your documents in the Setup tab first.")
    else:
        if not st.session_state.chat_history:
            st.markdown(f'<div class="bot-message">👋 Hello! I\'m <strong>{st.session_state.bot_name}</strong>, your AI support assistant for <strong>{st.session_state.company_name}</strong>. How can I help you today?</div>', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                s = msg.get("sentiment", {})
                badge = f'<span style="background:{s.get("color","#FFB300")}22;color:{s.get("color","#FFB300")};padding:2px 8px;border-radius:20px;font-size:0.7rem;border:1px solid {s.get("color","#FFB300")}44;">{s.get("emoji","😐")} {s.get("sentiment","")}</span>' if s else ""
                st.markdown(f'<div class="user-message">👤 {msg["content"]} {badge}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        col_i, col_b = st.columns([5, 1])
        with col_i:
            user_input = st.text_input("Message", placeholder="Ask anything about our products or services...",
                                       label_visibility="collapsed", key="user_msg")
        with col_b:
            send = st.button("Send ➤", use_container_width=True)

        if send and user_input.strip():
            with st.spinner("Thinking..."):
                try:
                    sentiment = analyze_sentiment(user_input, sentiment_model)
                    collection = create_or_get_collection(chroma_client, st.session_state.collection_name)
                    chunks = retrieve_relevant_chunks(user_input, collection, embedding_model)
                    groq_client = get_groq_client(st.session_state.groq_api_key)
                    response = generate_response(
                        groq_client, user_input, chunks,
                        st.session_state.chat_history,
                        st.session_state.bot_name,
                        st.session_state.company_name
                    )
                    if "don't have information" in response.lower() or "contact our support" in response.lower():
                        st.session_state.unanswered_questions.append(user_input)
                    st.session_state.chat_history.append({"role": "user", "content": user_input, "sentiment": sentiment})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.conversations_log.append({
                        "message": user_input, "sentiment": sentiment["sentiment"],
                        "emoji": sentiment["emoji"], "color": sentiment["color"], "response": response
                    })
                    st.session_state.total_messages += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}. Please check your Groq API key.")

        st.markdown("**💡 Sample Questions to Try:**")
        sc = st.columns(3)
        samples = ["What is your refund policy?", "How do I track my order?", "What payment methods do you accept?"]
        for i, q in enumerate(samples):
            if sc[i].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["user_msg"] = q

# ── TAB 3: ANALYTICS ─────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">📊 Conversation Analytics & Insights</p>', unsafe_allow_html=True)
    if not st.session_state.conversations_log:
        st.info("💬 Start chatting with your bot to see analytics here. The dashboard will populate as customers interact.")
    else:
        logs = st.session_state.conversations_log
        sc = get_sentiment_summary(logs)
        total = len(logs)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Messages</div></div>', unsafe_allow_html=True)
        sat_pct = round(sc.get("Satisfied", 0) / total * 100) if total else 0
        c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#6ee7b7">{sat_pct}%</div><div class="metric-label">Satisfaction Rate</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#fbbf24">{len(st.session_state.unanswered_questions)}</div><div class="metric-label">Unanswered Queries</div></div>', unsafe_allow_html=True)
        frust_pct = round(sc.get("Frustrated", 0) / total * 100) if total else 0
        c4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171">{frust_pct}%</div><div class="metric-label">Frustration Rate</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)

        with cl:
            st.markdown("**Customer Sentiment Distribution**")
            fig = go.Figure(data=[go.Pie(
                labels=list(sc.keys()), values=list(sc.values()), hole=0.5,
                marker=dict(colors=["#6ee7b7", "#fbbf24", "#f87171"], line=dict(color="#0d1117", width=2)),
                textfont=dict(color="white")
            )])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e2e8f0"), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
                              margin=dict(t=20, b=20, l=20, r=20), height=280)
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown("**Sentiment Over Time**")
            sm = {"Satisfied": 1, "Neutral": 0, "Frustrated": -1}
            df = pd.DataFrame([{"i": i+1, "score": sm.get(l["sentiment"], 0)} for i, l in enumerate(logs)])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["i"], y=df["score"], mode="lines+markers",
                line=dict(color="#60a5fa", width=2),
                marker=dict(size=8, color=["#6ee7b7" if s==1 else "#fbbf24" if s==0 else "#f87171" for s in df["score"]]),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.1)"
            ))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e293b", title="Message #"),
                yaxis=dict(gridcolor="#1e293b", tickvals=[-1,0,1], ticktext=["Frustrated","Neutral","Satisfied"]),
                margin=dict(t=20, b=20, l=20, r=20), height=280)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("**📝 Full Conversation Log**")
        df_log = pd.DataFrame([{
            "#": i+1,
            "Customer Message": l["message"],
            "Sentiment": f"{l['emoji']} {l['sentiment']}",
            "Bot Response (Preview)": l["response"][:80]+"..."
        } for i, l in enumerate(logs)])
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        if st.session_state.unanswered_questions:
            st.markdown("---")
            st.markdown("**🔍 Knowledge Gap Analysis — Unanswered Questions Clustering (KMeans ML)**")
            st.caption("Questions your bot couldn't answer — clustered by topic using KMeans to show what to add to your knowledge base.")
            clustered = cluster_unanswered_questions(st.session_state.unanswered_questions)
            df_c = pd.DataFrame(clustered)
            df_c.columns = ["Cluster ID", "Unanswered Question", "Detected Topic"]
            topic_counts = df_c["Detected Topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            fig3 = px.bar(topic_counts, x="Topic", y="Count", color="Count",
                         color_continuous_scale=["#1e3a5f", "#60a5fa"])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"), coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20), height=250)
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(df_c, use_container_width=True, hide_index=True)
