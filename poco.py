"""app.py - Streamlit app for automated CLO -> PO (AICTE PO1-PO12) mapping
Features:
 - Upload CLO CSV (column name 'CLO') or paste CLOs
 - Uses sentence-transformers embeddings (all-MiniLM-L6-v2 by default)
 - Computes cosine similarity CLO x PO (12 AICTE POs preloaded)
 - Threshold mapping to weights {3,2,1,0} (configurable)
 - Bloom verb extraction reinforcement (optional)
 - Explainable justification text per mapping
 - Generates downloadable Excel and simple PDF report
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Ensure NLTK stuff
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# -------------------------
# Helper / configuration
# -------------------------
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MULTI_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# AICTE PO1..PO12 (short standard descriptions; replace if you have institution-specific wording)
AICTE_POS = [
    "Engineering knowledge: Apply knowledge of mathematics, science, engineering fundamentals and an engineering specialization to the solution of complex engineering problems.",
    "Problem analysis: Identify, formulate, review research literature and analyze complex engineering problems reaching substantiated conclusions using first principles of mathematics, natural sciences and engineering sciences.",
    "Design/development of solutions: Design solutions for complex engineering problems and design system components or processes that meet the specified needs with appropriate consideration for public health and safety, cultural, societal, and environmental considerations.",
    "Conduct investigations of complex problems: Use research-based knowledge and research methods including design of experiments, analysis and interpretation of data, and synthesis of information to provide valid conclusions.",
    "Modern tool usage: Create, select and apply appropriate techniques, resources, and modern engineering and IT tools including prediction and modeling to complex engineering activities with an understanding of the limitations.",
    "The engineer and society: Apply reasoning informed by contextual knowledge to assess societal, health, safety, legal and cultural issues and the consequent responsibilities relevant to professional engineering practice.",
    "Environment & sustainability: Understand the impact of the professional engineering solutions in societal and environmental contexts and demonstrate knowledge of and need for sustainable development.",
    "Ethics: Apply ethical principles and commit to professional ethics and responsibilities and norms of engineering practice.",
    "Individual and team work: Function effectively as an individual, and as a member or leader in diverse teams and in multi-disciplinary settings.",
    "Communication: Communicate effectively on complex engineering activities with the engineering community and with society at large.",
    "Project management and finance: Demonstrate knowledge and understanding of engineering and management principles and apply these to one’s own work, as a member and leader in a team and to manage projects in multi-disciplinary environments.",
    "Life-long learning: Recognize the need for, and have the preparation and ability to engage in independent and life-long learning in the broadest context of technological change."
]

# Bloom verbs simple list (can be expanded or replaced with a robust parser)
BLOOM_VERBS = [
    "remember","understand","apply","analyze","evaluate","create",
    "identify","describe","explain","construct","design","implement","test","synthesize","compare"
]

# -------------------------
# Utility functions
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name=DEFAULT_MODEL):
    return SentenceTransformer(model_name)

def preprocess_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_verbs(text: str):
    # crude verb extraction: look for known bloom verbs (lemmatized)
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    found = sorted(set([v for v in BLOOM_VERBS if v in lemmas]))
    return found

def embed_texts(model, texts, batch_size=32):
    texts = [preprocess_text(t) for t in texts]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
    return embeddings

def similarity_matrix(clo_emb, po_emb):
    # returns matrix shape (n_clo, n_po)
    sim = cosine_similarity(clo_emb, po_emb)
    # clip numerical noise
    sim = np.clip(sim, 0.0, 1.0)
    return sim

def map_thresholds(sim_matrix, tau_s=0.70, tau_m=0.40, tau_w=0.20):
    # produce integer matrix of {3,2,1,0}
    def score_from_sim(s):
        if s >= tau_s: return 3
        if s >= tau_m: return 2
        if s >= tau_w: return 1
        return 0
    vec = np.vectorize(score_from_sim)
    return vec(sim_matrix).astype(int)

def apply_bloom_reinforcement(weight_matrix, cloverbs, po_verbs_list, alpha=0.2):
    # po_verbs_list: for each PO, list of verbs expected (we'll try simple overlap)
    hybrid = weight_matrix.astype(float).copy()
    n_clo, n_po = hybrid.shape
    for i in range(n_clo):
        for j in range(n_po):
            overlap = 0
            if len(cloverbs[i])>0 and len(po_verbs_list[j])>0:
                # overlap if any verb matches
                overlap = 1 if any(v in po_verbs_list[j] for v in cloverbs[i]) else 0
            hybrid[i,j] = min(3.0, hybrid[i,j] + alpha * overlap)
    # round to nearest integer (but keep ints)
    return np.rint(hybrid).astype(int)

def justification_for_pair(clo_text, po_text, sim_score, weight, clo_verbs, po_verbs):
    # builds a human-readable justification sentence
    reasons = []
    # similarity reason
    reasons.append(f"Semantic similarity = {sim_score:.2f}.")
    # verb match
    if clo_verbs and po_verbs:
        common = set(clo_verbs).intersection(set(po_verbs))
        if common:
            reasons.append(f"Shared verbs: {', '.join(common)}.")
    # keyword overlap (simple approach)
    clo_tokens = set([t for t in re.findall(r'\w+', clo_text.lower()) if len(t)>2])
    po_tokens = set([t for t in re.findall(r'\w+', po_text.lower()) if len(t)>2])
    common_kw = sorted(list(clo_tokens.intersection(po_tokens)))
    if len(common_kw)>0:
        # limit number of printed keywords
        reasons.append(f"Common keywords: {', '.join(common_kw[:6])}.")
    # final mapping label
    label = {3:"Strong (3)", 2:"Moderate (2)", 1:"Weak (1)", 0:"No mapping (0)"}[weight]
    return f"{label} — {' '.join(reasons)}"

def df_to_excel_bytes(matrix_df, clo_list, AICTE_POS, justification_df):
    import pandas as pd
    import io

    output = io.BytesIO()

    # Use context manager (correct in pandas 2.x)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        matrix_df.to_excel(writer, index=False, sheet_name="CO-PO Matrix")
        pd.DataFrame({"CLOs": clo_list}).to_excel(writer, index=False, sheet_name="CLO List")
        pd.DataFrame({"POs": AICTE_POS}).to_excel(writer, index=False, sheet_name="PO List")
        justification_df.to_excel(writer, index=False, sheet_name="Justification")

    output.seek(0)
    return output.read()

def df_to_pdf_bytes(matrix_df, clo_list, AICTE_POS, justification_df):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    import io

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    h2 = styles["Heading2"]
    body = styles["BodyText"]

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=30)

    story = []

    # ---------- Title ----------
    story.append(Paragraph("CO–PO Mapping Report", title_style))
    story.append(Spacer(1, 12))

    # ---------- PO LIST ----------
    story.append(Paragraph("<b>Program Outcomes (POs)</b>", h2))
    for po in AICTE_POS:
        story.append(Paragraph(po, body))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))

    # ---------- CO–PO MATRIX ----------
    story.append(Paragraph("<b>CO–PO Matrix</b>", h2))

    # Prepare matrix table
    table_data = [ ["CLO"] + [f"PO{i+1}" for i in range(len(AICTE_POS))] ]

    for idx, clo in enumerate(clo_list):
        row = [f"CLO{idx+1}"]
        row += list(matrix_df.iloc[idx])
        table_data.append(row)

    table = Table(table_data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.black),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))

    story.append(table)
    story.append(Spacer(1, 18))

    # ---------- JUSTIFICATIONS ----------
    story.append(Paragraph("<b>Mapping Justifications</b>", h2))
    story.append(Spacer(1, 6))

    for _, row in justification_df.iterrows():
        clo = row["CLO"]
        po = row["PO"]
        level = row["Level"]
        sim = row["Similarity"]
        keywords = row["Common Keywords"]

        text = (
            f"<b>{clo} → {po}</b>: "
            f"{level} — Similarity = {sim:.2f}. "
            f"Common keywords: {', '.join(keywords)}"
        )
        story.append(Paragraph(text, body))
        story.append(Spacer(1, 6))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def split_text_to_lines(text, max_chars=100):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="CLO → PO (AICTE) Mapper", layout="wide")
st.title("Automated CLO → PO (AICTE PO1–PO12) Mapper")
st.markdown("Upload Course Learning Outcomes (CLOs) and generate AICTE-aligned CO–PO mapping matrix with explanations. "
            "Uses sentence-transformer embeddings and threshold-based mapping (3,2,1,0).")

with st.expander("Example / Quick Start"):
    st.write("You can paste CLOs (one per line) or upload CSV with column `CLO`.")
    st.write("Try the example CLOs:")
    if st.button("Load example CLOs"):
        example = [
            "Apply knowledge of mathematics and basic sciences to analyze engineering problems.",
            "Design and implement software modules to meet specified requirements.",
            "Conduct testing and validation for software systems using modern tools.",
            "Work effectively as a team member and communicate project outcomes.",
            "Understand professional ethics and societal responsibilities in engineering practice."
        ]
        st.session_state['example_clos'] = example

# Left column: inputs
col1, col2 = st.columns([1,2])
with col1:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV (column name 'CLO')", type=['csv','txt'])
    text_area = st.text_area("Or paste CLOs (one per line)", height=150)
    use_example = st.checkbox("Use the example CLOs loaded above", value=False)
    if use_example and 'example_clos' in st.session_state:
        clo_list = st.session_state['example_clos']
    else:
        clo_list = None

    model_choice = st.selectbox("Embedding model", options=[DEFAULT_MODEL, MULTI_MODEL], index=0,
                                help="Use multilingual model for non-English CLOs")
    st.write("Mapping thresholds (adjust if required):")
    tau_s = st.slider("Strong threshold (>=)", 0.5, 0.95, 0.70, step=0.01)
    tau_m = st.slider("Moderate threshold (>=)", 0.2, 0.69, 0.40, step=0.01)
    tau_w = st.slider("Weak threshold (>=)", 0.0, 0.39, 0.20, step=0.01)
    alpha = st.slider("Bloom verb reinforcement (alpha)", 0.0, 1.0, 0.2, step=0.05)

    st.write("PO verbs (optional) — used for Bloom overlap reinforcement.")
    st.caption("Leave blank to use default empty lists. Enter verbs for each PO separated by commas (optional).")
    po_verbs_inputs = []
    with st.expander("Edit PO descriptions / verbs"):
        for i, po in enumerate(AICTE_POS):
            default_verbs = ""  # you may add PO-specific verbs if desired
            txt = st.text_input(f"PO{i+1} short description", value=po, key=f"po_desc_{i}")
            vb = st.text_input(f"PO{i+1} verbs (comma-separated)", value="", key=f"po_verb_{i}")
            po_verbs_inputs.append([x.strip().lower() for x in vb.split(',') if x.strip()])

# Prepare CLO list from user inputs
if clo_list is None:
    # Read uploaded file
    if uploaded is not None:
        try:
            dfu = pd.read_csv(uploaded)
            if 'CLO' in dfu.columns:
                clo_list = dfu['CLO'].astype(str).tolist()
            else:
                # attempt to use first column
                clo_list = dfu.iloc[:,0].astype(str).tolist()
        except Exception as e:
            st.error(f"Could not read file: {e}")
            clo_list = []
    elif text_area and text_area.strip() != "":
        clo_list = [line.strip() for line in text_area.splitlines() if line.strip()!='']
    else:
        clo_list = []

# Right column: processing and results
with col2:
    st.header("Run Mapping")
    st.write(f"Detected {len(clo_list)} CLO(s).")
    if len(clo_list) == 0:
        st.info("Provide CLOs via upload, paste, or example to run mapping.")
    else:
        if st.button("Compute Mapping"):
            with st.spinner("Loading model and computing embeddings..."):
                model = load_model(model_choice)
                clo_emb = embed_texts(model, clo_list)
                po_emb = embed_texts(model, AICTE_POS)
                sim_mat = similarity_matrix(clo_emb, po_emb)
                raw_weight = map_thresholds(sim_mat, tau_s=tau_s, tau_m=tau_m, tau_w=tau_w)

                # extract verbs for clo and po
                clo_verbs = [extract_verbs(c) for c in clo_list]
                po_verbs = []
                # prefer user-supplied verbs if present (po_verbs_inputs)
                for idx in range(len(AICTE_POS)):
                    user_verbs = st.session_state.get(f"po_verb_{idx}", "")
                    if isinstance(user_verbs, str) and user_verbs.strip() != "":
                        parsed = [v.strip().lower() for v in user_verbs.split(',') if v.strip()]
                        po_verbs.append(parsed)
                    else:
                        po_verbs.append([])

                # apply bloom reinforcement if alpha>0
                if alpha > 0:
                    hybrid = apply_bloom_reinforcement(raw_weight, clo_verbs, po_verbs, alpha=alpha)
                else:
                    hybrid = raw_weight

                # prepare matrix DataFrame
                col_names = [f"PO{j+1}" for j in range(len(AICTE_POS))]
                matrix_df = pd.DataFrame(hybrid, columns=col_names)
                matrix_df.index = [f"CLO{idx+1}" for idx in range(len(clo_list))]

                # prepare justifications
                just_rows = []
                for i in range(sim_mat.shape[0]):
                    for j in range(sim_mat.shape[1]):
                        just = justification_for_pair(
                            clo_text=clo_list[i],
                            po_text=AICTE_POS[j],
                            sim_score=sim_mat[i,j],
                            weight=hybrid[i,j],
                            clo_verbs=clo_verbs[i],
                            po_verbs=po_verbs[j]
                        )
                        just_rows.append({
                            "CLO_index": i,
                            "PO_index": j,
                            "CLO": clo_list[i],
                            "PO": AICTE_POS[j],
                            "Similarity": round(float(sim_mat[i,j]),3),
                            "Weight": int(hybrid[i,j]),
                            "Justification": just
                        })
                justification_df = pd.DataFrame(just_rows)

                st.success("Mapping computed.")
                st.subheader("CO–PO Matrix (weights 0/1/2/3)")
                st.dataframe(matrix_df.style.format(precision=0))
                st.download_button("Download Matrix as Excel",
                                   data=df_to_excel_bytes(matrix_df, clo_list, AICTE_POS, justification_df),
                                   file_name="co_po_matrix.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # small summary: counts of strong mappings
                strong_counts = (matrix_df == 3).sum(axis=0)
                st.write("Strong mapping counts per PO (number of CLOs mapped strongly):")
                st.bar_chart(strong_counts)

                # Show sample justifications (top mappings by similarity)
                st.subheader("Top Mappings (by similarity)")
                topk = justification_df.sort_values(by='Similarity', ascending=False).head(20)
                st.table(topk[['CLO_index','PO_index','Similarity','Weight','Justification']].rename(
                    columns={"CLO_index":"CLO#","PO_index":"PO#"}
                ))

                # Create PDF bytes
                pdf_bytes = df_to_pdf_bytes(matrix_df, clo_list, AICTE_POS, justification_df,
                                           title="CO-PO Mapping Report")
                st.download_button("Download PDF Report", data=pdf_bytes, file_name="co_po_report.pdf", mime="application/pdf")

                # Optional: show raw similarity heatmap (small)
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, max(3, len(clo_list)*0.2)))
                    im = ax.imshow(sim_mat, vmin=0, vmax=1, aspect='auto')
                    ax.set_yticks(np.arange(len(clo_list)))
                    ax.set_yticklabels([f"CLO{i+1}" for i in range(len(clo_list))])
                    ax.set_xticks(np.arange(len(AICTE_POS)))
                    ax.set_xticklabels([f"PO{j+1}" for j in range(len(AICTE_POS))], rotation=45, ha='right')
                    ax.set_title("CLO vs PO similarity heatmap")
                    fig.colorbar(im, ax=ax)
                    st.pyplot(fig)
                except Exception:
                    pass

                # expose variables in session for further actions
                st.session_state['last_matrix_df'] = matrix_df
                st.session_state['last_just_df'] = justification_df
                st.session_state['last_sim_mat'] = pd.DataFrame(sim_mat, columns=col_names)
                st.session_state['last_clos'] = clo_list

        # end compute block

    # Buttons to re-download last results if present
    if 'last_matrix_df' in st.session_state:
        st.markdown("---")
        st.write("Previous run:")
        if st.button("Download last Excel"):
            matrix_df = st.session_state['last_matrix_df']
            clo_list = st.session_state['last_clos']
            just_df = st.session_state['last_just_df']
            data = df_to_excel_bytes(matrix_df, clo_list, AICTE_POS, just_df)
            st.download_button("Click to download", data=data, file_name="co_po_matrix_last.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("This tool is provided as an academic prototype. For production deployment, consider "
           "model fine-tuning on domain mappings, secure hosting of the model, and additional QA steps.")




