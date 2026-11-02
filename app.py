import os
# THIS MUST BE AT THE VERY TOP BEFORE TORCH
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, util
import fitz
import time
import threading
from pathlib import Path

import pandas as pd

from text_processing import chunk_text, clean_pdf_text

st.set_page_config(page_title="Argus AI Command Center", layout="wide")

# The fine-tuned synthesis checkpoint is produced by train.py and excluded from
# version control by .gitignore. When it is absent the stage falls back to the
# public base checkpoint so a fresh clone still runs.
SYNTHESIS_MODEL_PATH = Path("argus_v2_bart")
FALLBACK_SYNTHESIS_MODEL = "facebook/bart-large-cnn"


@st.cache_resource
def load_models():
    # Stage 1: Triage (zero-shot classification)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Stage 2: Semantic auditor (bi-encoder)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Stage 3: Synthesis
    using_finetuned = (SYNTHESIS_MODEL_PATH / "config.json").is_file()
    source = str(SYNTHESIS_MODEL_PATH) if using_finetuned else FALLBACK_SYNTHESIS_MODEL
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)

    return classifier, semantic_model, tokenizer, model, using_finetuned


classifier, semantic_model, tokenizer, model, using_finetuned = load_models()

# Security Knowledge Base
COMPLIANCE_REQUIREMENTS = {
    "Data Privacy and GDPR": [
        "Requires strong encryption for data at rest", 
        "Explicit user consent mechanisms", 
        "Right to be forgotten protocol", 
        "Data breach notification timeline"
    ],
    "Healthcare and HIPAA": [
        "Protected health information safeguards", 
        "Strict role based access control", 
        "Audit logging for system access", 
        "Secure transmission protocols"
    ],
    "Network Security (NIST)": [
        "Zero trust network architecture", 
        "Multi-factor authentication enforcement", 
        "Incident response planning", 
        "Continuous monitoring systems"
    ]
}

# Pre-written remediation text, keyed by control objective. Fixed templates,
# not model output.
REMEDIATION_TEMPLATES = {
    "Requires strong encryption for data at rest": "All sensitive data repositories must implement AES-256 encryption at rest to prevent unauthorised exfiltration.",
    "Explicit user consent mechanisms": "Data collection interfaces must require explicit, opt-in consent from the user, with clear provisions for withdrawal.",
    "Right to be forgotten protocol": "A formalised protocol must be established to purge user data from all operational databases and backups within 30 days of request.",
    "Data breach notification timeline": "In the event of a confirmed breach, regulatory bodies and affected users must be notified within 72 hours.",
    "Protected health information safeguards": "PHI must be strictly compartmentalised on dedicated, isolated subnets with end-to-end encryption.",
    "Strict role based access control": "System access must follow the principle of least privilege, enforcing strict Role-Based Access Control (RBAC) matrices.",
    "Audit logging for system access": "Immutable audit logs must record all authentication attempts, privilege escalations, and critical data access events.",
    "Secure transmission protocols": "All internal and external data transmission must be secured using TLS 1.3 or higher.",
    "Zero trust network architecture": "The network must implement Zero Trust architecture, requiring continuous authentication for all lateral traffic.",
    "Multi-factor authentication enforcement": "MFA is mandatory for all user accounts, administrative portals, and VPN access endpoints.",
    "Incident response planning": "A documented Incident Response plan must be maintained, detailing specific containment and eradication procedures.",
    "Continuous monitoring systems": "Network perimeters and internal endpoints must be subjected to 24/7 continuous monitoring via SIEM solutions."
}

# --- PIPELINE ORCHESTRATION ---
class PipelineOrchestrator:
    def __init__(self):
        self.state = {}
        
    def execute_node_1_triage(self, text, ui_element):
        ui_element.info("Stage 1 (Triage): classifying regulatory domain...")
        categories = list(COMPLIANCE_REQUIREMENTS.keys())
        classification = classifier(text[:2000], categories, multi_label=False)
        self.state['domain'] = classification['labels'][0]
        time.sleep(0.5)
        ui_element.success(f"Stage 1 complete. Domain: **{self.state['domain']}**")

    def execute_node_2_semantic_audit(self, text, ui_element):
        ui_element.info("Stage 2 (Auditor): scoring controls against the document...")
        domain = self.state['domain']
        required_controls = COMPLIANCE_REQUIREMENTS[domain]
        
        chunks = chunk_text(text)
        doc_embeddings = semantic_model.encode(chunks, convert_to_tensor=True)
        control_embeddings = semantic_model.encode(required_controls, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(control_embeddings, doc_embeddings)
        
        found = []
        missing = []
        evidence = {}
        heatmap_data = []

        for i, control in enumerate(required_controls):
            max_score, max_idx = torch.max(cosine_scores[i], dim=0)
            score = max_score.item()
            
            heatmap_data.append({"Control Objective": control, "Cosine Similarity": score})
            
            if score > 0.50:
                found.append(control)
                # Store the exact paragraph that satisfied the requirement
                evidence[control] = chunks[max_idx.item()]
            else:
                missing.append(control)
                
        self.state['found_controls'] = found
        self.state['missing_controls'] = missing
        self.state['evidence'] = evidence
        self.state['heatmap_data'] = heatmap_data
        time.sleep(0.5)

# --- UI Layout ---
st.title("Argus AI: Policy Compliance Auditor")
st.markdown("Upload a policy document to run the compliance audit.")

if not using_finetuned:
    st.warning(
        f"Running on the public `{FALLBACK_SYNTHESIS_MODEL}` checkpoint: no fine-tuned "
        f"synthesis model was found at `{SYNTHESIS_MODEL_PATH}/`. Summary wording will "
        "differ from the reported results; every other stage is unaffected. "
        "See the README to train or supply a checkpoint."
    )

with st.sidebar:
    st.header("Pipeline")
    
    st.markdown("**Stage 1:** Triage Router (MNLI)")
    st.caption("Zero-shot classification to autonomously determine the regulatory domain.")
    
    st.markdown("**Stage 2:** Semantic Similarity (MiniLM)")
    st.caption("Bi-encoder scoring with supporting-chunk extraction.")
    
    st.markdown("**Stage 3:** Synthesis (BART)")
    st.caption("Generative model synthesising an executive summary.")

    st.markdown("**Stage 4:** Remediation Lookup")
    st.caption("Returns the pre-written policy text for each failed control.")
    
    st.markdown("**Course:** CSE398 Deep and Generative learning Project")
    st.divider()
    st.markdown("**AI Use Section:** Generative AI was utilised to brainstorm test cases for the semantic validation loops.")

uploaded_file = st.file_uploader("Upload Security Document (.pdf)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting and normalising PDF text..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        raw_text = ""
        for page in doc[:5]:
            raw_text += page.get_text()
        document_text = clean_pdf_text(raw_text)
        
    st.success("PDF Extracted Successfully!")

    if st.button("Run Audit Pipeline", type="primary"):
        
        orchestrator = PipelineOrchestrator()
        terminal = st.empty()
        
        orchestrator.execute_node_1_triage(document_text, terminal)
        orchestrator.execute_node_2_semantic_audit(document_text, terminal)
        
        domain = orchestrator.state['domain']
        found_controls = orchestrator.state['found_controls']
        missing_controls = orchestrator.state['missing_controls']
        evidence = orchestrator.state['evidence']
        heatmap_data = orchestrator.state['heatmap_data']
        
        st.divider()

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Stage 3: Executive Synthesis")
            st.caption("Live streaming generative summary based on established context.")
            summary_placeholder = st.empty()
            
            input_text = f"Context: {domain}. {document_text}"
            inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
            
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                input_ids=inputs["input_ids"],
                max_length=128,
                min_length=30,
                num_beams=1,
                no_repeat_ngram_size=2,
                repetition_penalty=2.0,
                early_stopping=True,
                streamer=streamer
            )
            
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            summary = ""
            for new_text in streamer:
                summary += new_text
                summary_placeholder.info(summary + " |")
            
            summary_placeholder.info(summary)
            
            st.subheader("Stage 4: Policy Remediation")
            st.caption("Pre-written remediation text for each missing control.")
            if missing_controls:
                st.error("MISSING CONTROLS DETECTED. SUGGESTED POLICY TEXT BELOW.")
                time.sleep(0.5) 
                for ctrl in missing_controls:
                    patch = REMEDIATION_TEMPLATES.get(ctrl, "Standard baseline security controls must be implemented.")
                    st.markdown(f"""
                    <div style='background-color: #2b1c1c; padding: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 12px; border-radius: 3px;'>
                        <span style='color: #ff4b4b; font-family: monospace; font-size: 14px; font-weight: bold;'>FAILED CONTROL: {ctrl.upper()}</span><br><br>
                        <span style='color: #00ffcc; font-family: monospace; font-size: 12px; font-weight: bold;'>SUGGESTED POLICY TEXT:</span><br>
                        <span style='color: #d1d1d1;'>{patch}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("SYSTEM HARDENED. No missing controls detected.")

        with col2:
            st.subheader("Control Similarity Scores")
            st.caption("Cosine similarity between each control and its closest passage (threshold 0.50).")
            
            df = pd.DataFrame(heatmap_data)
            st.dataframe(
                df,
                column_config={
                    "Cosine Similarity": st.column_config.ProgressColumn(
                        "Confidence Score", format="%.2f", min_value=0, max_value=1
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.subheader("RAG Evidence Extractor")
            st.caption("The passage that scored highest for each matched control.")
            
            # UPDATED UI LOGIC TO HANDLE EMPTY STATES
            if found_controls:
                for ctrl in found_controls:
                    st.markdown(f"""
                    <div style='background-color: #1a231f; padding: 15px; border-left: 5px solid #33a02c; margin-bottom: 12px; border-radius: 3px;'>
                        <span style='color: #33a02c; font-family: monospace; font-size: 14px; font-weight: bold;'>VERIFIED: {ctrl.upper()}</span><br><br>
                        <span style='color: #a0a0a0; font-style: italic;'>"...{evidence[ctrl]}..."</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No controls met the >0.50 verification threshold. Zero verifiable evidence extracted.")