# Argus AI: Deterministic Security Compliance Engine

## Project Description
Argus AI is a multi-agent orchestration framework designed to eliminate hallucination in generative cybersecurity auditing. By utilizing a Directed Acyclic Graph (DAG) state manager, the system bounds generative synthesis within deterministic, mathematical semantic entailment loops. 

The pipeline ingests enterprise regulatory documents (e.g., GDPR, HIPAA, NIST) and executes through four nodes:
1. **Triage Router (BART-Large-MNLI):** Zero-shot classification to autonomously determine the regulatory domain.
2. **Semantic Auditor (all-MiniLM-L6-v2):** Dual-encoder verification mapping cybersecurity terminology into high-dimensional vector space to establish mathematical proof of compliance (Threshold > 0.50).
3. **Synthesis Stream (Custom argus_v2_bart):** Generates executive summaries strictly constrained to the verified RAG context.
4. **Remediation Agent:** Autonomously drafts compliant policy patches for failed controls.

The architecture is designed for 100% local inference, ensuring air-gapped data privacy for sensitive corporate network topologies, achieving end-to-end execution in ~4.9 seconds on an RTX 4080.

## Data Source
The intelligence of the Argus AI system relies on a hybrid data pipeline:
* **Pre-Trained Base Corpuses:** Node 1 leverages the Multi-Genre Natural Language Inference (MNLI) dataset (433,000+ sentence pairs) for zero-shot routing. Node 2 utilizes a native training corpus of over 1 billion sentence pairs for semantic vectorization.
* **Custom Fine-Tuning Dataset:** The Node 3 generative synthesis model (`argus_v2_bart`) was fine-tuned on a custom, curated corpus of open-source regulatory documentation, including parsed sections of NIST SP 800-53, GDPR legal text, and HIPAA safeguard guidelines, paired with synthetic edge-case summaries.
* **Note on Cloud Deployment:** A lightweight version of this app using standard base models is hosted at [https://huggingface.co/spaces/rittude/Argus-AI](https://huggingface.co/spaces/rittude/Argus-AI) due to cloud storage limits. 

## Packages Required
The following deep learning and UI libraries are required to run the local inference engine. They are also included in the `requirements.txt` file:
* `streamlit`
* `torch`
* `transformers`
* `sentence-transformers`
* `pandas`
* `PyMuPDF`

## Instructions on How to Run the Code
To execute the true architecture locally with the custom fine-tuned models:

1. **Clone the Repository:** Download or clone this repository to your local machine.
2. **Download Custom Model Weights:** The custom fine-tuned models exceed GitHub's 100MB file limit. You must download the `argus_v1` and `argus_v2_bart` folders from the following secure drive: 
   [INSERT YOUR GOOGLE DRIVE / BOX LINK HERE]
3. **Place the Models:** Extract and place both the `argus_v1` and `argus_v2_bart` folders directly into the root directory of this project.
4. **Install Dependencies:** Open your terminal in the project directory and run:
   `pip install -r requirements.txt`
5. **Launch the Application:** Initialize the dashboard by running:
   `streamlit run main.py`
