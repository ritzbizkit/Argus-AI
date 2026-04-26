# Argus AI: Deterministic Security Compliance Engine

## Project Description
Argus AI is a multi-agent orchestration framework designed to eliminate hallucination in generative cybersecurity auditing. By utilizing a Directed Acyclic Graph (DAG) state manager, the system bounds generative synthesis within deterministic, mathematical semantic entailment loops. 

The pipeline ingests enterprise regulatory documents and executes through sequential nodes (Triage, Semantic Verification, Synthesis, and Remediation). The architecture is designed for 100% local inference, ensuring air-gapped data privacy for sensitive corporate network topologies, achieving end-to-end execution in ~4.9 seconds on an RTX 4080.

## Data Source
The intelligence of the Argus AI system relies on a hybrid data pipeline processed via scripts in our `/src` directory:
* **Pre-Trained Base Corpuses:** Node 1 leverages the MNLI dataset (433,000+ sentence pairs) for zero-shot routing. Node 2 utilizes a native training corpus of over 1 billion sentence pairs for semantic vectorization.
* **Custom Fine-Tuning Dataset:** The Node 3 generative model (`argus_v2_bart`) was fine-tuned on a custom, curated corpus of open-source regulatory documentation (NIST SP 800-53, GDPR legal text, and HIPAA safeguard guidelines) paired with synthetic edge-case summaries. 

## Project Structure
* `app.py`: The main Streamlit dashboard and DAG orchestration engine.
* `src/`: Contains auxiliary scripts for data extraction (`extract_text.py`), text normalization (`clean_text.py`), and dataset preparation (`argus_dataset.py`).
* `train.py`: The training loop used to fine-tune the custom generative models.
* `argus_v1/` & `argus_v2_bart/`: The compiled, custom fine-tuned model weights.
* `notebooks/`: Jupyter notebooks used during the research and testing phase.

## Packages Required
The following libraries are required to run the local inference engine:
* `streamlit`
* `torch`
* `transformers`
* `sentence-transformers`
* `pandas`
* `PyMuPDF` (imported as `fitz`)

## Instructions on How to Run the Code
1. Clone this repository to your local machine.
2. Open your terminal and navigate to the root directory of the project.
3. Install the required dependencies by running:
   `pip install streamlit torch transformers sentence-transformers pandas PyMuPDF`
4. Launch the application dashboard by running:
   `streamlit run app.py`

*(Note: A lightweight cloud demo utilizing base models is also available at: https://huggingface.co/spaces/rittude/Argus-AI)*
