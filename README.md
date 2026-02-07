# Argus AI: Transformer-Based Security Summarization

## Project Overview
**Argus AI** is an "all-seeing" generative deep learning system designed to distill complex, multi-page security documentation (NIST, ISO, GDPR) into actionable executive insights. Named after the hundred-eyed giant of myth, Argus leverages modern Transformer architectures to ensure that no critical security control is overlooked, regardless of document length.

## Project Goals
* **Abstractive Summarization:** Moving beyond simple keyword extraction to generate human-like executive summaries.
* **Contextual Integrity:** Using Multi-Head Attention to link distant security requirements across long-form PDFs.
* **Modular Engineering:** A solo-developed pipeline from raw data ingestion to a functional web interface.

## Technical Architecture
Based on the **DSCI/CSE-498/398** Spring 2026 syllabus, Argus AI implements:
* **Encoder-Decoder Transformer:** For robust sequence-to-sequence text generation.
* **Multi-Head Self-Attention:** To simultaneously weight the importance of disparate security controls.
* **Sinusoidal Positional Encoding:** Utilizing interleaved sine and cosine functions to provide a unique coordinate system for every token without using a recursive loop.

## Project Structure
* `/data`: Curated security policies and compliance frameworks.
* `/notebooks`: Google Colab experiments for model prototyping.
* `/src`: Modular PyTorch implementations of the Transformer blocks.
* `app.py`: Streamlit web interface for CISO-level interactions.

