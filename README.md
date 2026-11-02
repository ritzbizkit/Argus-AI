# Argus AI — Local Policy Compliance Auditor

Argus AI reads a security policy document and reports, control by control,
whether the document provides evidence for the obligations expected in its
regulatory domain — showing the passage that supports each match, so a reviewer
can check the result rather than take it on trust.

It runs entirely on local inference. That constraint shaped the design: the
intended users audit sensitive network topologies and cannot send policy text to
a hosted model.

Hosted demo on public base models:
**[huggingface.co/spaces/rittude/Argus-AI](https://huggingface.co/spaces/rittude/Argus-AI)**

## How it works

Upload a policy PDF and four stages run in sequence:

| Stage | What it does | How |
| --- | --- | --- |
| 1. Triage | Picks which regulatory domain the document belongs to | Zero-shot classification (`facebook/bart-large-mnli`) |
| 2. Audit | Scores each expected control against the document and extracts the passage that best supports it | Bi-encoder cosine similarity (`all-MiniLM-L6-v2`), threshold 0.50 |
| 3. Synthesis | Writes an executive summary of the findings | Seq2seq generation, streamed to the UI |
| 4. Remediation | Returns the policy text that should be added for each control that failed | Lookup against a fixed template per control |

Stage 2 runs before stage 3 deliberately: the summary is written against
controls already matched to specific passages, rather than the model being asked
to judge compliance itself.

Measured end to end at roughly 4.9 seconds on an RTX 4080.

## Scope and limitations

What this prototype does and does not do:

- **The control set is hand-written, not parsed from the standards.** Twelve
  control objectives — four each for GDPR, HIPAA and NIST — are defined in
  `app.py`. Documents are checked against those twelve statements, not against
  the full text of NIST SP 800-53, GDPR or HIPAA.
- **Stage 4 is a lookup, not generation.** Remediation text is pre-written per
  control and returned verbatim.
- **Matching is semantic similarity, not entailment.** A high cosine score means
  a passage is topically close to a control statement. It is not proof the
  policy satisfies that control, and the 0.50 threshold has not been calibrated
  against labelled data.
- **The stages run in a fixed linear sequence.** There is no branching or
  conditional routing.
- **The fine-tuning corpus is small and its references are mostly
  LLM-generated.** See *Corpus* below.
- Only the first five pages of an uploaded PDF are read.

## Running it

```bash
git clone https://github.com/ritzbizkit/Argus-AI
cd Argus-AI
pip install -r requirements.txt
streamlit run app.py
```

That works from a fresh clone. The three models are pulled from Hugging Face on
first run (several GB; expect a wait).

### The fine-tuned synthesis checkpoint

`train.py` produces `argus_v2_bart/`, which `.gitignore` excludes because the
weights exceed GitHub's file size limit. They are not distributed here.

When the directory is absent, stage 3 falls back to `facebook/bart-large-cnn`
and the app shows a notice. Every other stage is unaffected; only the wording of
the summary changes. To use a checkpoint, place it at `argus_v2_bart/` in the
project root — it is picked up on the next launch.

## Corpus

```
data/raw/         source regulatory PDFs
data/processed/   extracted and normalised text  (extract_text.py → clean_text.py)
data/summaries/   reference summaries
```

Rebuild with:

```bash
python extract_text.py   # data/raw → data/processed
python clean_text.py     # normalise data/processed in place
```

**Current state: six document/summary pairs**, one per source PDF.

Two corpus bugs were fixed and are worth knowing about if you rebuild:

- **NIST SP 800-53 was missing.** The PDF and its summary were both present but
  extraction had never produced text for it, so the pair silently did not
  exist — the largest source document was absent from the corpus. `corpus.py`
  now reports unpaired files on every training and evaluation run, and
  `extract_text.py` exits non-zero on failure rather than looking like a clean
  run.
- **The GDPR text was corrupted.** Extraction with `pypdf` inserted spaces
  inside words — "Ar ticle", "Ha ving regar d", "Committ ee" — affecting
  thousands of tokens and inflating that document from its true 55,790 words to
  69,786. Extraction now uses PyMuPDF, which reads it correctly and matches the
  extractor `app.py` already uses at runtime.

Six pairs is still far too few to fine-tune a model of this size and get a
result that generalises. `train.py` refuses to run below twenty pairs unless
passed `--allow-tiny-dataset`, and warns when it proceeds.

### Scaling the corpus

Two documents dominate it: NIST SP 800-53 (~198k words) and the GDPR (~56k).
Both are already divided into sections — NIST into twenty control families, the
GDPR into eleven chapters — so splitting on those boundaries yields 35 coherent
sources without inventing anything:

```bash
python segment_corpus.py          # -> data/sections/ (35 sections)
python summary_worklist.py --init # -> data/summary_worklist.md
# write the summaries in that file, then:
python summary_worklist.py        # progress
python summary_worklist.py --apply
```

Boundaries come from each family's catalogue heading
(`3.17 SYSTEM AND SERVICES ACQUISITION Quick link to ...`), which is unique per
family — a bare control ID is not, because every control's "Related Controls"
list cites IDs from across the document. Every section is checked to be
dominated by its own family.

### Where the reference summaries come from

Nothing in these documents is a summary of a section: NIST's `Discussion:`
blocks elaborate on a control rather than condense it, and GDPR article titles
run about five words against a ~37-word reference style. So the references had
to be authored.

**31 of the 35 are LLM-generated, not written by a person.** They were drafted
by Claude Opus 5 from each section's control/article inventory on 2026-11-02.
The remaining 4 are the author's own, written for the original
document-level corpus.

This matters for how any result should be read:

- Fine-tuning on these targets is **distillation**. The model cannot be
  expected to exceed the quality of the model that produced them.
- Scoring a model against them measures agreement with that generator, not
  with a human reference. A model trained on these and scored against these
  will look better than it is.

The split is recorded per summary in `data/summaries/provenance.tsv`, and
`corpus.py` prints it with a warning on every training and evaluation run, so
it cannot be lost by accident. Replacing a generated summary with a written one
is a matter of editing the file and changing its row to `human`.

## Evaluation

`evaluate.py` scores summarisation against the reference summaries with ROUGE,
on the same documents for every system, so results are comparable.

```bash
python evaluate.py                       # lead-3 extractive baseline
python evaluate.py --system model        # ./argus_v2_bart
python evaluate.py --system model --model-path facebook/bart-large-cnn
```

```bash
python evaluate.py --processed-dir data/sections   # the 35-section corpus
```

Lead-3 extractive baseline:

| Corpus | n | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- | --- |
| Document-level (`data/processed`) | 6 | 0.2364 | 0.0697 | 0.1643 |
| Section-level (`data/sections`) | 35 | 0.2930 | 0.0711 | 0.2095 |

Any summarisation model is only worth using if it beats the baseline on the
same corpus. Note that 31 of the 35 section references are LLM-generated, so
the section-level figures measure agreement with that generator — see *Where
the reference summaries come from*.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit app and pipeline orchestration |
| `extract_text.py` | PDF → text (PyMuPDF), `data/raw` → `data/processed` |
| `clean_text.py` | Normalise the extracted corpus in place |
| `text_processing.py` | Cleaning, normalisation and chunking helpers |
| `corpus.py` | Pairs documents with summaries; reports unpaired files |
| `argus_dataset.py` | Torch dataset over those pairs |
| `train.py` | Fine-tuning loop for the synthesis model |
| `evaluate.py` | ROUGE scoring for the baseline and a checkpoint |
| `segment_corpus.py` | Splits NIST and the GDPR into 35 section-level sources |
| `summary_worklist.py` | Prepares and ingests the reference summaries |
| `baseline_lead3.py` | Prints baseline summaries (scoring lives in `evaluate.py`) |
| `tests/` | Unit tests for the text, corpus and baseline layers |

## Development

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

Tests, the baseline evaluation and a compile check run on every push via GitHub
Actions (`.github/workflows/ci.yml`).

## Context

Built for CSE398 Deep and Generative Learning at Lehigh University.

Generative AI was used to brainstorm test cases for the semantic validation
loops.
