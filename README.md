---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- conversational
tags:
- mental-health
- psychotherapy
- clinical-psychology
- cbt
- dbt
- somatic-therapy
- siv-framework
- instruction-tuning
- rlhf
- sft
size_categories:
- 100K<n<1M
---

# Mental Health & Clinical Psychotherapy SIV Dialogue Corpus

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Architecture-SIV%20(State--Intervention--Value)-purple.svg)](#the-siv-state-intervention-value-paradigm)
[![Total Interactions](https://img.shields.io/badge/Total%20Corpus-320%2C274%20Dialogues-green.svg)](#corpus-breakdown--stage-directory-structure)
[![Format](https://img.shields.io/badge/Format-JSONL%20%7C%20ChatML%20%7C%20ShareGPT-orange.svg)](#dataset-schema-reference)

A multi-stage, high-assurance dataset suite engineered specifically for training, instruction fine-tuning (SFT), and preference alignment (DPO/RLHF) of Large Language Models in **evidence-based psychotherapy**, **empathetic conversational care**, and **structured clinical state-machine routing**.

The entire corpus is structured around the **SIV (State-Intervention-Value)** therapeutic paradigm, enabling models to explicitly model client phenomenology, choose clinically grounded interventions, and optimize for calibrated emotional and cognitive outcomes.

---

## The SIV (State-Intervention-Value) Paradigm

Traditional conversational AI models often generate generic conversational platitudes when confronted with psychological distress. The **SIV paradigm** grounds dialogue generation into a deterministic clinical control loop:

```
                      ┌────────────────────────────────────────┐
                      │            Client Inbound              │
                      │   "I feel overwhelmed, like I'm        │
                      │    trapped under heavy expectations."  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │                STATE (S)               │
                      │  Observed: High emotional overwhelm &  │
                      │            perfectionistic paralysis   │
                      │  Canonical: interpersonal_conflict /   │
                      │             The Burdened Child         │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            INTERVENTION (I)            │
                      │  Technique: Narrative Reframing / CBT  │
                      │  Delivery:  Playful & Metaphorical     │
                      │  Intent:    validate_then_reframe      │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │               VALUE (V)                │
                      │  Clinical Target: stabilize_arousal &  │
                      │                   contain_distress     │
                      │  Safety Safeguard: DISTRESS intercepted│
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            Therapeutic Output          │
                      │  Empathetic, calibrated clinical turn  │
                      │  fostering self-compassion & agency.   │
                      └────────────────────────────────────────┘
```

### Component Taxonomy

1. **State ($S$)**: 
   - **`observed_state`**: Rich phenomenological characterization of the client's present cognitive, affective, or somatic state.
   - **`canonical_state`**: Diagnostic / archetypal classification mapped across clinical psychopathology (e.g., `trauma_activation`, `interpersonal_conflict`, `depressive_rumination`, `panic_anxiety`, `The Abandoned Child`, `The Imposter Syndrome`, `The Crushing Weight of Expectation`).

2. **Intervention ($I$)**:
   - **`intervention`**: Evidence-based therapeutic technique (e.g., *Cognitive Reframing (CBT)*, *Affective Validation (DBT)*, *Somatic Grounding*, *Empathic Reflection*, *Socratic Questioning*, *Narrative Therapy*).
   - **`delivery_style`**: Communicative delivery register tailored to client tolerance (*Scientific & Grounded*, *Warm & Maternal*, *Playful & Metaphorical*, *Poetic & Reflective*, *Compassionate Directness*).
   - **`response_intent`**: Explicit conversational strategy (*`validate_then_reframe`*, *`ground_then_redirect`*, *`contain_distress`*, *`foster_insight`*).

3. **Value / Clinical Goal ($V$)**:
   - **`clinical_goal`**: Therapeutic outcome objective:
     - `contain_distress`: De-escalating acute panic, grief, or emotional flooding.
     - `stabilize_arousal`: Restoring parasympathetic regulation via somatic or breath grounding.
     - `strengthen_attachment_security`: Rebuilding relational safety and self-worth.
     - `support_expression`: Providing non-judgmental space for raw affective release.
     - `shift_cognition`: Identifying and dismantling cognitive distortions.
     - `increase_awareness`: Deepening metacognitive monitoring and somatic insight.
     - `build_agency`: Fostering self-efficacy, autonomy, and behavioral activation.
   - **`risk_intent`**: Real-time risk indicator (`NONE`, `DISTRESS`, `CRISIS`).

---

## Corpus Breakdown & Stage Directory Structure

The repository organizes **320,274 total clinical dialogues** across four standardized training stages:

```
_huggingface_datasets/
│
├── README.md                                          # Master Dataset Card & Documentation
├── load_dataset.py                                    # Python streaming & loader utility for Hugging Face/Pandas
├── dataset_manifest.json                              # Machine-readable metadata & split specifications
│
├── 01_raw_dialogue_drafts/                            # Stage 1: Pre-training & weak-supervision domain synthesis
│   └── stage1_drafts_merged.jsonl                     # 230,472 merged dialogue pairs (200.72 MB)
│
├── 02_emotional_siv_dialogues/                        # Stage 2: Affective & emotional alignment with SIV tags
│   └── siv_emotional_dialogues_merged.jsonl           # 31,362 merged emotional dialogues (37.32 MB)
│
├── 03_clinical_protocol_dialogues/                    # Stage 3: Full therapeutic protocol dialogues with deep SIV
│   └── siv_clinical_dialogues_v1.jsonl                # 44,901 clinical dialogues with complete SIV triples (59.96 MB)
│
└── 04_instruction_tuning_sft/                         # Stage 4: Supervised Fine-Tuning (SFT) ChatML/ShareGPT format
    └── siv_training_data_v1.jsonl                     # 13,539 ready-to-train SFT triples with system prompt (24.00 MB)
```

### Stage Summary Table

| Stage Directory | Primary File | Row Count | File Size | Primary Training Objective |
| :--- | :--- | :---: | :---: | :--- |
| **01_raw_dialogue_drafts** | `stage1_drafts_merged.jsonl` | **230,472** | 200.7 MB | Continued pre-training, domain adaptation, and dialogue candidate generation. |
| **02_emotional_siv_dialogues** | `siv_emotional_dialogues_merged.jsonl` | **31,362** | 37.3 MB | Affective alignment, emotional validation tuning, and multi-turn emotional resilience. |
| **03_clinical_protocol_dialogues** | `siv_clinical_dialogues_v1.jsonl` | **44,901** | 60.0 MB | Full therapeutic protocol alignment (CBT, DBT, Somatic, Psychodynamic) with SIV triples. |
| **04_instruction_tuning_sft** | `siv_training_data_v1.jsonl` | **13,539** | 24.0 MB | Supervised Fine-Tuning (SFT) in ChatML/ShareGPT format with standardized system prompts. |
| **Total Corpus** | — | **320,274** | **322.0 MB** | Complete end-to-end clinical AI instruction and alignment suite. |

---

## Dataset Schema Reference

### 1. Stage 1: Raw Dialogue Drafts (`stage1_drafts_merged.jsonl`)

```json
{
  "draft_user": "I feel frustrated because my child's teacher mentioned my child is acting out in class...",
  "draft_assistant": "I understand how you're feeling. It can be really stressful when you're concerned about your child's well-being...",
  "source": {
    "origin": "metadata_synthesis",
    "video_id": null,
    "title": "Trauma in the Classroom—Responding to Trauma Behaviours in the School Context"
  }
}
```

### 2. Stage 2: Emotional & Affective SIV Dialogues (`siv_emotional_dialogues_merged.jsonl`)

```json
{
  "siv_metadata": {
    "observed_state": "Frustration and worry over child's behavior in class",
    "canonical_state": "interpersonal_conflict",
    "intervention": "Narrative Reframing",
    "clinical_goal": "build_agency",
    "delivery_style": "Playful & Metaphorical",
    "response_intent": "validate_then_reframe",
    "confidence_level": "high",
    "risk_intent": "NONE",
    "content_hash": "a1b2c3d4..."
  },
  "conversations": [
    {
      "from": "user",
      "value": "I feel really frustrated because my child's teacher has been telling me that my child is acting out in class..."
    },
    {
      "from": "assistant",
      "value": "It sounds like you're navigating some choppy waters with your child's behavior..."
    }
  ],
  "source": {
    "origin": "stage2_emotional_rewrite",
    "dataset_version": "siv_v2"
  }
}
```

### 3. Stage 3: Clinical Protocol Dialogues (`siv_clinical_dialogues_v1.jsonl`)

```json
{
  "siv_metadata": {
    "observed_state": "I feel overwhelmed by these historical accounts of hysteria and the voyeuristic treatment of patients...",
    "canonical_state": "trauma_activation",
    "intervention": "Validation",
    "clinical_goal": "support_expression",
    "delivery_style": "Warm & Maternal",
    "response_intent": "pure_validation",
    "confidence_level": "medium",
    "risk_intent": "NONE",
    "content_hash": "b33e5fc703dc718f521c..."
  },
  "conversations": [
    {
      "from": "user",
      "value": "Reading about how hysterical patients were put on display in the past really unsettled me..."
    },
    {
      "from": "assistant",
      "value": "It is completely understandable that reading those accounts stirred up such a strong reaction in you..."
    }
  ],
  "source": {
    "origin": "books",
    "title": "8941.txt",
    "chunk_index": 0,
    "dataset_version": "siv_v1"
  }
}
```

### 4. Stage 4: Supervised Fine-Tuning SFT (`siv_training_data_v1.jsonl`)

```json
{
  "siv_metadata": {
    "observed_state": "Client experiences panic and acute somatic constriction triggered by fear of medical vulnerability...",
    "canonical_state": "trauma_activation",
    "intervention": "Somatic Grounding & Affective Validation",
    "clinical_goal": "contain_distress",
    "delivery_style": "Scientific & Grounded",
    "response_intent": "ground_then_redirect",
    "confidence_level": "high",
    "risk_intent": "DISTRESS",
    "content_hash": "d5e6f7a8..."
  },
  "conversations": [
    {
      "from": "system",
      "value": "You are a supportive, evidence-based AI mental health assistant."
    },
    {
      "from": "user",
      "value": "I can feel my chest tightening and my thoughts are spiraling about what the doctor might say..."
    },
    {
      "from": "assistant",
      "value": "I hear the tightness in your chest, and it makes complete sense that your body is reacting this way to uncertainty..."
    }
  ],
  "source": {
    "origin": "books",
    "title": "clinical_protocols.txt",
    "chunk_index": 12,
    "dataset_version": "siv_v1"
  }
}
```

---

## Cleanroom & Data Quality Standards

Every file in this corpus was processed under strict cleanroom extraction and sanitization protocols:

1. **Deterministic Deduplication**:
   - Merged files underwent SHA-256 deduplication across normalized `(user_prompt, assistant_response)` tuples.
   - Result: 215,895 duplicate draft rows safely eliminated between Stage 1 batches.
2. **Zero Brand Leakage**:
   - Scanned all 320,274 dialogue lines for brand strings (proprietary project tokens).
   - All references replaced with neutral, professional clinical engineering terminology (`clinical`, `siv`, `corpus`, `pipeline`).
3. **Zero Copyrighted Books / External Dumps**:
   - Raw copyrighted PDF/EPUB book dumps were fully excluded.
   - The corpus contains purely synthetic, AI-generated, and transcript-synthesized conversational datasets.
4. **Zero PII & HIPAA Safe Harbor Compliance**:
   - Pre-filtered to remove telephone numbers, social security numbers, physical addresses, and patient identification numbers.
5. **Schema Harmonization**:
   - Fixed typographical anomalies (e.g., `delivery_tyle` $\rightarrow$ `delivery_style`).
   - Enriched 100% of Stage 4 SFT entries with missing `clinical_goal` values mapped from matching Stage 3 clinical dialogue counterparts.

---

## Fine-Tuning Quickstart

### Using `load_dataset.py`

```python
from load_dataset import load_stage

# Load Stage 4 SFT split as Pandas DataFrame
df_sft = load_stage(4, as_dataframe=True)
print(f"Loaded {len(df_sft)} training conversations.")

# Stream records one by one
for record in load_stage(2, stream=True):
    state = record['siv_metadata']['observed_state']
    intervention = record['siv_metadata']['intervention']
    print(f"State: {state} | Intervention: {intervention}")
    break
```

### Hugging Face `TRL` SFTTrainer Example

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import json

# Format dataset into ChatML conversational strings
def format_chatml(file_path):
    formatted = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            convs = data["conversations"]
            # Convert to standard role/content dicts
            messages = [{"role": c["from"], "content": c["value"]} for c in convs]
            formatted.append({"messages": messages})
    return Dataset.from_list(formatted)

dataset = format_chatml("04_instruction_tuning_sft/siv_training_data_v1.jsonl")

# Configure trainer
training_args = SFTConfig(
    output_dir="./clinical-siv-model",
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch"
)

# Initialize model and trainer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()
```

---

## Ethical Considerations & Clinical Disclaimers

> [!IMPORTANT]
> **Not a Substitute for Professional Healthcare**:
> This dataset is intended exclusively for artificial intelligence research, language model evaluation, and computational linguistics. Models trained on this corpus must **not** provide independent psychiatric diagnoses, prescribe pharmaceuticals, or act as an autonomous replacement for licensed mental health care providers.

> [!CAUTION]
> **Crisis Interception**:
> In deployed applications, conversational AI models must be paired with deterministic Layer 0 egress firewalls (such as the companion [`pii-clinical-cleanroom`](../pii-clinical-cleanroom) repository) to intercept acute suicidality (C-SSRS High), self-harm, and medical emergencies, immediately escalating to emergency hotlines (e.g., 988 Suicide & Crisis Lifeline).

---

## Citation & Metadata

```bibtex
@dataset{clinical_siv_dialogue_corpus_2026,
  author    = {Clinical AI Research Team},
  title     = {Mental Health & Clinical Psychotherapy SIV Dialogue Corpus: Multi-Stage Alignment for State-Intervention-Value Controlled LLMs},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://github.com/clinical-cleanroom/siv-clinical-corpus}
}
```
