# 🔍 LLM Response Reliability Evaluation Pipeline

A lightweight, production-oriented evaluation pipeline to automatically assess the **reliability of LLM-generated responses** in real-time.

This project was built as part of the **BeyondChats Internship Assignment** and is designed with scalability, interpretability, and cost-efficiency in mind.

---

## 🚀 Why This Project?

Large Language Models (LLMs) can generate fluent responses, but fluency alone does not guarantee:
- relevance to the user query,
- completeness of the answer,
- or factual correctness (hallucinations).

In real-world products, especially in **high-risk domains like healthcare**, it is critical to **systematically evaluate LLM outputs** before trusting them.

This pipeline addresses exactly that problem.

---

## 📌 What This Pipeline Evaluates

For every **User → AI response pair**, the pipeline computes:

### 1️⃣ Response Relevance  
Measures how semantically aligned the AI response is with the user query.

### 2️⃣ Response Completeness  
Checks whether the response sufficiently covers the key aspects of the user’s question.

### 3️⃣ Hallucination / Factual Accuracy  
Identifies unsupported or hallucinated claims by verifying each response sentence against retrieved context.

### 4️⃣ Latency  
Estimates response generation latency using message timestamps.

### 5️⃣ Cost  
Estimates token-based inference cost using configurable pricing.

---

## 🧠 High-Level Architecture

```bash
Chat JSON ──┐
├─▶ Evaluation Pipeline ──▶ evaluation_report.json
Context JSON ─┘

```

### **Key Design Choice:**  
The pipeline evaluates **only the context vectors actually used by the RAG system**, ensuring fair and accurate hallucination detection.

---


## 🏗️ Repository Structure

```
llm-evaluation-pipeline/
│
├── src/
│ └── evaluator.py # Core evaluation logic
│
├── data/
│ ├── chat1.json # Chat conversation (sample 1)
│ ├── context1.json # Vector DB context (sample 1)
│ ├── chat2.json # Chat conversation (sample 2)
│ ├── context2.json # Vector DB context (sample 2)
│
├── requirements.txt
├── README.md
└── .gitignore

```

---


## ⚙️ Local Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/llm-evaluation-pipeline.git
cd llm-evaluation-pipeline

```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```
###3️⃣ Run the Evaluation

```bash
python src/evaluator.py

```

📄 Output will be generated as:

```bash
evaluation_report.json

```

---


## 📊 Evaluation Methodology

### 🔹 Relevance
Uses a lightweight CrossEncoder model

Produces a normalized score ∈ [0, 1]

### 🔹 Completeness
Keyword coverage heuristic

Optimized for speed and interpretability

Suitable for large-scale real-time evaluation

### 🔹 Hallucination Detection
Sentence-level verification

Each sentence is checked for entailment against retrieved context

Implemented using a Natural Language Inference (NLI) model

⚠️ The hallucination metric is intentionally conservative to avoid false negatives in sensitive domains.

### 🔹 Latency & Cost
Latency derived from timestamps

Cost estimated using token counts and configurable per-token pricing

---


## 📈 Example Output

```
{
  "dataset_id": 1,
  "turn_id": 14,
  "metrics": {
    "relevance": 0.92,
    "completeness": 0.66,
    "faithfulness": 0.0,
    "latency_sec": 9.0,
    "cost_usd": 0.000043
  },
  "hallucinated_sentences": [
    "We also offer specially subsidized rooms at our clinic."
  ]
}

```

---

## ⚖️ Design Decisions & Trade-offs

### Why not use an LLM to evaluate another LLM?
High latency

High operational cost

Circular dependency

### Why sentence-level hallucination detection?
Identifies exact unsupported claims

More actionable for debugging and monitoring

Commonly used in production trust & safety systems

### Why heuristic completeness instead of generative scoring?
Faster

Deterministic

Scales to millions of evaluations per day

---

## 🚀 Scalability & Production Readiness
This pipeline is designed to scale efficiently:

❌ No external API calls

✅ Lightweight models suitable for batch inference

✅ Stateless evaluation → easy horizontal scaling

✅ Deterministic metrics for monitoring dashboards

---

##📝 Notes
Low faithfulness scores do not necessarily indicate poor responses — they indicate missing or weak grounding in retrieved context.

This conservative behavior is intentional and desirable for safety-critical applications.

---

##👤 Author

Ishika Dubey
Applied for Part-Time Internship at BeyondChats
