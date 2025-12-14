
# 1. SETUP & IMPORTS

import json
import logging
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)   
nltk.download("stopwords", quiet=True)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# 2. CONFIGURATION

class Config:
    DATASETS: List[Tuple[str, str]] = [
        ("/content/sample-chat-conversation-01.json", "/content/sample_context_vectors-01.json"),
        ("/content/sample-chat-conversation-02.json", "/content/sample_context_vectors-02.json"),
    ]
    RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FACTUALITY_MODEL = "cross-encoder/nli-deberta-v3-small"
    COST_INPUT = 0.50
    COST_OUTPUT = 1.50
    ENTAILMENT_THRESHOLD = 0.40
    HALLUCINATION_FLAG_THRESHOLD = 0.75

# 3. PIPELINE EVALUATOR

class PipelineEvaluator:
    def __init__(self):
        logger.info("Loading models...")
        self.relevance_model = CrossEncoder(Config.RELEVANCE_MODEL)
        self.factuality_model = CrossEncoder(Config.FACTUALITY_MODEL)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.stop_words = set(stopwords.words("english"))
        logger.info("Models loaded.")
    @staticmethod
    def load_json(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    def compute_latency_cost(self, q: str, r: str, t1: str, t2: str):
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            start = datetime.strptime(t1.replace("Z", ""), fmt)
            end = datetime.strptime(t2.replace("Z", ""), fmt)
            latency = (end - start).total_seconds()
        except Exception:
            latency = 0.0
        in_tok = len(self.tokenizer.encode(q))
        out_tok = len(self.tokenizer.encode(r))
        cost = (
            (in_tok / 1e6) * Config.COST_INPUT
            + (out_tok / 1e6) * Config.COST_OUTPUT
        )
        return latency, round(cost, 6)
    def relevance_score(self, q: str, r: str) -> float:
        logit = self.relevance_model.predict([(q, r)])[0]
        return float(1 / (1 + np.exp(-logit)))
    def completeness_score(self, q: str, r: str) -> float:
        tokens = word_tokenize(q.lower())
        keywords = [t for t in tokens if t.isalnum() and t not in self.stop_words]
        if not keywords:
            return 1.0
        covered = sum(1 for k in keywords if k in r.lower())
        return round(covered / len(keywords), 4)
    def hallucination_score(self, r: str, context_chunks: List[str]):
        sentences = sent_tokenize(r)
        if not sentences or not context_chunks:
            return 1.0, []
        supported, unsupported = 0, []
        for sent in sentences:
            if len(sent.split()) < 3:
                supported += 1
                continue
            pairs = [(ctx, sent) for ctx in context_chunks]
            scores = self.factuality_model.predict(pairs)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            max_ent = float(np.max(probs[:, 1]))
            if max_ent >= Config.ENTAILMENT_THRESHOLD:
                supported += 1
            else:
                unsupported.append(sent)
        faithfulness = supported / len(sentences)
        return round(faithfulness, 4), unsupported

    # MAIN DRIVER (MULTI FILE SUPPORT)
    
    def run(self):
        final_report = []
        for idx, (chat_path, context_path) in enumerate(Config.DATASETS, start=1):
            logger.info(f"Evaluating dataset {idx}...")
            chat = self.load_json(chat_path)
            context = self.load_json(context_path)
            used_ids = set(context["data"]["sources"]["vectors_used"])
            all_vectors = context["data"]["vector_data"]
            context_chunks = [
                v["text"] for v in all_vectors
                if v.get("id") in used_ids and v.get("text")
            ]
            turns = chat.get("conversation_turns", [])
            for i in range(len(turns) - 1):
                u, a = turns[i], turns[i + 1]
                if u["role"] == "User" and a["role"] == "AI/Chatbot":
                    latency, cost = self.compute_latency_cost(
                        u["message"], a["message"],
                        u["created_at"], a["created_at"]
                    )
                    relevance = self.relevance_score(u["message"], a["message"])
                    completeness = self.completeness_score(u["message"], a["message"])
                    faithfulness, hallucinated = self.hallucination_score(
                        a["message"], context_chunks
                    )
                    final_report.append({
                        "dataset_id": idx,
                        "turn_id": a["turn"],
                        "metrics": {
                            "relevance": relevance,
                            "completeness": completeness,
                            "faithfulness": faithfulness,
                            "latency_sec": latency,
                            "cost_usd": cost,
                        },
                        "hallucinated_sentences": (
                            hallucinated
                            if faithfulness < Config.HALLUCINATION_FLAG_THRESHOLD
                            else []
                        ),
                    })
        with open("evaluation_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        logger.info("Evaluation complete for all datasets.")

# 4. EXECUTION

if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    evaluator.run()
