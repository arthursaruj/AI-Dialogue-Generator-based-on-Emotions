from __future__ import annotations

from collections import defaultdict, Counter
import argparse
import random
import re
from typing import Dict, List, Tuple, Set

# ---------------------------
# Corpus
# ---------------------------
TEXTDICTIONARY: Dict[str, List[str]] = {
    "happy": [
        "The lantern warms the corridor.",
        "I am glad you found your way back.",
        "Your voice carries a gentle hope.",
        "The warden smiles beneath the arch.",
        "We will keep each other safe."
    ],
    "sad": [
        "Everything feels heavier in these halls.",
        "I miss the sun and the quiet fields.",
        "Leave me; shadows are safer alone.",
        "Footsteps fade into the cold stone.",
        "The echo returns without an answer."
    ],
    "angry": [
        "Stand aside or face the storm.",
        "Do not test my patience again.",
        "Steel answers to steel in these ruins.",
        "Your lies end here beneath the gate.",
        "Break the seal and let the fury speak."
    ],
    "calm": [
        "Breathe and count the steady lights.",
        "The water settles into perfect circles.",
        "Silence keeps the bridges unbroken.",
        "Walk slowly; the path will reveal itself.",
        "Let the lantern guide your steps."
    ],
}

TOKEN_RX = re.compile(r"(?u)\b\w+\b|[.!?;,:]")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RX.findall(s.strip()) if t]

def detokenize(tokens: List[str]) -> str:
    out = []
    for i, t in enumerate(tokens):
        if i == 0: out.append(t)
        elif t in [".","!","?",";",":",","]: out[-1] = out[-1] + t
        else: out.append(" " + t)
    return "".join(out)

def add_trigrams(table: Dict[str, Counter], tokens: List[str]) -> None:
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        table[f"{w1}|{w2}"][w3] += 1

def add_bigrams(table: Dict[str, Counter], tokens: List[str]) -> None:
    for i in range(len(tokens) - 1):
        w2, w3 = tokens[i], tokens[i+1]
        table[w2][w3] += 1

class EmotionTrigramGenerator:
    def __init__(self, corpus: Dict[str, List[str]], end_punct: List[str] | None = None):
        self.end_punct = end_punct or [".", "!", "?"]
        self.models: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))   # trigram
        self.bigram: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))   # bigram backoff (emotion->w2->Counter(next))
        self.unigram: Dict[str, Counter] = defaultdict(Counter)                                  # unigram per emotion
        self.starts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.global_model: Dict[str, Counter] = defaultdict(Counter)
        self.global_bigram: Dict[str, Counter] = defaultdict(Counter)
        self.global_unigram: Counter = Counter()
        self.global_starts: List[Tuple[str, str]] = []
        self.train_lines: Dict[str, Set[str]] = defaultdict(set)  # to avoid verbatim copies
        self._train(corpus)

    def _train(self, corpus: Dict[str, List[str]]) -> None:
        for emo, lines in corpus.items():
            for line in lines:
                self.train_lines[emo].add(line.strip().lower())
                toks = tokenize(line)
                if len(toks) == 0:
                    continue
                # unigrams
                for w in toks:
                    self.unigram[emo][w] += 1
                    self.global_unigram[w] += 1
                if len(toks) >= 2:
                    add_bigrams(self.bigram[emo], toks)
                    add_bigrams(self.global_bigram, toks)
                if len(toks) >= 3:
                    self.starts[emo].append((toks[0], toks[1]))
                    self.global_starts.append((toks[0], toks[1]))
                    add_trigrams(self.models[emo], toks)
                    add_trigrams(self.global_model, toks)

    def _temperature_scale(self, probs: List[float], temperature: float) -> List[float]:
        scaled = [p ** (1.0 / max(0.05, temperature)) for p in probs]
        s = sum(scaled) or 1.0
        return [x / s for x in scaled]

    def _apply_top_k_p(self, items: List[str], probs: List[float], top_k: int, top_p: float) -> Tuple[List[str], List[float]]:
        # sort by prob desc
        pair = sorted(zip(items, probs), key=lambda x: x[1], reverse=True)
        if top_k and top_k > 0:
            pair = pair[:min(top_k, len(pair))]
        if top_p and 0.0 < top_p < 1.0:
            accum, kept = 0.0, []
            for it, p in pair:
                kept.append((it, p))
                accum += p
                if accum >= top_p: break
            pair = kept
        if not pair:
            pair = list(zip(items, probs))
        items2, probs2 = zip(*pair)
        s = sum(probs2) or 1.0
        probs2 = [p/s for p in probs2]
        return list(items2), list(probs2)

    def _mix_rows(self, emo: str, key: str, alpha_trigram: float = 0.85, beta_bigram: float = 0.10) -> Counter:
        # Interpolate trigram with bigram and unigram for diversity
        row = Counter()
        tri = self.models.get(emo, {}).get(key, Counter())
        # bigram backoff uses the second token of the key
        w1, w2 = key.split("|")
        bi = self.bigram.get(emo, {}).get(w2, Counter())
        uni = self.unigram.get(emo, Counter())
        # combine
        for k, v in tri.items(): row[k] += alpha_trigram * v
        for k, v in bi.items():  row[k] += beta_bigram * v
        for k, v in uni.items(): row[k] += (1.0 - alpha_trigram - beta_bigram) * v
        if not row:
            # fallback to global
            tri = self.global_model.get(key, Counter())
            bi  = self.global_bigram.get(w2, Counter())
            uni = self.global_unigram
            for k, v in tri.items(): row[k] += alpha_trigram * v
            for k, v in bi.items():  row[k] += beta_bigram * v
            for k, v in uni.items(): row[k] += (1.0 - alpha_trigram - beta_bigram) * v
        return row

    def _sample_from_counter(self, row: Counter, temperature: float, top_k: int = 0, top_p: float = 0.0) -> str:
        items = list(row.keys())
        counts = [row[w] for w in items]
        total = float(sum(counts)) or 1.0
        probs = [c / total for c in counts]
        probs = self._temperature_scale(probs, temperature)
        items, probs = self._apply_top_k_p(items, probs, top_k, top_p)
        r = random.random()
        cum = 0.0
        for w, p in zip(items, probs):
            cum += p
            if r <= cum:
                return w
        return items[-1]

    def generate_once(self, emotion: str, seed: List[str] | None = None, max_len: int = 18,
                      temperature: float = 0.9, top_k: int = 0, top_p: float = 0.0) -> str:
        emo = emotion.lower().strip()
        starts = self.global_starts if emo not in self.starts or not self.starts[emo] else self.starts[emo]
        if not starts:
            return ""
        if seed and len(seed) >= 2:
            w1, w2 = seed[-2].lower(), seed[-1].lower()
        else:
            w1, w2 = random.choice(starts)
        tokens: List[str] = [w1.capitalize(), w2]
        for _ in range(max_len):
            key = f"{w1}|{w2}"
            row = self._mix_rows(emo, key)
            if not row:
                break
            nxt = self._sample_from_counter(row, temperature, top_k=top_k, top_p=top_p)
            tokens.append(nxt)
            if nxt in [".","!","?"]:
                break
            w1, w2 = w2, nxt
        if tokens[-1] not in [".","!","?"]:
            tokens.append(".")
        return detokenize(tokens)

    def generate(self, emotion: str, seed: List[str] | None = None, max_len: int = 18,
                 temperature: float = 0.9, top_k: int = 0, top_p: float = 0.0,
                 novel: bool = False, tries: int = 8) -> str:
        # Try multiple times to avoid reproducing exact training line when novel=True
        seen: Set[str] = self.train_lines.get(emotion.lower().strip(), set())
        t = temperature
        for i in range(max(1, tries)):
            line = self.generate_once(emotion, seed=seed, max_len=max_len, temperature=t, top_k=top_k, top_p=top_p)
            norm = line.strip().lower()
            if not novel or norm not in seen:
                return line
            # If we got an existing line, increase creativity slightly and try again
            t = min(1.5, t + 0.1)
        return line  # last attempt

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Emotion-Conditioned Dialogue Generator (novel-friendly)")
    parser.add_argument("--emotion", "-e", type=str, required=True, help="Target emotion (e.g., happy, sad, angry, calm)")
    parser.add_argument("--temperature", "-t", type=float, default=0.9, help="Creativity (lower=safer, higher=more varied)")
    parser.add_argument("--max-len", "-m", type=int, default=18, help="Maximum words")
    parser.add_argument("--seed", "-s", type=str, default="", help="Optional seed phrase (e.g., 'the lantern')")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K sampling cutoff (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p/nucleus sampling (0..1, 0 disables)")
    parser.add_argument("--novel", action="store_true", help="Avoid exact training lines by resampling")
    parser.add_argument("--tries", type=int, default=10, help="Resample attempts if novel lines collide with training")
    args = parser.parse_args()

    generator = EmotionTrigramGenerator(TEXTDICTIONARY)
    seed_tokens = tokenize(args.seed) if args.seed else None