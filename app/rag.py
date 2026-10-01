from pathlib import Path
import re

import faiss
from sentence_transformers import SentenceTransformer


class PolicyRAG:

    def __init__(self, policy_dir="data/hr_policy_documents"):

        self.policy_dir = Path(policy_dir)

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chunks = []
        self.index = None

        self._build_index()

    # =========================================================
    # POLICY TOPIC KEYWORDS
    # =========================================================

    POLICY_KEYWORDS = {

        "resignation_policy.txt": [
            "resignation",
            "resign",
            "resigning",
            "notice period",
            "notice",
            "quit",
            "leaving company",
            "leaving the company",
            "termination"
        ],

        "leave_policy.txt": [
            "leave",
            "pto",
            "paid time off",
            "vacation",
            "sick leave",
            "casual leave",
            "maternity",
            "paternity",
            "parental leave",
            "leave balance",
            "carryover"
        ],

        "attendance_policy.txt": [
            "attendance",
            "absent",
            "absence",
            "late",
            "tardy",
            "tardiness",
            "attendance record",
            "job abandonment"
        ],

        "work_from_home_policy.txt": [
            "work from home",
            "wfh",
            "remote work",
            "remote",
            "home working",
            "working from home"
        ],

        "reimbursement_policy.txt": [
            "reimbursement",
            "expense",
            "expenses",
            "claim",
            "travel expense",
            "business expense",
            "expense claim"
        ],

        "it_policy.txt": [
            "it",
            "computer",
            "laptop",
            "password",
            "security",
            "cybersecurity",
            "software",
            "hardware",
            "device",
            "technical",
            "information security"
        ],

        "performance_policy.txt": [
            "performance",
            "performance review",
            "appraisal",
            "evaluation",
            "promotion",
            "performance improvement",
            "rating"
        ],

        "payroll_policy.txt": [
            "payroll",
            "salary",
            "pay",
            "wage",
            "payment",
            "salary credited",
            "salary credit",
            "payslip",
            "pay slip",
            "deduction",
            "bonus"
        ],
    }

    # =========================================================
    # CREATE POLICY SECTIONS
    # =========================================================

    def _chunk_text(self, text):

        lines = text.splitlines()

        sections = []

        current_section = None
        current_text = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Example:
            # 1. ANNUAL PAID TIME OFF (PTO)

            heading = re.match(
                r"^(\d+)\.\s+(.+)$",
                line
            )

            # Example:
            # SECTION 1
            # Section 1: Attendance

            section_heading = re.match(
                r"^(?:SECTION|Section)\s+(\d+)\s*:?\s*(.*)$",
                line
            )

            if heading:

                # Save previous section
                if current_section and current_text:

                    sections.append({
                        "section": current_section,
                        "text": " ".join(current_text)
                    })

                current_section = (
                    f"{heading.group(1)}. "
                    f"{heading.group(2)}"
                )

                current_text = []

            elif section_heading:

                if current_section and current_text:

                    sections.append({
                        "section": current_section,
                        "text": " ".join(current_text)
                    })

                number = section_heading.group(1)
                title = section_heading.group(2).strip()

                current_section = (
                    f"Section {number}"
                    + (f": {title}" if title else "")
                )

                current_text = []

            else:

                current_text.append(line)

        # Save final section
        if current_section and current_text:

            sections.append({
                "section": current_section,
                "text": " ".join(current_text)
            })

        # -----------------------------------------------------
        # FALLBACK FOR DOCUMENTS WITHOUT HEADINGS
        # -----------------------------------------------------

        if not sections:

            words = text.split()

            size = 250
            overlap = 50

            start = 0

            while start < len(words):

                part = " ".join(
                    words[start:start + size]
                )

                if part.strip():

                    sections.append({
                        "section": "Not explicitly numbered",
                        "text": part
                    })

                start += max(
                    1,
                    size - overlap
                )

        return sections

    # =========================================================
    # BUILD FAISS INDEX
    # =========================================================

    def _build_index(self):

        for file in sorted(
            self.policy_dir.glob("*.txt")
        ):

            text = file.read_text(
                encoding="utf-8"
            )

            sections = self._chunk_text(text)

            for i, section in enumerate(sections):

                self.chunks.append({

                    "text": section["text"],

                    "document": file.name,

                    "section": section["section"],

                    "chunk_id": i,
                })

        if not self.chunks:

            raise RuntimeError(
                "No HR policy documents found."
            )

        embeddings = self.model.encode(

            [c["text"] for c in self.chunks],

            normalize_embeddings=True,

            show_progress_bar=False,

        ).astype("float32")

        # Normalized vectors + Inner Product
        # = cosine similarity

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    # =========================================================
    # DETECT POLICY TOPIC
    # =========================================================

    def _detect_policy_documents(self, query):

        query_lower = query.lower()

        matches = {}

        for document, keywords in self.POLICY_KEYWORDS.items():

            score = 0

            for keyword in keywords:

                if keyword in query_lower:

                    # Longer phrases are more specific,
                    # therefore give them more weight.

                    if len(keyword.split()) > 1:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                matches[document] = score

        return matches

    # =========================================================
    # SEARCH
    # =========================================================

    def search(self, query, k=3):

        q = self.model.encode(

            [query],

            normalize_embeddings=True,

            show_progress_bar=False,

        ).astype("float32")

        # Retrieve more candidates first.
        # This allows us to apply policy-topic boosting.

        candidate_k = min(
            max(k * 4, 10),
            len(self.chunks)
        )

        scores, ids = self.index.search(
            q,
            candidate_k
        )

        topic_matches = self._detect_policy_documents(
            query
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0]
        ):

            if idx < 0:
                continue

            chunk = self.chunks[idx]

            semantic_score = float(score)

            topic_boost = 0.0

            document = chunk["document"]

            # -------------------------------------------------
            # BOOST DOCUMENT IF QUERY CLEARLY MATCHES ITS TOPIC
            # -------------------------------------------------

            if document in topic_matches:

                # Strong enough to overcome a small semantic
                # difference between documents.

                topic_boost = min(
                    0.15,
                    topic_matches[document] * 0.05
                )

            final_score = semantic_score + topic_boost

            results.append({

                **chunk,

                "semantic_score": semantic_score,

                "topic_boost": topic_boost,

                "score": final_score
            })

        # Highest final score first

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:k]

    # =========================================================
    # EXTRACT RELEVANT SENTENCES
    # =========================================================

    def _get_relevant_sentences(
        self,
        query,
        text,
        max_sentences=2
    ):

        query_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                query.lower()
            )
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        scored = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    sentence.lower()
                )
            )

            overlap = len(
                query_words & sentence_words
            )

            scored.append(
                (overlap, sentence)
            )

        # Highest word overlap first

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = []

        for score, sentence in scored:

            if score > 0:

                selected.append(sentence)

            if len(selected) >= max_sentences:
                break

        # If no direct word overlap exists,
        # return the first sentence.

        if not selected and sentences:

            selected.append(
                sentences[0].strip()
            )

        return selected

    # =========================================================
    # ANSWER
    # =========================================================

    def answer(self, query, k=3):

        results = self.search(
            query,
            k
        )

        if not results:

            return {

                "answer": (
                    "I could not find a relevant "
                    "HR policy."
                ),

                "sources": []
            }

        top = results[0]

        relevant_sentences = (
            self._get_relevant_sentences(
                query,
                top["text"]
            )
        )

        answer = " ".join(
            relevant_sentences
        )

        return {

            "answer": answer,

            "sources": [

                {

                    "document": r["document"],

                    "section": r["section"],

                    "score": round(
                        r["score"],
                        4
                    ),

                }

                for r in results
            ],
        }