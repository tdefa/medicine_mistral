import re
from typing import List

# ───────── sentence utilities ────────────────────────────────────────────────
def _sentences(text: str) -> List[str]:
    """
    Split *text* into sentences using a simple rule:
    cut after '.', '!' or '?' that are followed by whitespace + capital/digit.
    """
    splitter = re.compile(r"""
        (?<=[.!?])          # keep the punctuation
        \s+                 # whitespace that follows it
        (?=[A-Z0-9])        # next sentence likely starts with capital/digit
    """, re.VERBOSE)
    clean = re.sub(r"\s+", " ", text.strip())
    return splitter.split(clean)

def _chunk_sentences(sentences: List[str], n: int) -> List[str]:
    """
    Group consecutive sentences into chunks whose total word‑count ≤ n.
    Never slices a sentence; if a single sentence already exceeds n words
    it becomes its own chunk.
    """
    chunks, buf, buf_len = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if buf and buf_len + w > n:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks

# ───────── main routine ──────────────────────────────────────────────────────
def chunk_by_section(raw: str, n: int) -> List[str]:
    """
    Extract each <section> … </section> block, split its body into ≤ n‑word
    sentence chunks, and prefix every chunk with the <section name>.
    """
    sec_rx = re.compile(
        r"<section>\s*<section name>(.*?)</section name>(.*?)(?=<section>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    result: List[str] = []
    for m in sec_rx.finditer(raw):
        title = m.group(1).strip()

        # Body text minus any closing </section> tag  ←― NEW LINE
        body = re.sub(r"</section\s*>", " ", m.group(2), flags=re.IGNORECASE)

        # Turn body into sentence chunks
        sentences = _sentences(body)
        for chunk in _chunk_sentences(sentences, n):
            result.append(f"{title}: {chunk.strip()}")
    return result

# ───────── example/demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    <section> <section name>indications and usage</section name>
    INDICATIONS AND USAGE Herpes Zoster Infections: Acyclovir tablets, USP are
    indicated for the acute treatment of herpes zoster (shingles). Genital
    Herpes: Acyclovir tablets, USP are indicated for the treatment of initial
    episodes and the management of recurrent episodes of genital herpes.
    Chickenpox: Acyclovir tablets, USP are indicated for the treatment of
    chickenpox (varicella).</section>

    <section> <section name>contraindications</section name>
    CONTRAINDICATIONS Acyclovir is contraindicated for patients who develop
    hypersensitivity to acyclovir or valacyclovir.</section>
    """

    list_chunks = chunk_by_section(sample, n=30)
    for i, c in enumerate(chunk_by_section(sample, n=30), 1):
        print(f"[{i}] {c}\n")
