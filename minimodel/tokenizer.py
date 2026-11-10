"""Serializable character and UTF-8 byte tokenizers. ID 0 is unknown for characters."""

class Tokenizer:
    def __init__(self, mode="char", alphabet=None):
        if mode not in {"char", "byte"}:
            raise ValueError("tokenizer must be char or byte")
        self.mode = mode
        self.alphabet = list(alphabet or [])
        self.lookup = {c: i + 1 for i, c in enumerate(self.alphabet)}

    @classmethod
    def fit(cls, text, mode="char"):
        return cls(mode, sorted(set(text)) if mode == "char" else [])

    @property
    def vocab_size(self):
        return 256 if self.mode == "byte" else len(self.alphabet) + 1

    def encode(self, text):
        return list(text.encode("utf-8")) if self.mode == "byte" else [self.lookup.get(c, 0) for c in text]

    def decode(self, ids):
        if self.mode == "byte":
            return bytes(ids).decode("utf-8", errors="replace")
        return "".join(self.alphabet[i - 1] if i else "�" for i in ids)

    def state(self):
        return {"mode": self.mode, "alphabet": self.alphabet}
