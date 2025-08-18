import os, csv, random, io
from pathlib import Path
import sentencepiece as spm

#PATH
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/kapampangan_english.csv"
ONMT = ROOT / "data" / "onmt"
SP   = ROOT / "data" / "sp"
ONMT.mkdir(parents=True, exist_ok=True)
SP.mkdir(parents=True, exist_ok=True)

#Load Rows

rows = []
with io.open(CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        src = (r["kapampangan"] or "").strip()
        tgt = (r["english"] or "").strip()
        if src and tgt:
            rows.append((src, tgt))
            
#Train test split
random.seed(42)
random.shuffle(rows)
n = len(rows)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)
train = rows[:n_train]
valid = rows[n_train:n_train+n_valid]
test  = rows[n_train+n_valid:]

def write_pairs(pairs, prefix):
    with io.open(ONMT / f"{prefix}.kap", "w", encoding="utf-8") as s, \
         io.open(ONMT / f"{prefix}.eng", "w", encoding="utf-8") as t:
        for a,b in pairs:
            s.write(a + "\n")
            t.write(b + "\n")

write_pairs(train, "train")
write_pairs(valid, "valid")
write_pairs(test,  "test")

print(f"Rows: {n} | train={len(train)} valid={len(valid)} test={len(test)}")

VOCAB_SIZE = 8000
spm.SentencePieceTrainer.Train(
    input=','.join(str(ONMT / f"{p}.kap") for p in ["train","valid"]) + "," +
          ','.join(str(ONMT / f"{p}.eng") for p in ["train","valid"]),
    model_prefix=str(SP / "spm_shared"),
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,  # keeps accented chars
    input_sentence_size=2000000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=False,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)

print("Trained SentencePiece at:", SP / "spm_shared.model")