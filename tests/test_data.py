import os
import numpy as np
from src.data.tokenizer import CharTokenizer
from src.data.prepare import PROCESSED_DIR, prepare_datasets

def test_tokenizer_roundtrip():
    sample = "ROMEO:\nSpeak, my lord!\n"
    tok = CharTokenizer.train_from_text(sample)
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    assert decoded == sample
    assert tok.vocab_size == len(set(sample))

def test_processed_files_exist_and_valid():
    if not os.path.exists(os.path.join(PROCESSED_DIR, "meta.pkl")):
        prepare_datasets()

    tok = CharTokenizer.load(os.path.join(PROCESSED_DIR, "meta.pkl"))
    assert tok.vocab_size == 65

    t25 = np.memmap(os.path.join(PROCESSED_DIR, "train_25.bin"), dtype=np.uint16, mode="r")
    t50 = np.memmap(os.path.join(PROCESSED_DIR, "train_50.bin"), dtype=np.uint16, mode="r")
    t100 = np.memmap(os.path.join(PROCESSED_DIR, "train_100.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(PROCESSED_DIR, "val.bin"), dtype=np.uint16, mode="r")
    test = np.memmap(os.path.join(PROCESSED_DIR, "test.bin"), dtype=np.uint16, mode="r")

    assert len(t25) == int(len(t100) * 0.25)
    assert len(t50) == int(len(t100) * 0.50)
    assert len(val) > 0 and len(test) > 0
    assert t100.max() < tok.vocab_size
