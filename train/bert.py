"""Fine-tune multilingual DistilBERT on both languages at once: a plain torch loop on CPU
(about an hour for 40,000 rows and 3 epochs; free Colab if that is too slow)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sentiment.score import BERT_DIR, LABELS, MAX_LEN

BASE = "distilbert/distilbert-base-multilingual-cased"
log = logging.getLogger(__name__)


def finetune(
    df: pd.DataFrame,
    out_dir: Path,
    epochs: int = 3,
    batch: int = 32,
    lr: float = 5e-5,
    seed: int = 0,
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    enc = tok(
        df["text"].tolist(),
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    y = torch.tensor([LABELS.index(lab) for lab in df["label"]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y),
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=epochs * len(loader)
    )
    model.train()
    for epoch in range(epochs):
        for step, (ids, mask, labels) in enumerate(loader):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            if step % 100 == 0:
                log.info("epoch %d step %d/%d loss %.3f", epoch + 1, step, len(loader), loss.item())
    model.eval()
    model.save_pretrained(out_dir / BERT_DIR)
    tok.save_pretrained(out_dir / BERT_DIR)
