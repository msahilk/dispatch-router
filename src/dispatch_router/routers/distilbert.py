"""Fine-tuned DistilBERT binary classifier: does Haiku answer this prompt correctly?

Settings follow the PRD (max_len 256, batch 16, lr 2e-5, weight decay 0.01,
fixed seed, early stopping on validation) with two justified changes:
  * Left truncation: prompts put the instructions first and the actual question
    (or SQL query / code) last, so when a prompt is longer than 256 tokens we keep
    the end, not the boilerplate at the start.
  * Class-weighted loss: Haiku is right ~93% of the time, so an unweighted model
    can score well by always predicting "right". We weight the rare class and pick
    the checkpoint by validation AUROC; probabilities are fixed by calibration later.
"""

from __future__ import annotations

import copy
import os
import random

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class DistilBertRouter:
    name = "distilbert"

    def __init__(self, model_name="distilbert-base-uncased", max_len=256, batch_size=16, lr=2e-5,
                 epochs=4, weight_decay=0.01, seed=42):
        self.model_name, self.max_len, self.batch_size = model_name, max_len, batch_size
        self.lr, self.epochs, self.weight_decay, self.seed = lr, epochs, weight_decay, seed
        self.history: list[dict] = []

    def _setup(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)

    def _encode(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")

    def _predict(self, texts, bs=64) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = {k: v.to(self.device) for k, v in self._encode(texts[i:i + bs]).items()}
                out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out)

    def fit(self, X_tr, y_tr, X_val, y_val, texts_tr=None, texts_val=None, log=print):
        import torch
        from sklearn.metrics import log_loss, roc_auc_score
        from transformers import get_linear_schedule_with_warmup

        self._setup()
        texts_tr, texts_val = list(texts_tr), list(texts_val)
        y_tr_t = torch.tensor(np.asarray(y_tr, dtype=np.int64))
        n_pos, n_neg = int(y_tr_t.sum()), int(len(y_tr_t) - y_tr_t.sum())
        weight = torch.tensor([len(y_tr_t) / (2 * n_neg), len(y_tr_t) / (2 * n_pos)], dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * ((len(texts_tr) + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        g = torch.Generator().manual_seed(self.seed)

        best_auc, best_state = -1.0, None
        for ep in range(1, self.epochs + 1):
            self.model.train()
            perm = torch.randperm(len(texts_tr), generator=g).tolist()
            run_loss = 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                enc = {k: v.to(self.device) for k, v in self._encode([texts_tr[j] for j in idx]).items()}
                loss = loss_fn(self.model(**enc).logits, y_tr_t[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item() * len(idx)
            p_tr, p_val = self._predict(texts_tr), self._predict(texts_val)
            rec = {
                "epoch": ep,
                "train_weighted_loss": run_loss / len(texts_tr),
                "train_auroc": roc_auc_score(y_tr, p_tr),
                "val_auroc": roc_auc_score(y_val, p_val),
                "train_logloss": log_loss(y_tr, np.clip(p_tr, 1e-6, 1 - 1e-6)),
                "val_logloss": log_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)),
            }
            self.history.append(rec)
            log("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if rec["val_auroc"] > best_auc:
                best_auc, self.best_epoch = rec["val_auroc"], ep
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(best_state)
        return self

    def save(self, directory) -> None:
        self.model.save_pretrained(directory)
        self.tok.save_pretrained(directory)

    @classmethod
    def load(cls, directory):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        r = cls()
        r.device = "mps" if torch.backends.mps.is_available() else "cpu"
        r.tok = AutoTokenizer.from_pretrained(directory)
        r.tok.truncation_side = "left"
        r.model = AutoModelForSequenceClassification.from_pretrained(directory).to(r.device)
        return r

    def score(self, X=None, texts=None, is_train: bool = False) -> np.ndarray:
        return self._predict(list(texts))

    def describe(self) -> dict:
        return {"device": self.device, "best_epoch": self.best_epoch, "history": self.history,
                "max_len": self.max_len, "batch_size": self.batch_size, "lr": self.lr, "epochs": self.epochs,
                "truncation_side": "left", "loss": "class-weighted cross-entropy"}
