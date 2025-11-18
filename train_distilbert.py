# train_distilbert.py
import os
import argparse
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# =========================
# Repro
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)


# =========================
# Label mapping for cross-domain test
# 0=Fake, 1=Real
# =========================
_FAKE = {"pants-fire", "false", "barely-true", "half-true", "fake", "f", "0"}
_REAL = {"true", "mostly-true", "real", "t", "1"}


def map_test_label(x) -> int:
    s = str(x).strip().lower()
    if s in _REAL:
        return 1
    if s in _FAKE:
        return 0
    return 0  # default to fake if unknown


# =========================
# Dataset
# =========================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=192):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


# =========================
# Column pickers for test CSV
# =========================
_GEN_LABEL_TOKENS = {
    "true", "false", "pants-fire", "barely-true", "half-true", "mostly-true",
    "real", "fake", "0", "1"
}


def _pick_text_column(df: pd.DataFrame) -> str:
    # Prefer text-like object columns with high average length and decent uniqueness
    candidates = []
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str)
            avg_len = s.str.len().dropna().mean()
            uniq = s.nunique(dropna=True) / max(len(s), 1)
            if avg_len >= 20 and uniq > 0.3:
                candidates.append((avg_len, col))
    if not candidates:
        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].astype(str)
                candidates.append((s.str.len().dropna().mean(), col))
    if not candidates:
        raise ValueError("❌ Could not find a suitable TEXT column in the test file.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def _pick_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        s = df[col].astype(str).str.lower().str.strip()
        if s.isin(_GEN_LABEL_TOKENS).mean() >= 0.05:
            return col
    for col in ["label", "target", "class", "isfake", "verdict"]:
        if col in df.columns:
            return col
    return None


# =========================
# Metrics: Calibration & Robustness
# =========================
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error (ECE) for binary classification.
    Uses max confidence per sample: c = max(p0, p1) and correctness indicator.
    """
    conf = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = np.where((conf > lo) & (conf <= hi))[0]
        if len(idx) == 0:
            rows.append([i, lo, hi, 0, np.nan, np.nan])
            continue
        acc_bin = correct[idx].mean()
        conf_bin = conf[idx].mean()
        gap = abs(acc_bin - conf_bin)
        weight = len(idx) / len(y_true)
        ece += weight * gap
        rows.append([i, lo, hi, len(idx), acc_bin, conf_bin])
    df_bins = pd.DataFrame(rows, columns=["bin", "lo", "hi", "count", "acc", "conf"])
    return float(ece), df_bins


def compute_overconfidence(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    conf = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    acc = (preds == y_true).mean()
    return float(conf.mean() - acc)  # >0 means overconfident


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Use prob of the true class
    p_true = y_prob[np.arange(len(y_true)), y_true]
    return float(np.mean((1.0 - p_true) ** 2))


def length_quartile_groups(texts: List[str]) -> np.ndarray:
    lengths = np.array([len(str(t)) for t in texts])
    q = np.quantile(lengths, [0.25, 0.5, 0.75])
    groups = np.digitize(lengths, q, right=True)  # 0..3
    return groups


def worst_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    stats = {}
    accs = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        acc = accuracy_score(y_true[idx], y_pred[idx])
        accs.append((g, acc, len(idx)))
    if not accs:
        return {"wga": None, "bga": None, "gap": None, "per_group": []}
    accs_sorted = sorted(accs, key=lambda x: x[1])
    wga = accs_sorted[0][1]
    bga = accs_sorted[-1][1]
    gap = bga - wga
    return {
        "wga": float(wga),
        "bga": float(bga),
        "gap": float(gap),
        "per_group": [{"group": int(g), "acc": float(a), "n": int(n)} for g, a, n in accs_sorted],
    }


def save_confusion(cm: np.ndarray, labels: List[str], out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(5.2, 4.4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels)
    plt.yticks(tick, labels)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str) -> Optional[float]:
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.figure(figsize=(5.2, 4.4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return float(auc)
    except Exception:
        return None


# =========================
# Train / Eval loops
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, use_amp=True):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(**batch)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def forward_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, probs = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        y = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        p = F.softmax(logits, dim=1).cpu().numpy()
        pred = p.argmax(axis=1)
        ys.extend(y.tolist())
        preds.extend(pred.tolist())
        probs.extend(p.tolist())
    return np.array(ys), np.array(preds), np.array(probs)


def pack_metrics(
    name: str,
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
) -> Dict[str, Any]:
    """Compute & save all metrics/plots for one split."""
    os.makedirs(out_dir, exist_ok=True)
    # Base metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Calibration & robustness
    ece, ece_bins = compute_ece(y_true, y_prob, n_bins=15)
    overconf = compute_overconfidence(y_true, y_prob)
    brier = compute_brier(y_true, y_prob)

    # AUROC (may be None if only one class present)
    auc = save_roc(y_true, y_prob, os.path.join(out_dir, "roc_curve.png"), f"{name} ROC")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    save_confusion(cm, ["Fake(0)", "Real(1)"], os.path.join(out_dir, "confusion_matrix.png"), f"{name} Confusion Matrix")

    # Classification report
    rep_txt = classification_report(y_true, y_pred, target_names=["Fake(0)", "Real(1)"])
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt)
    rep_dict = classification_report(y_true, y_pred, target_names=["Fake(0)", "Real(1)"], output_dict=True)
    pd.DataFrame(rep_dict).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))

    # Save ECE bins
    ece_bins.to_csv(os.path.join(out_dir, "calibration_bins.csv"), index=False)

    # Length-quartile robustness
    groups = length_quartile_groups(texts)
    wga = worst_group_accuracy(y_true, y_pred, groups)

    # Summary JSON
    summary = {
        "split": name,
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "ece": float(ece),
        "overconfidence": float(overconf),
        "brier": float(brier),
        "auroc": None if auc is None else float(auc),
        "wga": wga,
    }
    pd.DataFrame([summary]).to_json(os.path.join(out_dir, "summary.json"), orient="records", indent=2)
    return summary


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT on combined_news.csv and evaluate cross-domain test with rich metrics."
    )
    parser.add_argument("--train_file", required=True, help="e.g., data/combined_news.csv (must have columns: text,label)")
    parser.add_argument("--test_file", required=True, help='e.g., data/test (1).csv or data/train (3).csv')
    parser.add_argument("--output_dir", default="models/distilbert_generalized")
    parser.add_argument("--epochs", default=2, type=int)  # per your request, default 2
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_len", default=192, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--use_amp", action="store_true", help="Enable CUDA AMP")
    args = parser.parse_args()

    model_dir = args.output_dir
    art_train = "artifacts"
    art_test = "artifacts_test"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(art_train, exist_ok=True)
    os.makedirs(art_test, exist_ok=True)

    # ---------- Load TRAIN ----------
    print("✅ Loading training dataset...")
    train_df = pd.read_csv(args.train_file, on_bad_lines="skip")
    if not {"text", "label"}.issubset(train_df.columns):
        raise ValueError("❌ Train file must contain columns: 'text' and 'label' (0=Fake, 1=Real).")
    train_df = train_df.dropna(subset=["text", "label"])
    train_df["label"] = train_df["label"].astype(int)
    print("✅ Training samples:", len(train_df))

    # ---------- Load TEST (cross-domain) ----------
    print("\n✅ Loading external test dataset...")
    raw_test = pd.read_csv(args.test_file, on_bad_lines="skip", engine="python")
    txt_col = _pick_text_column(raw_test)
    lbl_col = _pick_label_column(raw_test)
    if lbl_col is None:
        raise ValueError("❌ No label column found! Need a verdict-like column (true/false/half-true/...).")

    print(f"✅ Using TEXT column: {txt_col}")
    print(f"✅ Using LABEL column: {lbl_col}")

    test_df = raw_test[[txt_col, lbl_col]].rename(columns={txt_col: "text", lbl_col: "label"})
    test_df.dropna(subset=["text", "label"], inplace=True)
    test_df["label"] = test_df["label"].apply(map_test_label)
    print("✅ Test samples:", len(test_df))

    # ---------- Model & Tokenizer ----------
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)
    model.to(device)

    # ---------- Dataloaders ----------
    train_ds = NewsDataset(train_df["text"], train_df["label"], tokenizer, args.max_len)
    test_ds = NewsDataset(test_df["text"], test_df["label"], tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=SequentialSampler(test_ds))

    # ---------- Optimizer / Scheduler / AMP ----------
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * max(args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    # ---------- Train ----------
    history = []
    print(f"\n✅ Training for {args.epochs} epoch(s) ...\n")
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, use_amp=args.use_amp)
        print(f"✅ Epoch {ep}/{args.epochs} — Loss: {loss:.4f}")
        history.append({"epoch": ep, "loss": loss})
    pd.DataFrame(history).to_csv(os.path.join(art_train, "training_history.csv"), index=False)

    # ---------- Evaluate TRAIN ----------
    y_tr, yhat_tr, p_tr = forward_all(model, train_loader, device)
    train_summary = pack_metrics("Training", list(train_df["text"]), y_tr, yhat_tr, p_tr, art_train)

    # ---------- Evaluate TEST ----------
    print("\n✅ Evaluating on cross-domain dataset...\n")
    y_te, yhat_te, p_te = forward_all(model, test_loader, device)
    test_summary = pack_metrics("Cross-Domain", list(test_df["text"]), y_te, yhat_te, p_te, art_test)

    # ---------- Robustness deltas ----------
    acc_drop = float(train_summary["accuracy"] - test_summary["accuracy"])
    auroc_drop = None
    if (train_summary["auroc"] is not None) and (test_summary["auroc"] is not None):
        auroc_drop = float(train_summary["auroc"] - test_summary["auroc"])

    # Save simple accuracies for the app
    pd.DataFrame(
        [
            {"metric": "train_accuracy", "value": train_summary["accuracy"]},
            {"metric": "test_accuracy", "value": test_summary["accuracy"]},
        ]
    ).to_csv(os.path.join(art_test, "accuracies.csv"), index=False)

    # Save a combined summary JSON
    combined = {
        "train": train_summary,
        "test": test_summary,
        "robustness": {
            "accuracy_drop_train_to_test": acc_drop,
            "auroc_drop_train_to_test": auroc_drop,
        },
    }
    pd.Series(combined).to_json(os.path.join(art_test, "combined_summary.json"), indent=2)

    # ---------- Save model ----------
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("\n✅ DONE. Model & ALL artifacts saved.")
    print(f"   • Model dir: {model_dir}")
    print(f"   • Training artifacts: {art_train}")
    print(f"   • Test artifacts: {art_test}")
    print(f"   • Accuracy (train): {train_summary['accuracy']:.4f}")
    print(f"   • Accuracy (test):  {test_summary['accuracy']:.4f}")
    print(f"   • Macro-F1 (test):  {test_summary['macro_f1']:.4f}")
    print(f"   • AUROC (test):     {test_summary['auroc']}")
    print(f"   • ECE (test):       {test_summary['ece']:.4f}")
    print(f"   • WGA (test):       {test_summary['wga']['wga'] if test_summary['wga']['wga'] is not None else 'N/A'}")


if __name__ == "__main__":
    main()
