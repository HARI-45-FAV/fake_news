import os, argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

from train_dann_bert import DANNModel, MODEL_NAME, MAX_LEN, MODELS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="CSV with columns: text,label")
parser.add_argument("--outdir", default="images")
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.csv)
if "text" not in df.columns:
    # Try to auto-pick a sensible column
    for col in ["content","full_description","title","headline"]:
        if col in df.columns:
            df["text"] = df[col]
            break

df["text"] = df["text"].fillna("").astype(str)
labels = df["label"].astype(int).tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
class EvalDS(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

loader = DataLoader(EvalDS(df["text"].tolist()), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DANNModel(MODEL_NAME).to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "dann_distilbert.pt"), map_location=device))
model.eval()

probs_all = []
with torch.no_grad():
    for ids, att in loader:
        ids, att = ids.to(device), att.to(device)
        logits, _ = model(ids, att, alpha=0.0)
        probs = torch.softmax(logits, dim=1)[:,1]
        probs_all.extend(probs.cpu().tolist())

import numpy as np
preds = (np.array(probs_all) >= args.threshold).astype(int)

rep = classification_report(labels, preds, target_names=["Real","Fake"], digits=4)
print(rep)
with open(os.path.join(args.outdir, "eval_report.txt"), "w") as f:
    f.write(rep)

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (CSV)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "eval_confusion.png"))
print("Saved metrics to", args.outdir)
