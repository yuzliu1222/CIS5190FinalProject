from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ORIGINAL: from model import NewsClassifier
# NEW: also import the baseline network (for comparison) and the counting helpers
from model import (
    CLASS_NAMES,
    MAX_TREE_NODES,
    NUM_BUCKETS,
    HashedTextClassifier,
    NewsClassifier,
    bucket_ids_to_counts,
    texts_to_bucket_ids,
)
# ORIGINAL: from preprocess import prepare_data
# NEW: also import the baseline cleaning (for the baseline comparison) and our new cleaning
from preprocess import _slug_to_text, prepare_data, url_to_text

# NEW imports
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  # same imports as hw3_skeleton.ipynb cell 70
from urllib.parse import urlparse
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _hash_token(token: str, mod: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def texts_to_bag_inputs(texts: Sequence[str], num_buckets: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build inputs for EmbeddingBag from a batch of texts.
    Returns:
      - token_ids: LongTensor of shape [N_tokens_in_batch]
      - offsets: LongTensor of shape [batch_size]
    """
    all_ids: List[int] = []
    offsets: List[int] = []
    running_total = 0
    for t in texts:
        offsets.append(running_total)
        ids = [_hash_token(tok, num_buckets) for tok in _simple_tokenize(t)] or [0]
        all_ids.extend(ids)
        running_total += len(ids)
    token_ids_t = torch.tensor(all_ids, dtype=torch.long)
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    return token_ids_t, offsets_t


def encode_labels(labels: Sequence[str]) -> Tuple[torch.Tensor, dict]:
    """
    Map labels to ints. Expected labels: 'FoxNews' and 'NBC'.
    """
    unique = sorted(list({str(l) for l in labels}))
    mapping = {lab: i for i, lab in enumerate(unique)}
    y = torch.tensor([mapping[str(l)] for l in labels], dtype=torch.long)
    return y, mapping


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum().item()
    return correct / max(1, targets.numel())


def train(
    model: NewsClassifier,
    texts_train: Sequence[str],
    y_train: torch.Tensor,
    texts_val: Sequence[str],
    y_val: torch.Tensor,
    *,
    device: str = "cpu",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> dict:
    # ORIGINAL baseline training loop. Now only used to retrain the baseline
    # network (HashedTextClassifier) on our split, for the comparison.
    # NEW: it also records and returns the per-epoch history, for the plots.
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}  # NEW

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_buckets = model.num_buckets

    n_train = len(texts_train)
    for epoch in range(1, epochs + 1):
        # shuffle
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]
            batch_texts = [texts_train[i] for i in idx]
            batch_y = y_train[idx]

            token_ids, offsets = texts_to_bag_inputs(batch_texts, num_buckets)
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            targets = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(token_ids, offsets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits(logits, targets)

            epoch_loss += loss.item()
            epoch_acc += acc
            steps += 1

        epoch_loss /= max(1, steps)
        epoch_acc /= max(1, steps)

        # Validation
        model.eval()
        with torch.no_grad():
            val_token_ids, val_offsets = texts_to_bag_inputs(texts_val, num_buckets)
            val_logits = model(val_token_ids.to(device), val_offsets.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_acc = accuracy_from_logits(val_logits, y_val.to(device))
        model.train()

        print(f"Epoch {epoch:02d} | train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        # NEW: remember this epoch's numbers
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

    return history  # NEW


# ===========================================================================
# NEW: helper functions for the new model
# ===========================================================================

def compute_metrics(pred_ids, true_ids):
    """Accuracy, confusion matrix, precision and recall.

    NBC (id 1) is treated as the "positive" class.
    Sources: confusion matrix and accuracy, L08_Logistic Regression+Logistic
    Regression.pdf p.12-14; precision and recall, L08 p.23-24.
    """
    pred_ids = np.array(pred_ids)
    true_ids = np.array(true_ids)
    tp = int(np.sum((pred_ids == 1) & (true_ids == 1)))
    fn = int(np.sum((pred_ids == 0) & (true_ids == 1)))
    fp = int(np.sum((pred_ids == 1) & (true_ids == 0)))
    tn = int(np.sum((pred_ids == 0) & (true_ids == 0)))

    metrics = {}
    metrics["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
    metrics["nbc_precision"] = tp / max(1, tp + fp)
    metrics["nbc_recall"] = tp / max(1, tp + fn)
    metrics["fox_precision"] = tn / max(1, tn + fn)
    metrics["fox_recall"] = tn / max(1, tn + fp)
    metrics["TP"] = tp
    metrics["FN"] = fn
    metrics["FP"] = fp
    metrics["TN"] = tn
    return metrics


def take_items(items, idx):
    """Return [items[i] for every i in idx], written as a plain loop."""
    out = []
    for i in idx:
        out.append(items[i])
    return out


class UrlCountDataset(Dataset):
    """A PyTorch Dataset of links, same pattern as STKDataset in hw3_skeleton.ipynb cell 75.

    Each item is the bag-of-words count vector of one link (and its label).
    The count vector is built only when the item is asked for, so we never
    hold a full (N, 50000) matrix in memory.

    Args:
        id_lists: output of texts_to_bucket_ids (one list of bucket ids per link)
        labels: label ids (0 = FoxNews, 1 = NBC), or None when there are no labels
    """

    def __init__(self, id_lists, labels=None):
        self.id_lists = id_lists
        self.labels = labels

    def __len__(self):
        return len(self.id_lists)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # count vector of link idx, shape (NUM_BUCKETS,)
        counts = bucket_ids_to_counts([self.id_lists[idx]])[0]

        if self.labels is not None:
            label = int(self.labels[idx])
            return counts, label
        return counts


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    One pass over dataloader, updating the model.
    Same as train_one_epoch in hw3_skeleton.ipynb cell 89
    (and the training function of L10_Neural Networks (ctd).pdf p.30).

    OUTPUT:
    avg_loss - float, the mean of the batch losses
    accuracy - float, the fraction of samples predicted correctly
    """
    model.train()
    total_loss = 0
    num_data = 0
    correct = 0

    for (x, y) in dataloader:
        inputs, label = x.to(device), y.to(device)
        forward = model(inputs)
        loss = criterion(forward, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_hat = forward.argmax(dim=1)
        correct += (y_hat == label).sum().item()
        num_data += y.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / num_data
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    One pass over dataloader WITHOUT updating the model.
    Same as evaluate in hw3_skeleton.ipynb cell 89.

    OUTPUT:
    avg_loss - float, the mean of the batch losses
    accuracy - float, the fraction of samples predicted correctly
    """
    model.eval()
    total_loss = 0
    num_data = 0
    correct = 0

    with torch.no_grad():
        for (x, y) in dataloader:
            inputs, label = x.to(device), y.to(device)
            forward = model(inputs)
            loss = criterion(forward, label)

            total_loss += loss.item()

            y_hat = forward.argmax(dim=1)
            correct += (y_hat == label).sum().item()
            num_data += y.shape[0]

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / num_data
    return avg_loss, accuracy


def predict_probs(model, id_lists, which, batch_size=500):
    """Class probabilities of one sub-model (or the ensemble), as an (N, 2) numpy array.

    Same pattern as predict in hw3_skeleton.ipynb cell 92: model.eval(),
    torch.no_grad(), loop over a DataLoader with no labels, collect the results in a list.

    Args:
        which: "lr", "tree" or "ensemble"
    """
    dataloader = DataLoader(UrlCountDataset(id_lists), batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x in dataloader:
            if which == "lr":
                probs = model.lr_probs(x)
            elif which == "tree":
                probs = model.tree_probs(x)
            else:
                probs = model.ensemble_probs(x)
            for row in probs.tolist():
                all_probs.append(row)
    return np.array(all_probs)


def make_optimizer(model, optimizer_name, lr, weight_decay):
    """NEW: build the optimizer for the logistic regression part.

    Weights get L2 weight decay, the bias does not (L11_Neural Networks
    (ctd)+K-Nearest Neighbors and Decision Trees.pdf p.17: "do not regularize
    bias terms"); same parameter-group trick as hw2_skeleton.ipynb cell 50.

    optimizer_name:
      "sgd"   : plain mini-batch SGD, w <- w - lr * (gradient + weight_decay * w).
                The extra weight_decay * w term is the gradient of the L2
                penalty, i.e. "weight decay" (L06_Linear Regression (contd.).pdf p.21).
                USED FOR THE FINAL MODEL.
      "adamw" : AdamW (L10_Neural Networks (ctd).pdf p.42). Kept only as a
                comparison experiment. Adam divides each weight's step by the size
                of that weight's own past gradients, so a rare headline word gets
                steps as large as the format tokens that appear in every link, so
                LR leans more on topic words. (AdamW gets a higher validation
                accuracy here, but our validation set has almost no NBC news links,
                so that score cannot show this weakness.)
    """
    groups = [{"params": [model.linear.weight], "weight_decay": weight_decay},
              {"params": [model.linear.bias], "weight_decay": 0.0}]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.SGD(groups, lr=lr)


def train_logistic_regression(model, train_ids, y_train, val_ids, y_val,
                              lr, weight_decay, max_epochs, batch_size, checkpoint_path,
                              patience=3, optimizer_name="sgd"):
    """Train the logistic regression part with mini-batch SGD + early stopping.

    The epoch loop is the one in hw3_skeleton.ipynb cell 90 (train_one_epoch,
    then evaluate, then record the history), plus early stopping.

    - Loss: cross-entropy, nn.CrossEntropyLoss as in hw3_skeleton.ipynb cell 88
      (softmax + negative log-likelihood, L08 p.7; for 2 classes this is the
      logistic regression loss of L07 p.22).
    - Mini-batch SGD: shuffled mini-batches from a DataLoader, L10_Neural Networks (ctd).pdf p.36.
    - L2 regularization / weight decay on the weights but NOT on the bias,
      L11_Neural Networks (ctd)+K-Nearest Neighbors and Decision Trees.pdf p.17
      (see make_optimizer).
    - Early stopping: save the checkpoint with the lowest validation loss and
      stop when it has not improved for `patience` epochs, L11 p.24.

    Returns:
        best_epoch: the epoch whose weights we kept
        history: per-epoch train/val loss and accuracy
    """
    # Step 1: data loaders, shuffle the training data only (hw3_skeleton.ipynb cell 82)
    train_dataloader = DataLoader(UrlCountDataset(train_ids, y_train), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(UrlCountDataset(val_ids, y_val), batch_size=batch_size, shuffle=False)

    # Step 2: loss and optimizer (weights get weight decay, the bias does not)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # Step 3: one pass over the training data, then check the validation set
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, "cpu")
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, "cpu")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Step 4: early stopping - save the best checkpoint to a file
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Step 5: load the best checkpoint back into the model
    model.load_state_dict(torch.load(checkpoint_path))
    return best_epoch, history


def train_logistic_regression_fixed_epochs(model, all_ids, all_y, lr, weight_decay, epochs, batch_size):
    """Same training as above, but for a fixed number of epochs and no validation.

    Used for the final model trained on all the data, where there is no
    validation set left to decide when to stop.
    """
    train_dataloader = DataLoader(UrlCountDataset(all_ids, all_y), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, "sgd", lr, weight_decay)
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, criterion, optimizer, "cpu")


def build_sparse_counts(id_lists):
    """Bag-of-words counts as a sparse matrix, for sklearn's decision tree.

    Same numbers as bucket_ids_to_counts, but only the non-zero entries are
    stored (a dense 3840 x 50000 matrix would need about 1.5 GB).
    """
    rows = []
    cols = []
    values = []
    for i in range(len(id_lists)):
        for j in id_lists[i]:
            rows.append(i)
            cols.append(j)
            values.append(1.0)
    # repeated (row, col) pairs are added up, which gives the counts
    return csr_matrix((values, (rows, cols)), shape=(len(id_lists), NUM_BUCKETS))


def choose_ccp_alpha(X_train, y_train, max_candidates=40):
    """Pick the pruning strength ccp_alpha by 5-fold cross-validated accuracy.

    Same approach as best_ccp_alpha_f1 in hw2_skeleton.ipynb cell 72
    (cost-complexity pruning is introduced in hw2_skeleton.ipynb cell 71).
    Sources: pruning and "add a complexity penalty", L13_K-Nearest Neighbors
    and Decision Trees (ctd.).pdf p.79-81; splits use the Gini index (sklearn's
    default), L13 p.67; choosing tree hyperparameters by cross-validation, L13 p.79 and
    L14_K-Nearest Neighbors and Decision Trees (ctd.).pdf p.26; k-fold
    cross-validation, L04_Linear Regression (contd.).pdf p.22-23.

    If two alphas tie, we keep the LARGER one (the smaller, simpler tree).
    """
    clf = DecisionTreeClassifier(random_state=42)
    ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alpha_array = ccp_path.ccp_alphas

    # The path can have hundreds of alphas; try at most max_candidates of them,
    # spread evenly along the path, to keep the run time short.
    if len(ccp_alpha_array) > max_candidates:
        pick = np.linspace(0, len(ccp_alpha_array) - 1, max_candidates).astype(int)
        ccp_alpha_array = ccp_alpha_array[pick]

    max_score = -1
    best_ccp_alpha = 0.0
    for ccp_alpha in ccp_alpha_array:
        clf_test = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        cv_scores = cross_val_score(clf_test, X_train, y_train, scoring="accuracy", cv=5)
        mean_score = np.mean(cv_scores)
        if mean_score >= max_score:  # ">=": on a tie, keep the larger alpha
            max_score = mean_score
            best_ccp_alpha = ccp_alpha
    return best_ccp_alpha, max_score


def copy_tree_into_model(model, tree_clf):
    """Copy a fitted sklearn tree into the model's tensors, so it is saved in model.pt.

    sklearn stores the tree as arrays (one entry per node):
      children_left / children_right : child node numbers (-1 at a leaf)
      feature / threshold            : the question asked at the node
      value                          : class amounts of the training samples at the node
    """
    t = tree_clf.tree_
    n_nodes = t.node_count
    if n_nodes > MAX_TREE_NODES:
        raise RuntimeError(f"Tree has {n_nodes} nodes, more than MAX_TREE_NODES={MAX_TREE_NODES}.")

    with torch.no_grad():
        for k in range(n_nodes):
            model.tree_left[k] = int(t.children_left[k])
            model.tree_right[k] = int(t.children_right[k])
            if t.children_left[k] == -1:
                # leaf: sklearn uses -2 for "no feature"; store 0, it is never read
                model.tree_feature[k] = 0
                model.tree_threshold[k] = 0.0
            else:
                model.tree_feature[k] = int(t.feature[k])
                model.tree_threshold[k] = float(t.threshold[k])
            # turn the class amounts into probabilities that sum to 1
            leaf_value = t.value[k][0]
            model.tree_prob[k, 0] = float(leaf_value[0] / leaf_value.sum())
            model.tree_prob[k, 1] = float(leaf_value[1] / leaf_value.sum())


def make_bucket_to_words(texts):
    """Map each hash bucket back to the words that land in it (only for printing)."""
    bucket_to_words = {}
    for text in texts:
        for tok in _simple_tokenize(text):
            b = _hash_token(tok, NUM_BUCKETS)
            if b not in bucket_to_words:
                bucket_to_words[b] = set()
            bucket_to_words[b].add(tok)
    return bucket_to_words


def describe_model(model, tree_clf, bucket_to_words, num_words=10):
    """Readable summary: the tree's rules and the logistic regression's strongest words."""
    lines = []

    # The tree, with bucket numbers replaced by words
    feature_names = []
    for b in range(NUM_BUCKETS):
        if b in bucket_to_words:
            feature_names.append("/".join(sorted(bucket_to_words[b])))
        else:
            feature_names.append("bucket_" + str(b))
    lines.append("Decision tree rules (class 0 = FoxNews, class 1 = NBC):")
    lines.append(export_text(tree_clf, feature_names=feature_names))

    # Logistic regression: weight for NBC minus weight for FoxNews, per bucket.
    # Positive = pushes towards NBC, negative = pushes towards FoxNews.
    w = model.linear.weight.detach()
    score = (w[1] - w[0]).numpy()
    order = np.argsort(score)
    lines.append(f"Logistic regression: {num_words} words that push most towards FoxNews:")
    for b in order[:num_words]:
        lines.append(f"  {'/'.join(sorted(bucket_to_words.get(b, {'?'}))):25s} {score[b]:+.3f}")
    lines.append(f"Logistic regression: {num_words} words that push most towards NBC:")
    for b in order[::-1][:num_words]:
        lines.append(f"  {'/'.join(sorted(bucket_to_words.get(b, {'?'}))):25s} {score[b]:+.3f}")
    return "\n".join(lines)


# ===========================================================================
# ORIGINAL main(), kept for comparison. Replaced by the new main() below.
# ===========================================================================
# def main():
#     parser = argparse.ArgumentParser(description="Train hashed text baseline and save model.pt")
#     parser.add_argument("--csv", type=str, default="url_train16k.csv", help="Path to url_only_data.csv")
#     parser.add_argument("--output", type=str, default="./model.pt", help="Output path to save trained model")
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--weight-decay", type=float, default=0.0)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
#     args = parser.parse_args()
#
#     set_seed(args.seed)
#
#     # Load and prepare data
#     texts, labels = prepare_data(args.csv)
#     if len(texts) == 0:
#         raise RuntimeError("No data produced by preprocess. Ensure URLs map to known outlets.")
#
#     y_all, mapping = encode_labels(labels)
#     # Prefer class order FoxNews=0, NBC=1 if present
#     # (mapping is only used for training; saved model predicts indices)
#
#     X_train, X_val, y_train, y_val = train_test_split(texts, y_all, test_size=0.2, random_state=args.seed, stratify=y_all)
#
#     # Build model
#     model = NewsClassifier()
#
#     # Train
#     train(
#         model=model,
#         texts_train=X_train,
#         y_train=y_train,
#         texts_val=X_val,
#         y_val=y_val,
#         device=args.device,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )
#
#     # Save full model object for easiest loading downstream
#     out_dir = os.path.dirname(os.path.abspath(args.output))
#     if out_dir and not os.path.exists(out_dir):
#         os.makedirs(out_dir, exist_ok=True)
#     torch.save(model.state_dict(), "model.pt")
#     print(f"Saved trained model to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train LR + pruned decision tree and save model.pt")
    # ORIGINAL default "url_train16k.csv" does not exist in the project folder
    parser.add_argument("--csv", type=str, default="../url_train.csv", help="Path to url_train.csv")
    parser.add_argument("--output", type=str, default="./model.pt", help="Output path to save trained model")
    parser.add_argument("--results-dir", type=str, default="./results", help="Folder for metrics and figures")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: load the data
    # -----------------------------------------------------------------------
    texts, labels = prepare_data(args.csv)
    raw_urls = list(pd.read_csv(args.csv)["url"])  # same order as texts
    if len(texts) == 0:
        raise RuntimeError("No data produced by preprocess. Ensure URLs map to known outlets.")

    # prepare_data keeps every row; for training we drop rows without a label
    keep_texts = []
    keep_urls = []
    keep_labels = []
    for i in range(len(texts)):
        if labels[i] != "":
            keep_texts.append(texts[i])
            keep_urls.append(raw_urls[i])
            keep_labels.append(labels[i])
    texts, raw_urls, labels = keep_texts, keep_urls, keep_labels

    y_all, mapping = encode_labels(labels)
    # make sure label ids match CLASS_NAMES in model.py (FoxNews = 0, NBC = 1)
    assert mapping == {"FoxNews": 0, "NBC": 1}, mapping
    y_all = y_all.numpy()
    print("Examples:", len(texts), "| FoxNews:", int(np.sum(y_all == 0)), "| NBC:", int(np.sum(y_all == 1)))

    # -----------------------------------------------------------------------
    # Step 2: train / validation / test split, 80 / 10 / 10, shuffled and stratified
    # (L04_Linear Regression (contd.).pdf p.18-21: train for parameters,
    #  validation for hyperparameters, test only for the final estimate)
    # -----------------------------------------------------------------------
    all_idx = np.arange(len(texts))
    train_idx, rest_idx = train_test_split(all_idx, test_size=0.2, random_state=args.seed, stratify=y_all)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=args.seed, stratify=y_all[rest_idx])

    train_texts = take_items(texts, train_idx)
    val_texts = take_items(texts, val_idx)
    test_texts = take_items(texts, test_idx)
    train_urls = take_items(raw_urls, train_idx)
    val_urls = take_items(raw_urls, val_idx)
    test_urls = take_items(raw_urls, test_idx)
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    print("Split sizes: train", len(train_idx), "| val", len(val_idx), "| test", len(test_idx))

    # Every evaluation set as raw urls + labels, so each model can clean them its own way
    eval_sets = {
        "val": (val_urls, y_val),
        "test": (test_urls, y_test),
    }

    # Hash every text once and reuse the bucket ids (saves time)
    train_ids = texts_to_bucket_ids(train_texts)
    val_ids = texts_to_bucket_ids(val_texts)
    eval_ids = {}
    for name in eval_sets:
        new_texts = []
        for url in eval_sets[name][0]:
            new_texts.append(url_to_text(url))
        eval_ids[name] = texts_to_bucket_ids(new_texts)

    results = []  # one row per (model, evaluation set)

    # -----------------------------------------------------------------------
    # Step 3: simple baselines to compare against
    # (L04 p.29 and L08 p.26: always compare with a trivial baseline)
    # -----------------------------------------------------------------------
    # 3a. majority class: always predict the most common training label
    majority = int(np.bincount(y_train).argmax())
    for name in eval_sets:
        true_y = eval_sets[name][1]
        row = {"model": "Majority class", "set": name}
        row.update(compute_metrics(np.full(len(true_y), majority), true_y))
        results.append(row)

    # 3b. the course baseline network, retrained on OUR training split with the
    #     baseline's own cleaning (_slug_to_text) and settings (5 epochs, Adam 1e-3)
    print("\n=== Baseline network (original code) ===")
    baseline_train_texts = []
    for url in train_urls:
        baseline_train_texts.append(_slug_to_text(urlparse(url).path))
    baseline_val_texts = []
    for url in val_urls:
        baseline_val_texts.append(_slug_to_text(urlparse(url).path))
    baseline_model = HashedTextClassifier(num_buckets=50000, embed_dim=128, hidden_dim=128, num_classes=2)
    baseline_history = train(baseline_model, baseline_train_texts, torch.tensor(y_train),
                             baseline_val_texts, torch.tensor(y_val), epochs=5)
    for name in eval_sets:
        b_texts = []
        for url in eval_sets[name][0]:
            b_texts.append(_slug_to_text(urlparse(url).path))
        preds = baseline_model.predict(b_texts)
        row = {"model": "Baseline NN", "set": name}
        row.update(compute_metrics(preds, eval_sets[name][1]))
        results.append(row)

    # -----------------------------------------------------------------------
    # Step 4: logistic regression - choose learning rate and weight decay on
    # the validation set (grid search on a log scale: L04 p.16, L11 p.28-29)
    # -----------------------------------------------------------------------
    print("\n=== Logistic regression: hyperparameter search on the validation set ===")
    # plain SGD needs larger learning rates than Adam
    lr_grid = [0.1, 0.3, 1.0]
    weight_decay_grid = [0.0, 0.001, 0.01]
    search_rows = []
    max_score = 0
    best = None
    checkpoint_path = os.path.join(args.results_dir, "best_lr_checkpoint.pt")
    criterion = nn.CrossEntropyLoss()
    val_dataloader = DataLoader(UrlCountDataset(val_ids, y_val), batch_size=args.batch_size, shuffle=False)
    for lr in lr_grid:
        for wd in weight_decay_grid:
            set_seed(args.seed)  # same random start for every setting (L11 p.29)
            model = NewsClassifier()
            best_epoch, history = train_logistic_regression(
                model, train_ids, y_train, val_ids, y_val,
                lr=lr, weight_decay=wd, max_epochs=args.max_epochs, batch_size=args.batch_size,
                checkpoint_path=checkpoint_path)
            val_loss, val_acc = evaluate(model, val_dataloader, criterion, "cpu")

            print(f"lr={lr:<4} weight_decay={wd:<6} best epoch={best_epoch:2d} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")
            search_rows.append({"lr": lr, "weight_decay": wd, "best_epoch": best_epoch,
                                "val_loss": val_loss, "val_acc": val_acc})

            # Choose the setting with the highest validation accuracy:
            # the holdout method of L04 p.20, written like best_ccp_alpha_f1
            # in hw2_skeleton.ipynb cell 72.
            if val_acc > max_score:
                max_score = val_acc
                best = {"lr": lr, "weight_decay": wd, "best_epoch": best_epoch,
                        "val_acc": val_acc, "val_loss": val_loss,
                        "model": model, "history": history}
    pd.DataFrame(search_rows).to_csv(os.path.join(args.results_dir, "lr_search.csv"), index=False)
    print(f"Chosen: lr={best['lr']}, weight_decay={best['weight_decay']}, epochs={best['best_epoch']}")
    eval_model = best["model"]

    # NEW: comparison experiment only - the first version's AdamW logistic regression
    # (lr=0.01, weight_decay=0.1, the setting validation accuracy picked). Not used
    # in the final model; it shows why we switched to SGD (see make_optimizer).
    print("\n=== Comparison: logistic regression trained with AdamW (first version) ===")
    set_seed(args.seed)
    adamw_model = NewsClassifier()
    adamw_epoch, _ = train_logistic_regression(
        adamw_model, train_ids, y_train, val_ids, y_val, lr=0.01, weight_decay=0.1,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        checkpoint_path=checkpoint_path, optimizer_name="adamw")
    for name in eval_sets:
        probs = predict_probs(adamw_model, eval_ids[name], "lr")
        row = {"model": "LR with AdamW (not used)", "set": name}
        row.update(compute_metrics(probs.argmax(axis=1), eval_sets[name][1]))
        results.append(row)

    # -----------------------------------------------------------------------
    # Step 5: pruned decision tree - choose ccp_alpha by 5-fold CV on the
    # training split (hw2_skeleton.ipynb cell 72)
    # -----------------------------------------------------------------------
    print("\n=== Decision tree: choosing ccp_alpha by 5-fold cross-validation ===")
    X_train_sparse = build_sparse_counts(train_ids)
    best_ccp_alpha, cv_acc = choose_ccp_alpha(X_train_sparse, y_train)
    tree_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)
    tree_clf.fit(X_train_sparse, y_train)
    print(f"Chosen ccp_alpha={best_ccp_alpha:.6f} | mean CV accuracy {cv_acc:.4f} | "
          f"{tree_clf.tree_.node_count} nodes, depth {tree_clf.get_depth()}")
    copy_tree_into_model(eval_model, tree_clf)

    # -----------------------------------------------------------------------
    # Step 6: evaluate LR, tree and the ensemble on every set
    # (the test split is only looked at here, after all choices were made)
    # -----------------------------------------------------------------------
    for which, label in [("lr", "Logistic regression"), ("tree", "Pruned tree"), ("ensemble", "LR + tree (final)")]:
        for name in eval_sets:
            probs = predict_probs(eval_model, eval_ids[name], which)
            row = {"model": label, "set": name}
            row.update(compute_metrics(probs.argmax(axis=1), eval_sets[name][1]))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.results_dir, "metrics.csv"), index=False)

    # Accuracy table: one row per model, one column per evaluation set
    model_order = ["Majority class", "Baseline NN", "LR with AdamW (not used)",
                   "Logistic regression", "Pruned tree", "LR + tree (final)"]
    set_order = ["val", "test"]
    accuracy_table = {}
    for model_name in model_order:
        accuracy_table[model_name] = []
        for set_name in set_order:
            for row in results:
                if row["model"] == model_name and row["set"] == set_name:
                    accuracy_table[model_name].append(row["accuracy"])

    print("\n=== Accuracy ===")
    print(f"{'model':26s}" + "".join(f"{s:>20s}" for s in set_order))
    for model_name in model_order:
        line = f"{model_name:26s}"
        for acc in accuracy_table[model_name]:
            line += f"{acc:>20.4f}"
        print(line)

    print("\nConfusion matrix of the final ensemble on the test split (NBC = positive):")
    for row in results:
        if row["model"] == "LR + tree (final)" and row["set"] == "test":
            print(f"  TP={row['TP']} FN={row['FN']} FP={row['FP']} TN={row['TN']} | "
                  f"NBC precision {row['nbc_precision']:.4f} recall {row['nbc_recall']:.4f}")

    # -----------------------------------------------------------------------
    # Step 7: figures (same style as hw3_skeleton.ipynb cell 90)
    # -----------------------------------------------------------------------
    # 7a. learning curves: baseline network vs our logistic regression
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    lr_hist = best["history"]
    ax1.plot(range(1, len(baseline_history["train_loss"]) + 1), baseline_history["train_loss"], "--", label="baseline train")
    ax1.plot(range(1, len(baseline_history["val_loss"]) + 1), baseline_history["val_loss"], "--", label="baseline validation")
    ax1.plot(range(1, len(lr_hist["train_loss"]) + 1), lr_hist["train_loss"], label="LR train")
    ax1.plot(range(1, len(lr_hist["val_loss"]) + 1), lr_hist["val_loss"], label="LR validation")
    ax1.axvline(best["best_epoch"], color="gray", linestyle=":", label="LR early-stopping epoch")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(range(1, len(baseline_history["train_acc"]) + 1), baseline_history["train_acc"], "--", label="baseline train")
    ax2.plot(range(1, len(baseline_history["val_acc"]) + 1), baseline_history["val_acc"], "--", label="baseline validation")
    ax2.plot(range(1, len(lr_hist["train_acc"]) + 1), lr_hist["train_acc"], label="LR train")
    ax2.plot(range(1, len(lr_hist["val_acc"]) + 1), lr_hist["val_acc"], label="LR validation")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "learning_curves.png"), dpi=150)
    plt.close()

    # 7b. one line per model across the four evaluation sets (includes the baseline,
    #     as the project asks: Project_Intro p.15 section 4.1)
    plt.figure(figsize=(8, 5))
    for model_name in model_order:
        plt.plot(set_order, accuracy_table[model_name], marker="o", label=model_name)
    plt.ylim(0, 1.05)
    plt.ylabel("accuracy")
    plt.title("Model comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "model_comparison.png"), dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # Step 8: final model - retrain on ALL labelled links with the chosen settings
    # (L04 p.23: "retrain on the whole dataset with the selected hyperparameters").
    # We also include our test split here, because the real test set is the
    # hidden leaderboard set; the numbers above come from the models trained
    # on the training split only.
    # -----------------------------------------------------------------------
    print("\n=== Final model: retraining on all", len(texts), "links ===")
    all_ids = texts_to_bucket_ids(texts)
    set_seed(args.seed)
    final_model = NewsClassifier()
    train_logistic_regression_fixed_epochs(final_model, all_ids, y_all, lr=best["lr"],
                                           weight_decay=best["weight_decay"],
                                           epochs=best["best_epoch"], batch_size=args.batch_size)
    final_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)
    final_tree.fit(build_sparse_counts(all_ids), y_all)
    copy_tree_into_model(final_model, final_tree)

    summary = describe_model(final_model, final_tree, make_bucket_to_words(texts))
    print(summary)
    with open(os.path.join(args.results_dir, "final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"lr={best['lr']}, weight_decay={best['weight_decay']}, epochs={best['best_epoch']}, "
                f"ccp_alpha={best_ccp_alpha}\n\n")
        f.write(summary)

    # Save the weights
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # ORIGINAL: torch.save(model.state_dict(), "model.pt")   <- ignored --output
    torch.save(final_model.state_dict(), args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
