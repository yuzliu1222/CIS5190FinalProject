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
from preprocess import NBC_ID_PATTERN, _slug_to_text, prepare_data, url_to_text

# NEW imports
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import torch.nn.functional as F
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

def make_probe_sets(test_urls, test_y):
    """Build two small robustness test sets from the links of ONE split.

    NEW (second version): called twice - on the validation split (used to choose
    hyperparameters) and on the test split (used only for the final report), so
    the test split never influences any choice (L04 p.21, "validation hygiene").

    The training data has almost no NBC news (94% of NBC links are shopping
    articles), but the leaderboard probably has many. These probes check
    whether a model looks at the link FORMAT or only at the TOPIC:

      - "NBC politics probe": a Fox political headline written in NBC's link
        format (3 path parts + NBC-style ID). True label: NBC.
      - "Fox shopping probe": an NBC shopping headline written in Fox's link
        format (2 path parts, no ID). True label: FoxNews.

    These links are made up by us. They only test whether the model uses the
    link format; they are NOT an estimate of the leaderboard accuracy.
    Idea: a validation set should look like the real exam (L04_Linear
    Regression (contd.).pdf p.18).

    Returns:
        probes: dict name -> (list of urls, numpy array of label ids)
    """
    politics_urls = []
    shopping_urls = []
    for i in range(len(test_urls)):
        path = urlparse(test_urls[i]).path
        segments = [seg for seg in path.split("/") if seg]
        slug = segments[-1]
        if test_y[i] == 0 and segments[0] in ["politics", "us", "world"]:
            politics_urls.append("/politics/congress/" + slug + "-rcna123456")
        if test_y[i] == 1 and segments[0] == "select":
            slug_no_id = re.sub(NBC_ID_PATTERN, "", slug)
            shopping_urls.append("/lifestyle/" + slug_no_id)

    probes = {}
    probes["NBC politics probe"] = (politics_urls, np.ones(len(politics_urls), dtype=int))
    probes["Fox shopping probe"] = (shopping_urls, np.zeros(len(shopping_urls), dtype=int))
    return probes


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


def predict_probs_in_chunks(model, id_lists, which, chunk_size=500):
    """Class probabilities of one sub-model (or the ensemble), shape (N, 2).

    We build the count vectors a chunk at a time, because a (N, 50000) matrix
    for all rows at once would use too much memory.

    Args:
        which: "lr", "tree" or "ensemble"
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(id_lists), chunk_size):
            counts = bucket_ids_to_counts(id_lists[start:start + chunk_size])
            if which == "lr":
                probs = model.lr_probs(counts)
            elif which == "tree":
                probs = model.tree_probs(counts)
            else:
                probs = model.ensemble_probs(counts)
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0)


def lr_loss_and_accuracy(model, id_lists, y):
    """Cross-entropy loss and accuracy of the logistic regression part."""
    probs = predict_probs_in_chunks(model, id_lists, "lr")
    y_t = torch.tensor(y, dtype=torch.long)
    # cross-entropy = mean of -log p(true class)   (L08 p.7)
    loss = F.nll_loss(torch.log(probs + 1e-12), y_t).item()
    acc = (probs.argmax(dim=1) == y_t).float().mean().item()
    return loss, acc


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
                steps as large as the format tokens that appear in every link. In
                our runs this made LR rely on topic words: about 99% validation
                accuracy, but only 15% / 0.4% on the two robustness probes.
    """
    groups = [{"params": [model.linear.weight], "weight_decay": weight_decay},
              {"params": [model.linear.bias], "weight_decay": 0.0}]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.SGD(groups, lr=lr)


def train_logistic_regression(model, train_ids, y_train, val_ids, y_val,
                              lr, weight_decay, max_epochs, batch_size, patience=3,
                              optimizer_name="sgd"):
    """Train the logistic regression part with mini-batch SGD + early stopping.

    - Loss: cross-entropy (softmax + negative log-likelihood), L08 p.7;
      for 2 classes this is the logistic regression loss of L07 p.22.
    - Mini-batch SGD: shuffle, then one step per batch, L10_Neural Networks (ctd).pdf p.36.
    - L2 regularization / weight decay on the weights but NOT on the bias,
      L11_Neural Networks (ctd)+K-Nearest Neighbors and Decision Trees.pdf p.17
      (see make_optimizer).
    - Early stopping: keep the checkpoint with the lowest validation loss and
      stop when it has not improved for `patience` epochs, L11 p.24.

    Returns:
        best_epoch: the epoch whose weights we kept
        history: per-epoch train/val loss and accuracy
    """
    train_y_t = torch.tensor(y_train, dtype=torch.long)

    # Step 1: weights get weight decay, the bias does not
    # ORIGINAL (first version): AdamW
    # optimizer = torch.optim.AdamW(
    #     [{"params": [model.linear.weight], "weight_decay": weight_decay},
    #      {"params": [model.linear.bias], "weight_decay": 0.0}], lr=lr)
    # NEW: plain SGD by default (see make_optimizer for why)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_weight = model.linear.weight.detach().clone()
    best_bias = model.linear.bias.detach().clone()
    epochs_without_improvement = 0

    n_train = len(train_ids)
    for epoch in range(1, max_epochs + 1):
        # Step 2: one pass over the shuffled training data, one step per mini-batch
        model.train()
        indices = np.random.permutation(n_train)
        total_loss = 0.0
        correct = 0
        for start in range(0, n_train, batch_size):
            idx = indices[start:start + batch_size]
            batch_ids = []
            for i in idx:
                batch_ids.append(train_ids[i])
            counts = bucket_ids_to_counts(batch_ids)
            targets = train_y_t[idx]

            optimizer.zero_grad()
            logits = model(counts)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == targets).sum().item()

        train_loss = total_loss / n_train
        train_acc = correct / n_train

        # Step 3: check the validation set
        val_loss, val_acc = lr_loss_and_accuracy(model, val_ids, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Step 4: early stopping - keep the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weight = model.linear.weight.detach().clone()
            best_bias = model.linear.bias.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Step 5: put the best weights back into the model
    with torch.no_grad():
        model.linear.weight.copy_(best_weight)
        model.linear.bias.copy_(best_bias)
    return best_epoch, history


def train_logistic_regression_fixed_epochs(model, all_ids, all_y, lr, weight_decay, epochs, batch_size):
    """Same training as above, but for a fixed number of epochs and no validation.

    Used for the final model trained on all the data, where there is no
    validation set left to decide when to stop.
    """
    all_y_t = torch.tensor(all_y, dtype=torch.long)
    # ORIGINAL (first version): AdamW
    # optimizer = torch.optim.AdamW(
    #     [{"params": [model.linear.weight], "weight_decay": weight_decay},
    #      {"params": [model.linear.bias], "weight_decay": 0.0}], lr=lr)
    # NEW: plain SGD, same as in train_logistic_regression
    optimizer = make_optimizer(model, "sgd", lr, weight_decay)
    n = len(all_ids)
    model.train()
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch_ids = []
            for i in idx:
                batch_ids.append(all_ids[i])
            counts = bucket_ids_to_counts(batch_ids)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(counts), all_y_t[idx])
            loss.backward()
            optimizer.step()


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

    def take(items, idx):
        out = []
        for i in idx:
            out.append(items[i])
        return out

    train_texts, val_texts, test_texts = take(texts, train_idx), take(texts, val_idx), take(texts, test_idx)
    train_urls, val_urls, test_urls = take(raw_urls, train_idx), take(raw_urls, val_idx), take(raw_urls, test_idx)
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    print("Split sizes: train", len(train_idx), "| val", len(val_idx), "| test", len(test_idx))

    # Robustness probes (see make_probe_sets)
    # ORIGINAL (first version): probes from the test split only
    # probes = make_probe_sets(test_urls, y_test)
    # NEW: validation-split probes for choosing hyperparameters,
    #      test-split probes only for the final report
    val_probes = make_probe_sets(val_urls, y_val)
    probes = make_probe_sets(test_urls, y_test)
    for name in probes:
        print(f"{name}: {len(val_probes[name][0])} made-up links from val, "
              f"{len(probes[name][0])} from test")
    val_probe_ids = {}
    for name in val_probes:
        new_texts = []
        for url in val_probes[name][0]:
            new_texts.append(url_to_text(url))
        val_probe_ids[name] = texts_to_bucket_ids(new_texts)

    # Every evaluation set as raw urls + labels, so each model can clean them its own way
    eval_sets = {
        "val": (val_urls, y_val),
        "test": (test_urls, y_test),
        "NBC politics probe": probes["NBC politics probe"],
        "Fox shopping probe": probes["Fox shopping probe"],
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
    # ORIGINAL (first version): AdamW learning rates, choose by validation accuracy only
    # lr_grid = [0.001, 0.01]
    # weight_decay_grid = [0.0, 0.001, 0.01, 0.1, 1.0]
    # NEW: plain SGD needs larger learning rates
    lr_grid = [0.1, 0.3, 1.0]
    weight_decay_grid = [0.0, 0.001, 0.01]
    search_rows = []
    best = None
    for lr in lr_grid:
        for wd in weight_decay_grid:
            set_seed(args.seed)  # same random start for every setting (L11 p.29)
            model = NewsClassifier()
            best_epoch, history = train_logistic_regression(
                model, train_ids, y_train, val_ids, y_val,
                lr=lr, weight_decay=wd, max_epochs=args.max_epochs, batch_size=args.batch_size)
            val_loss, val_acc = lr_loss_and_accuracy(model, val_ids, y_val)

            # NEW: accuracy on the two validation-split probes
            probe_acc = {}
            for name in val_probes:
                probs = predict_probs_in_chunks(model, val_probe_ids[name], "lr")
                probe_acc[name] = float(np.mean(probs.argmax(dim=1).numpy() == val_probes[name][1]))

            # NEW: robust selection score = average of the three accuracies.
            # The leaderboard links differ from url_train.csv (it has many more NBC
            # news links), so the validation check must include that difference
            # too: "the validation set is like a sample exam" (L04 p.18).
            score = (val_acc + probe_acc["NBC politics probe"] + probe_acc["Fox shopping probe"]) / 3

            print(f"lr={lr:<4} weight_decay={wd:<6} best epoch={best_epoch:2d} | val loss {val_loss:.4f} "
                  f"acc {val_acc:.4f} | NBC politics probe {probe_acc['NBC politics probe']:.3f} "
                  f"Fox shopping probe {probe_acc['Fox shopping probe']:.3f} | score {score:.4f}")
            search_rows.append({"lr": lr, "weight_decay": wd, "best_epoch": best_epoch,
                                "val_loss": val_loss, "val_acc": val_acc,
                                "val_nbc_politics_probe": probe_acc["NBC politics probe"],
                                "val_fox_shopping_probe": probe_acc["Fox shopping probe"],
                                "selection_score": score})
            # ORIGINAL (first version): choose by validation accuracy only
            # if best is None or val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            # NEW: choose by the robust score; on a tie, the lower validation loss
            if best is None or score > best["score"] or (score == best["score"] and val_loss < best["val_loss"]):
                best = {"lr": lr, "weight_decay": wd, "best_epoch": best_epoch,
                        "val_acc": val_acc, "val_loss": val_loss, "score": score,
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
        max_epochs=args.max_epochs, batch_size=args.batch_size, optimizer_name="adamw")
    for name in eval_sets:
        probs = predict_probs_in_chunks(adamw_model, eval_ids[name], "lr")
        row = {"model": "LR with AdamW (not used)", "set": name}
        row.update(compute_metrics(probs.argmax(dim=1).numpy(), eval_sets[name][1]))
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
            probs = predict_probs_in_chunks(eval_model, eval_ids[name], which)
            row = {"model": label, "set": name}
            row.update(compute_metrics(probs.argmax(dim=1).numpy(), eval_sets[name][1]))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.results_dir, "metrics.csv"), index=False)
    table = results_df.pivot(index="model", columns="set", values="accuracy")
    table = table[["val", "test", "NBC politics probe", "Fox shopping probe"]]
    # ORIGINAL (first version):
    # table = table.loc[["Majority class", "Baseline NN", "Logistic regression", "Pruned tree", "LR + tree (final)"]]
    # NEW: also show the AdamW comparison experiment
    table = table.loc[["Majority class", "Baseline NN", "LR with AdamW (not used)",
                       "Logistic regression", "Pruned tree", "LR + tree (final)"]]
    print("\n=== Accuracy (probes: fraction of made-up links labelled correctly) ===")
    print(table.round(4).to_string())
    print("\nConfusion matrix of the final ensemble on the test split (NBC = positive):")
    final_test = results_df[(results_df["model"] == "LR + tree (final)") & (results_df["set"] == "test")].iloc[0]
    print(f"  TP={final_test['TP']} FN={final_test['FN']} FP={final_test['FP']} TN={final_test['TN']} | "
          f"NBC precision {final_test['nbc_precision']:.4f} recall {final_test['nbc_recall']:.4f}")

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
    for model_name in table.index:
        plt.plot(list(table.columns), list(table.loc[model_name]), marker="o", label=model_name)
    plt.ylim(0, 1.05)
    plt.ylabel("accuracy")
    plt.title("Model comparison (probes are made-up links)")
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
