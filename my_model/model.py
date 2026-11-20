from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ORIGINAL: these three imports were never used in the baseline, so they are
# commented out (the leaderboard only needs torch for this file).
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# NEW: constants shared by the model and the training script
# ---------------------------------------------------------------------------
# Class order used everywhere: index 0 = FoxNews, index 1 = NBC.
CLASS_NAMES = ["FoxNews", "NBC"]
# Number of hash buckets = length of the bag-of-words vector (same as the baseline).
NUM_BUCKETS = 50000
# The decision tree is saved inside model.pt as fixed-size tensors, so we fix the
# largest tree we can store. After pruning our tree is much smaller than this.
MAX_TREE_NODES = 255


def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    # replace punctuation with space, keep letters and numbers
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _hash_token(token: str, mod: int) -> int:
    # stable hashing into [0, mod)
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def texts_to_bucket_ids(texts: Sequence[str], num_buckets: int = NUM_BUCKETS) -> List[List[int]]:
    """NEW: tokenize each text and hash every token to a bucket number.

    Uses the baseline's tokenizer and hashing trick (tested baseline/model.py,
    _simple_tokenize and _hash_token), so no vocabulary has to be stored.

    Args:
        texts: list of N strings (the output of preprocess.prepare_data)
        num_buckets: number of hash buckets

    Returns:
        id_lists: list of N lists; id_lists[i] holds the bucket of each token of text i
    """
    id_lists = []
    for text in texts:
        ids = []
        for tok in _simple_tokenize(text):
            ids.append(_hash_token(tok, num_buckets))
        id_lists.append(ids)
    return id_lists


def bucket_ids_to_counts(id_lists: Sequence[List[int]], num_buckets: int = NUM_BUCKETS) -> torch.Tensor:
    """NEW: build the bag-of-words count vectors from bucket ids.

    Entry [i, j] = how many tokens of text i fall into bucket j. This is the
    bag-of-words feature map: "number of occurrences of word in the document"
    (L20_Natural Language Processing Part 2.pdf p.5-6).

    Args:
        id_lists: output of texts_to_bucket_ids
        num_buckets: length of each count vector

    Returns:
        counts: torch.Tensor of shape (N, num_buckets)
    """
    counts = torch.zeros(len(id_lists), num_buckets)
    for i in range(len(id_lists)):
        for j in id_lists[i]:
            counts[i, j] += 1
    return counts


def texts_to_counts(texts: Sequence[str], num_buckets: int = NUM_BUCKETS) -> torch.Tensor:
    """NEW: texts -> bag-of-words count vectors, shape (N, num_buckets)."""
    id_lists = texts_to_bucket_ids(texts, num_buckets)
    return bucket_ids_to_counts(id_lists, num_buckets)


class HashedTextClassifier(nn.Module):
    """
    ORIGINAL baseline model, kept unchanged.
    It is no longer the submitted model; train_baseline.py retrains it on the
    same data split so we can compare our new model against it.

    Simple neural baseline:
    - Tokenization: regex-based
    - Vocab: hashing trick to a fixed bucket size (no stored vocab needed)
    - Architecture: EmbeddingBag -> ReLU -> Linear(num_classes=2)
    - Predicts class indices (e.g., 0/1)

    This module implements `predict(batch_of_texts)` for leaderboard compatibility.
    """

    def __init__(self, num_buckets: int = 50000, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, embed_dim, mode="mean")
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # token_ids: [N_tokens], offsets: [batch_size]
        x = self.embedding(token_ids, offsets)  # [batch_size, embed_dim]
        logits = self.fc(x)  # [batch_size, num_classes]
        return logits

    @torch.no_grad()
    def predict(self, texts: Iterable[str]) -> List[int]:
        self.eval()
        texts_list = list(texts)
        # Build flattened token id tensor and offsets for EmbeddingBag
        all_ids: List[int] = []
        offsets: List[int] = []
        running_total = 0
        for t in texts_list:
            offsets.append(running_total)
            tokens = _simple_tokenize(t)
            ids = [_hash_token(tok, self.num_buckets) for tok in tokens] or [0]
            all_ids.extend(ids)
            running_total += len(ids)
        token_ids_t = torch.tensor(all_ids, dtype=torch.long)
        offsets_t = torch.tensor(offsets, dtype=torch.long)
        logits = self.forward(token_ids_t, offsets_t)
        preds = torch.argmax(logits, dim=-1)
        return preds.cpu().tolist()


# ORIGINAL: the submitted model was the baseline network.
# class NewsClassifier(HashedTextClassifier):
#     """
#     Backward-compatible name expected by the evaluator. Inherits the hashed classifier.
#     """
#
#     def __init__(self):
#         super().__init__(num_buckets=50000, embed_dim=128, hidden_dim=128, num_classes=2)


class NewsClassifier(nn.Module):
    """NEW submitted model: logistic regression + pruned decision tree, averaged.

    Both sub-models read the same bag-of-words count vector (texts_to_counts).

    1. Logistic regression (L07_Logistic Regression.pdf p.13; for 2 classes we
       use the softmax form, L08_Logistic Regression+Logistic Regression.pdf p.7):
           p(class | x) = softmax(W x + b)
       Written as one nn.Linear layer, like L10_Neural Networks (ctd).pdf p.29.
       Trained in train_baseline.py with cross-entropy + L2 weight decay.

    2. Pruned decision tree (L13_K-Nearest Neighbors and Decision Trees
       (ctd.).pdf p.41-47; splits chosen by Gini index reduction, sklearn's
       default, L13 p.67; pruning to avoid overfitting, L13 p.79-81):
       trained with sklearn in train_baseline.py,
       then copied into the tensors below so it is saved in model.pt.
       Node k: if counts[tree_feature[k]] <= tree_threshold[k] go to
       tree_left[k], else go to tree_right[k]; a node with tree_left == -1
       is a leaf and tree_prob[k] holds its class probabilities.

    3. Ensemble: average the two probability vectors and take the larger class
       (Project_Intro p.10 section 2.7 and p.18 FAQ: put all sub-models and
       the averaging inside the one Model class).
    """

    def __init__(self):
        super().__init__()
        self.num_buckets = NUM_BUCKETS

        # Sub-model 1: logistic regression = one linear layer, 2 outputs (logits)
        self.linear = nn.Linear(NUM_BUCKETS, 2)

        # Sub-model 2: the decision tree, stored as "buffers".
        # A buffer is a tensor that is saved in state_dict (so in model.pt)
        # but is not trained by gradient descent.
        # Default = a single leaf that says 50/50, i.e. "no opinion".
        self.register_buffer("tree_feature", torch.zeros(MAX_TREE_NODES, dtype=torch.long))
        self.register_buffer("tree_threshold", torch.zeros(MAX_TREE_NODES))
        self.register_buffer("tree_left", -torch.ones(MAX_TREE_NODES, dtype=torch.long))
        self.register_buffer("tree_right", -torch.ones(MAX_TREE_NODES, dtype=torch.long))
        self.register_buffer("tree_prob", torch.full((MAX_TREE_NODES, 2), 0.5))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Logistic regression scores (logits), shape (N, 2).

        Like L10 p.29 we return logits, not probabilities: F.cross_entropy
        applies the softmax itself during training.
        """
        return self.linear(counts)

    def lr_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """Logistic regression probabilities, shape (N, 2) (L10 p.31: softmax of the logits)."""
        return F.softmax(self.forward(counts), dim=1)

    def tree_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """Decision tree probabilities, shape (N, 2).

        For each example, start at the root (node 0) and follow the questions
        down to a leaf, then read that leaf's class probabilities.
        """
        probs = torch.zeros(counts.shape[0], 2)
        for i in range(counts.shape[0]):
            node = 0
            # keep going down until we reach a leaf (a leaf has no left child)
            while self.tree_left[node].item() != -1:
                feature = self.tree_feature[node].item()
                if counts[i, feature].item() <= self.tree_threshold[node].item():
                    node = self.tree_left[node].item()
                else:
                    node = self.tree_right[node].item()
            probs[i] = self.tree_prob[node]
        return probs

    def ensemble_probs(self, counts: torch.Tensor) -> torch.Tensor:
        """Average of the two sub-models' probabilities, shape (N, 2)."""
        return (self.lr_probs(counts) + self.tree_probs(counts)) / 2

    @torch.no_grad()
    def predict(self, texts: Iterable[str]) -> List[str]:
        """Called by the leaderboard with a batch of texts from prepare_data.

        Returns the label strings "FoxNews" / "NBC" directly, so the result does
        not depend on the evaluator guessing which number means which outlet.
        """
        self.eval()  # L10 p.31: switch to inference mode before predicting
        texts_list = list(texts)
        counts = texts_to_counts(texts_list, self.num_buckets)
        probs = self.ensemble_probs(counts)
        pred_ids = torch.argmax(probs, dim=1).tolist()

        preds = []
        for k in pred_ids:
            preds.append(CLASS_NAMES[k])
        return preds


def get_model() -> NewsClassifier:
    return NewsClassifier()
