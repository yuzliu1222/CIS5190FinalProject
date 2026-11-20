from __future__ import annotations

import csv
import html
import os
import re
from typing import List, Tuple
from urllib.parse import urlparse, unquote


# ---------------------------------------------------------------------------
# NEW: settings used by the new text cleaning (url_to_text below)
# ---------------------------------------------------------------------------
# An NBC article link ends with an ID such as "-rcna158308", "-ncna1286372" or
# "-n1234567": a dash, a few letters, then 6+ digits. Fox links never end like
# this in url_train.csv (2,137 NBC links vs 0 Fox links). Fox links that end in
# a number end in a 4-digit year ("-2025"), which this pattern does not match.
NBC_ID_PATTERN = r"-[a-z]*\d{6,}$"


def _slug_to_text(path: str) -> str:
    # ORIGINAL baseline function. Kept unchanged: train_baseline.py still uses it
    # to retrain the baseline on the same split, so we can compare against it.
    # Take last segments, split on -, _, and decode
    segments = [seg for seg in path.split("/") if seg]
    if not segments:
        return ""
    tail = segments[-1]
    tail = tail.split("?")[0]
    tail = tail.split("#")[0]
    tail = tail.rsplit(".", 1)[0]  # drop extension
    # Replace separators with spaces and unescape
    text = tail.replace("-", " ").replace("_", " ")
    text = unquote(html.unescape(text))
    return text.strip()


def url_to_text(url: str) -> str:
    """NEW: turn one link into the text our model reads.

    The text has two parts:
      1. the words of the headline (the last part of the path), lowercased,
         with the article ID and all numbers removed
      2. two "format tokens" that describe the shape of the link:
           - "segcountK"  : the path has K parts, e.g. /select/shopping/x -> segcount3
           - "nbcidtoken" : the link ends with an NBC-style article ID

    Example:
      "/select/shopping/best-gaming-mouse-ncna1294260"
        -> "best gaming mouse segcount3 nbcidtoken"
      "/politics/while-pelosi-mulls-political-future"
        -> "while pelosi mulls political future segcount2"

    Why: the baseline only used the headline words, so it learned TOPICS
    (shopping = NBC, politics = Fox). In url_train.csv 94% of NBC links are
    shopping articles, so NBC political news gets labelled Fox. The two format
    tokens do not depend on the topic.
    Sources: hand-made feature map for text, L02_Linear_regression.pdf p.58-63
    and L20_Natural Language Processing Part 2.pdf p.3; cleaning before
    bag-of-words, L20 p.7; Project_Intro p.16 (section 4.3.1, lowercase and
    remove special characters).

    Args:
        url: a full link (training data) or a path without the website name
             (leaderboard data), e.g. "/politics/some-headline"

    Returns:
        text: a string of space-separated tokens
    """
    parsed = urlparse(url.strip())
    path = parsed.path

    # Step 1: split the path into its parts, e.g. ["select", "shopping", "best-..."]
    segments = [seg for seg in path.split("/") if seg]
    format_tokens = "segcount" + str(len(segments))
    if len(segments) == 0:
        return format_tokens

    # Step 2: take the headline part (the last segment) and clean it
    tail = segments[-1]
    tail = unquote(html.unescape(tail)).lower()
    tail = re.sub(r"\.(html?|php|aspx?)$", "", tail)  # drop a file extension if any

    # Step 3: check for an NBC-style article ID, then remove it from the words
    if re.search(NBC_ID_PATTERN, tail):
        format_tokens = format_tokens + " nbcidtoken"
        tail = re.sub(NBC_ID_PATTERN, "", tail)

    # Step 4: keep letters only (removes dashes, digits and punctuation)
    words = re.sub(r"[^a-z]+", " ", tail).strip()

    # Step 5: words first, then the format tokens
    text = words + " " + format_tokens
    return text.strip()


def _label_from_host(netloc: str) -> str:
    host = netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host or "nbc.com" in host or "nbc" in host:
        return "NBC"
    return ""


def _process(url: str, texts: List[str], labels: List[str]) -> None:
    parsed = urlparse(url)
    label = _label_from_host(parsed.netloc)
    # ORIGINAL: links from other websites were skipped here.
    # Skip links from other websites. Links WITHOUT a website name (how the leaderboard
    # gives them to you) are kept, with an empty label, so every row gets a prediction.
    # if parsed.netloc and not label:
    #     return
    # NEW: never skip a row. The project contract says prepare_data must return
    # exactly one input per row, in order (Project_Intro p.7, section 2.3).
    # Rows with an empty label are removed later, in the training script only.

    # ORIGINAL: only the last part of the path was used.
    # text = _slug_to_text(parsed.path)
    # if not text:
    #     text = parsed.netloc
    # NEW: headline words + format tokens (see url_to_text above)
    text = url_to_text(url)
    texts.append(text)
    labels.append(label)


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads a CSV with a single 'url' column and produces:
      - X: list of text strings derived from URL slugs
           (NEW: headline words + format tokens, see url_to_text)
      - y: list of labels inferred from host ('FoxNews' or 'NBC'), or '' if the
           link has no website name (the leaderboard removes it from test links)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    texts: List[str] = []
    labels: List[str] = []

    # ORIGINAL: with open(csv_path, "r") as f:
    # NEW: read as UTF-8, so the file opens the same way on Windows and on the backend
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # If no header named 'url', try to handle a single-column CSV
        if "url" not in (reader.fieldnames or []):
            f.seek(0)
            for row in csv.reader(f):
                if row:
                    _process(row[0], texts, labels)
        else:
            for row in reader:
                _process(row.get("url", "") or "", texts, labels)

    return texts, labels


if __name__ == "__main__":
    # Example usage
    # ORIGINAL: input_csv = "url_train.csv"
    # NEW: the CSV sits one folder up from my_model/
    input_csv = "../url_train.csv"
    X, y = prepare_data(input_csv)
    # NEW: print a few examples so we can check the cleaning by eye
    for i in range(5):
        print(y[i], "|", X[i])
