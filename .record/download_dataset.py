#!/usr/bin/env python
"""Materialize SkillOpt benchmark splits from their raw upstream sources.

The repository only ships lightweight *manifests* under ``data/<bench>_id_split/``
(stable IDs / path hints, no questions/answers/images). This script downloads the
raw upstream payloads and joins them against those manifests to produce the
runnable ``--split_dir`` directories that ``scripts/train.py`` expects.

What it produces (matching each env's dataloader + the configs):

    searchqa               -> data/searchqa_split/{train,val,test}/items.json
    livemathematicianbench -> data/livemathematicianbench_split/{split}/items.json
    docvqa                 -> data/docvqa/splits/{split}/items.csv  (+ data/docvqa_images/*.png)
    officeqa               -> data/officeqa_split/{split}/items.csv  (gated HF dataset)
    spreadsheetbench       -> data/spreadsheetbench_split/{split}/items.json
                              (+ data/spreadsheetbench_verified_400/  spreadsheet payloads)

ALFWorld is intentionally NOT handled here: its manifest (data/alfworld_path_split)
is already runnable; you only need the game files via ``alfworld-download`` and
``$ALFWORLD_DATA`` (see data/README.md).

Usage
-----
    pip install datasets huggingface_hub pillow

    # everything that does not require gated access (default)
    python .record/download_dataset.py

    # explicit subset
    python .record/download_dataset.py --only searchqa,docvqa

    # gated dataset (OfficeQA) needs a token with access granted on HF
    python .record/download_dataset.py --only officeqa --hf-token hf_xxx

    # China mirror (recommended if huggingface.co is slow/blocked)
    HF_ENDPOINT=https://hf-mirror.com python .record/download_dataset.py

Each benchmark runs independently and is wrapped in try/except, so one failure
(e.g. a gated dataset you lack access to) will not abort the others. A summary
is printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
import traceback
from pathlib import Path

# Repo root = parent of the .record/ directory holding this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

ALL_BENCHMARKS = [
    "searchqa",
    "livemathematicianbench",
    "docvqa",
    "officeqa",
    "spreadsheetbench",
]


# ── small helpers ─────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(bench: str) -> dict[str, list[dict]]:
    """Return {split: [manifest item, ...]} for one ``*_id_split`` directory."""
    base = DATA / f"{bench}_id_split"
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        items_file = base / split / "items.json"
        if not items_file.is_file():
            raise FileNotFoundError(f"Missing manifest: {items_file}")
        out[split] = json.loads(items_file.read_text())
    counts = {k: len(v) for k, v in out.items()}
    log(f"{bench}: manifest counts {counts}")
    return out


def write_json_split(out_root: Path, split: str, items: list[dict]) -> None:
    d = ensure_dir(out_root / split)
    (d / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    log(f"  wrote {split}: {len(items)} -> {d / 'items.json'}")


def resolve_token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ── SearchQA ──────────────────────────────────────────────────────────────

def build_searchqa(token: str | None) -> None:
    from datasets import load_dataset

    manifest = load_manifest("searchqa")
    out_root = DATA / "searchqa_split"

    log("searchqa: loading lucadiliello/searchqa (all splits)...")
    ds = load_dataset("lucadiliello/searchqa", token=token)
    by_key: dict[str, dict] = {}
    for split in ds.values():
        for r in split:
            by_key[r["key"]] = r

    for split, ids in manifest.items():
        items, missing = [], 0
        for x in ids:
            r = by_key.get(x["id"])
            if r is None:
                missing += 1
                continue
            items.append(
                {
                    "id": r["key"],
                    "question": r["question"],
                    "context": r["context"],
                    "answers": r["answers"],
                }
            )
        if missing:
            log(f"  WARNING {split}: {missing} ids not found upstream")
        write_json_split(out_root, split, items)


# ── LiveMathematicianBench ────────────────────────────────────────────────

_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _lmb_norm_label(text) -> str:
    return str(text).strip().upper().rstrip(".):")


def _lmb_choices(raw) -> list[dict]:
    if isinstance(raw, list):
        out = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                label = str(item.get("label") or _CHOICE_LABELS[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label, text = _CHOICE_LABELS[idx], str(item).strip()
            if text:
                out.append({"label": label, "text": text})
        return out
    if isinstance(raw, dict):
        return [
            {"label": str(k).strip(), "text": str(raw[k]).strip()}
            for k in sorted(raw)
            if str(raw[k]).strip()
        ]
    return []


def _lmb_theorem_types(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _lmb_normalize(item: dict) -> dict:
    """Mirror skillopt/envs/livemathematicianbench/dataloader.py::_normalize_item."""
    mcq = item.get("mcq", {}) if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _lmb_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}
    if isinstance(correct, dict):
        correct_label = _lmb_norm_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label, correct_text = _lmb_norm_label(correct), ""

    by_label = {_lmb_norm_label(c["label"]): c["text"] for c in choices}
    if correct_label and not correct_text:
        correct_text = by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in by_label:
        choices.append({"label": correct_label, "text": correct_text})

    month = str(item.get("month") or "").strip()
    no = item.get("no")
    item_id = f"{month}:{no}" if month else str(no)
    return {
        "id": item_id,
        "month": month,
        "no": no,
        "paper_link": str(item.get("paper_link") or "").strip(),
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _lmb_theorem_types(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
    }


def build_livemathematicianbench(token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    manifest = load_manifest("livemathematicianbench")
    out_root = DATA / "livemathematicianbench_split"
    repo = "LiveMathematicianBench/LiveMathematicianBench"
    revision = "b72450f6ce96c26158d64d945a5d31ef7727be41"

    # Every monthly source file referenced by the manifest.
    source_files = sorted(
        {x["source_file"] for items in manifest.values() for x in items if x.get("source_file")}
    )
    log(f"livemathematicianbench: source files {source_files}")

    by_id: dict[str, dict] = {}
    for rel in source_files:
        month = next((p for p in rel.split("/") if p.isdigit() and len(p) == 6), "")
        local = hf_hub_download(repo, filename=rel, repo_type="dataset", revision=revision, token=token)
        raw = json.loads(Path(local).read_text())
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array in {rel}")
        for row in raw:
            if month and not row.get("month"):
                row = {**row, "month": month}
            norm = _lmb_normalize(row)
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                by_id[norm["id"]] = norm

    for split, ids in manifest.items():
        items, missing = [], 0
        for x in ids:
            norm = by_id.get(x["id"])
            if norm is None:
                missing += 1
                continue
            items.append(norm)
        if missing:
            log(f"  WARNING {split}: {missing} ids not found upstream")
        write_json_split(out_root, split, items)


# ── DocVQA ────────────────────────────────────────────────────────────────

_DOCVQA_COLUMNS = [
    "questionId", "docId", "question", "answer", "image_path", "topic",
    "ucsf_document_id", "ucsf_document_page_no", "source_split",
]


def build_docvqa(token: str | None) -> None:
    from datasets import load_dataset

    manifest = load_manifest("docvqa")
    out_root = DATA / "docvqa" / "splits"
    revision = "539088ef8a8ada01ac8e2e6d4e372586748a265e"

    log("docvqa: loading lmms-lab/DocVQA (config=DocVQA, split=validation)...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", revision=revision, token=token)
    by_qid: dict[str, dict] = {str(r["questionId"]): r for r in ds}

    for split, rows in manifest.items():
        out_rows, missing = [], 0
        split_dir = ensure_dir(out_root / split)
        for m in rows:
            qid = str(m.get("questionId") or m.get("id"))
            r = by_qid.get(qid)
            if r is None:
                missing += 1
                continue
            # Save the page image to the manifest-declared path (relative to repo root).
            image_rel = m.get("image_path") or f"data/docvqa_images/q{qid}.png"
            image_abs = REPO_ROOT / image_rel
            ensure_dir(image_abs.parent)
            img = r.get("image")
            if img is not None and not image_abs.exists():
                img.save(image_abs)
            answers = r.get("answers") or []
            out_rows.append(
                {
                    "questionId": qid,
                    "docId": m.get("docId", ""),
                    "question": r.get("question", ""),
                    "answer": json.dumps(list(answers), ensure_ascii=False),
                    "image_path": image_rel,
                    "topic": m.get("topic", ""),
                    "ucsf_document_id": m.get("ucsf_document_id", ""),
                    "ucsf_document_page_no": m.get("ucsf_document_page_no", ""),
                    "source_split": m.get("source_split", "validation"),
                }
            )
        if missing:
            log(f"  WARNING {split}: {missing} questionIds not found upstream")
        out_csv = split_dir / "items.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_DOCVQA_COLUMNS)
            w.writeheader()
            w.writerows(out_rows)
        log(f"  wrote {split}: {len(out_rows)} -> {out_csv}")


# ── OfficeQA (gated) ──────────────────────────────────────────────────────

_OFFICEQA_COLUMNS = [
    "uid", "question", "ground_truth", "category", "source_files", "source_docs", "split",
]


def build_officeqa(token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    if not token:
        raise RuntimeError(
            "OfficeQA (databricks/officeqa) is gated: pass --hf-token / set HF_TOKEN "
            "with access granted on Hugging Face."
        )
    manifest = load_manifest("officeqa")
    out_root = DATA / "officeqa_split"
    revision = "8ecbf18d3833daf4750a903d14963e4c4c1d4cd8"

    log("officeqa: downloading officeqa_full.csv ...")
    local = hf_hub_download(
        "databricks/officeqa", filename="officeqa_full.csv",
        repo_type="dataset", revision=revision, token=token,
    )
    by_uid: dict[str, dict] = {}
    with open(local, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = str(row.get("uid") or row.get("id") or "").strip()
            if uid:
                by_uid[uid] = row

    for split, rows in manifest.items():
        out_rows, missing = [], 0
        split_dir = ensure_dir(out_root / split)
        for m in rows:
            uid = str(m.get("uid") or m.get("id"))
            r = by_uid.get(uid)
            if r is None:
                missing += 1
                continue
            out_rows.append(
                {
                    "uid": uid,
                    "question": r.get("question", ""),
                    "ground_truth": r.get("ground_truth") or r.get("answer") or "",
                    "category": m.get("category") or r.get("category", ""),
                    "source_files": m.get("source_files", ""),
                    "source_docs": m.get("source_docs", ""),
                    "split": split,
                }
            )
        if missing:
            log(f"  WARNING {split}: {missing} uids not found upstream")
        out_csv = split_dir / "items.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_OFFICEQA_COLUMNS)
            w.writeheader()
            w.writerows(out_rows)
        log(f"  wrote {split}: {len(out_rows)} -> {out_csv}")

    log(
        "  NOTE: OfficeQA also needs the offline document corpus at "
        "data/officeqa_docs_official/ (config env.data_dirs). Place the supporting "
        "documents there separately; this script only materializes the QA split."
    )


# ── SpreadsheetBench ──────────────────────────────────────────────────────

def _find_instruction_index(root: Path) -> dict[str, dict]:
    """Scan extracted SpreadsheetBench payload for per-task instruction records."""
    index: dict[str, dict] = {}
    for jf in sorted(root.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = list(data.values())
        else:
            records = []
        for rec in records:
            if isinstance(rec, dict) and "id" in rec and "instruction" in rec:
                index[str(rec["id"])] = rec
    return index


def build_spreadsheetbench(token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    manifest = load_manifest("spreadsheetbench")
    out_root = DATA / "spreadsheetbench_split"
    data_root = ensure_dir(DATA / "spreadsheetbench_verified_400")
    revision = "ab0b742b0fc95b946f212d80ac7771b5531272e4"

    log("spreadsheetbench: downloading spreadsheetbench_verified_400.tar.gz ...")
    tar_path = hf_hub_download(
        "KAKA22/SpreadsheetBench", filename="spreadsheetbench_verified_400.tar.gz",
        repo_type="dataset", revision=revision, token=token,
    )
    log(f"  extracting -> {data_root}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_root)

    # If the tar nested everything under a single top dir, follow it so that
    # spreadsheet_path (e.g. "spreadsheet/<id>") resolves under data_root.
    if not (data_root / "spreadsheet").exists():
        for child in sorted(data_root.iterdir()):
            if child.is_dir() and (child / "spreadsheet").exists():
                log(f"  detected payload root: {child}")
                data_root = child
                break

    instr = _find_instruction_index(data_root)
    log(f"  found instruction records for {len(instr)} tasks")

    for split, rows in manifest.items():
        items, missing = [], 0
        for m in rows:
            tid = str(m["id"])
            rec = instr.get(tid, {})
            if not rec:
                missing += 1
            items.append(
                {
                    "id": tid,
                    "instruction": rec.get("instruction", ""),
                    "instruction_type": m.get("instruction_type") or rec.get("instruction_type", ""),
                    "answer_position": rec.get("answer_position", ""),
                    "answer_sheet": rec.get("answer_sheet", ""),
                    "spreadsheet_path": m.get("spreadsheet_path", f"spreadsheet/{tid}"),
                }
            )
        if missing:
            log(f"  WARNING {split}: {missing} task ids had no instruction record")
        write_json_split(out_root, split, items)

    if data_root != DATA / "spreadsheetbench_verified_400":
        log(
            "  NOTE: payload extracted under a nested dir. Point env.data_root at "
            f"'{data_root.relative_to(REPO_ROOT)}' or move its contents up one level."
        )


# ── driver ────────────────────────────────────────────────────────────────

BUILDERS = {
    "searchqa": build_searchqa,
    "livemathematicianbench": build_livemathematicianbench,
    "docvqa": build_docvqa,
    "officeqa": build_officeqa,
    "spreadsheetbench": build_spreadsheetbench,
}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", default="", help="comma-separated benchmarks to build (default: all non-gated)")
    p.add_argument("--skip", default="", help="comma-separated benchmarks to skip")
    p.add_argument("--hf-token", default=None, help="Hugging Face token (needed for gated officeqa)")
    args = p.parse_args()

    token = resolve_token(args.hf_token)

    if args.only:
        selected = [b.strip() for b in args.only.split(",") if b.strip()]
    else:
        # Default: everything except the gated OfficeQA.
        selected = [b for b in ALL_BENCHMARKS if b != "officeqa"]
    skip = {b.strip() for b in args.skip.split(",") if b.strip()}
    selected = [b for b in selected if b not in skip]

    unknown = [b for b in selected if b not in BUILDERS]
    if unknown:
        log(f"Unknown benchmark(s): {unknown}. Valid: {list(BUILDERS)}")
        return 2

    if os.environ.get("HF_ENDPOINT"):
        log(f"Using HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    log(f"Building: {selected}")

    results: dict[str, str] = {}
    for bench in selected:
        log(f"===== {bench} =====")
        try:
            BUILDERS[bench](token)
            results[bench] = "OK"
        except Exception as e:  # one benchmark failing must not abort the rest
            results[bench] = f"FAILED: {e}"
            traceback.print_exc()

    log("===== summary =====")
    for bench in selected:
        log(f"  {bench}: {results.get(bench)}")
    log("ALFWorld is not handled here: use `alfworld-download` + $ALFWORLD_DATA (see data/README.md).")
    return 0 if all(v == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
