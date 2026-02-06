#!/usr/bin/env python3
"""
基于同源比对（DIAMOND/MMseqs2 导出的 TSV）构建 silver standard 参考标签，并与模型输出合并评估。

定位：
- 支持论文里的“真实性能评估(B)”的第一版可复现实现。
- 重点是流程跑通 + 指标可计算；后续可扩展更多分组/更严格的泄漏控制。

输入要求：
- ARG 参考库 FASTA：用于把 subject id 映射到 ARG 类别（取 header 最后一个 '|' 字段）。
- Hits TSV：至少包含列：
  qseqid sseqid pident alnlen qlen slen evalue bitscore
- 二分类输出 CSV：来自 binary/model_test/predict.ipynb 的 *_scores.csv
- 多分类输出 CSV：来自 multi/model_test/classify.ipynb 的 classification_results.csv（可选）

silver standard v0（默认）：
- strict ARG:
  evalue <= 1e-10 AND pident >= 80 AND min(qcov, scov) >= 0.80
- strict non-ARG:
  在 relaxed 阈值下完全无命中才标 non-ARG
  relaxed: evalue <= 1e-5 AND pident >= 30 AND min(qcov, scov) >= 0.50
- 其它：unlabeled（可在二分类主评估中排除，或可选当作 non-ARG）
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="silver standard v0 + 模型输出合并评估（真实场景测试 B）")
    p.add_argument("--arg-db", required=True, help="ARG 参考库 FASTA（subject -> class 映射）")
    p.add_argument("--hits", required=True, help="同源比对 hits TSV（qseqid sseqid pident alnlen qlen slen evalue bitscore）")
    p.add_argument("--binary-csv", required=True, help="二分类输出 *_scores.csv")
    p.add_argument("--multi-csv", default="", help="多分类输出 classification_results.csv（可选）")
    p.add_argument("--multi-ckpt", default="", help="多分类 ckpt .pth（可选，用于类别体系对齐/映射到 Others）")
    p.add_argument("--out", required=True, help="输出合并后的评估明细 CSV")
    p.add_argument(
        "--unlabeled",
        choices=["exclude", "as_nonarg"],
        default="exclude",
        help="二分类评估时如何处理 unlabeled：exclude(默认) / as_nonarg",
    )
    return p.parse_args()


def iter_fasta_headers(path: str) -> Iterable[Tuple[str, str]]:
    """yield (seq_id, full_header_without_gt)"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            hdr = line[1:].strip()
            if not hdr:
                continue
            seq_id = hdr.split()[0]  # 与 DIAMOND/MMseqs2 默认 id 对齐（first token）
            yield seq_id, hdr


def build_subject_class_map(arg_db_fasta: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for sid, hdr in iter_fasta_headers(arg_db_fasta):
        parts = hdr.split("|")
        raw = parts[-1].strip() if parts else ""
        cls = raw.split()[0] if raw else ""
        if cls:
            m[sid] = cls
    return m


def try_load_class_names_from_ckpt(path: str) -> Optional[List[str]]:
    if not path:
        return None
    try:
        import torch  # type: ignore
    except Exception:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    names = ckpt.get("class_names")
    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        return names
    return None


@dataclass
class Hit:
    q: str
    s: str
    pident: float
    alnlen: int
    qlen: int
    slen: int
    evalue: float
    bits: float

    @property
    def qcov(self) -> float:
        return self.alnlen / self.qlen if self.qlen else 0.0

    @property
    def scov(self) -> float:
        return self.alnlen / self.slen if self.slen else 0.0

    @property
    def mincov(self) -> float:
        return min(self.qcov, self.scov)


def parse_hits_tsv(path: str) -> List[Hit]:
    hits: List[Hit] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                # allow space-separated
                parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"hits TSV 列数不足（需要 8 列）：{line[:200]}")
            q, s = parts[0], parts[1]
            pident = float(parts[2])
            alnlen = int(float(parts[3]))
            qlen = int(float(parts[4]))
            slen = int(float(parts[5]))
            evalue = float(parts[6])
            bits = float(parts[7])
            hits.append(Hit(q, s, pident, alnlen, qlen, slen, evalue, bits))
    return hits


def is_strict_arg(h: Hit) -> bool:
    return (h.evalue <= 1e-10) and (h.pident >= 80.0) and (h.mincov >= 0.80)


def is_relaxed_hit(h: Hit) -> bool:
    return (h.evalue <= 1e-5) and (h.pident >= 30.0) and (h.mincov >= 0.50)


def choose_best_hit(hits: List[Hit]) -> Hit:
    # bitscore 优先，其次 pident、mincov
    return sorted(hits, key=lambda x: (x.bits, x.pident, x.mincov), reverse=True)[0]


def build_silver_labels(
    hits: List[Hit],
    subj2class: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """
    return:
      q -> {
        ref_label: ARG/non-ARG/unlabeled/ambiguous
        ref_class: class name or empty
        ref_best_s: best subject id or empty
      }
    """
    by_q: Dict[str, List[Hit]] = defaultdict(list)
    for h in hits:
        by_q[h.q].append(h)

    out: Dict[str, Dict[str, str]] = {}
    for q, hs in by_q.items():
        strict = [h for h in hs if is_strict_arg(h)]
        relaxed = [h for h in hs if is_relaxed_hit(h)]

        if strict:
            # map to classes (unknown subject -> empty class)
            best = choose_best_hit(strict)
            classes = {subj2class.get(h.s, "") for h in strict}
            classes.discard("")
            if len(classes) >= 2:
                out[q] = {"ref_label": "ambiguous", "ref_class": "", "ref_best_s": best.s}
            else:
                cls = subj2class.get(best.s, "")
                out[q] = {"ref_label": "ARG", "ref_class": cls, "ref_best_s": best.s}
        else:
            if not relaxed:
                out[q] = {"ref_label": "non-ARG", "ref_class": "", "ref_best_s": ""}
            else:
                best = choose_best_hit(relaxed)
                out[q] = {"ref_label": "unlabeled", "ref_class": "", "ref_best_s": best.s}
    return out


def read_binary_scores(path: str) -> Dict[str, Dict[str, str]]:
    m: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row["SequenceID"]
            m[q] = row
    return m


def read_multi_preds(path: str) -> Dict[str, Dict[str, str]]:
    m: Dict[str, Dict[str, str]] = {}
    if not path:
        return m
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row.get("SequenceID") or row.get("seq_id") or row.get("id")
            if not q:
                continue
            m[q] = row
    return m


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def confusion(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def prf(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true))
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main() -> int:
    args = parse_args()
    subj2class = build_subject_class_map(args.arg_db)
    class_names = try_load_class_names_from_ckpt(args.multi_ckpt) or []

    hits = parse_hits_tsv(args.hits)
    silver = build_silver_labels(hits, subj2class)

    bin_rows = read_binary_scores(args.binary_csv)
    multi_rows = read_multi_preds(args.multi_csv)

    # Merge
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "SequenceID",
        "SeqLen",
        "ARG_Prob",
        "ARG_Pred",
        "ref_label",
        "ref_class",
        "ref_best_s",
        "multi_pred_class",
        "multi_prob",
    ]

    merged: List[Dict[str, str]] = []
    ref_counts = Counter()
    q_in_hits = 0

    for q, r in bin_rows.items():
        # 注意：DIAMOND/MMseqs2 的 TSV 默认不会输出“无命中”的 query。
        # 对于 silver standard v0 的 strict non-ARG 定义，“没有任何 relaxed 命中”应标为 non-ARG。
        # 因此：若某条 query 不在 hits TSV 中，默认标为 non-ARG（而不是 unlabeled）。
        ref = silver.get(q, {"ref_label": "non-ARG", "ref_class": "", "ref_best_s": ""})
        if q in silver:
            q_in_hits += 1
        ref_label = ref["ref_label"]
        ref_class = ref["ref_class"]
        ref_best_s = ref["ref_best_s"]

        # 对齐多分类类别体系：不在 class_names 的映射到 Others（若存在）
        if class_names and ref_label == "ARG" and ref_class and (ref_class not in class_names):
            if "Others" in class_names:
                ref_class = "Others"

        m = multi_rows.get(q, {})
        mp = m.get("PredictedClass", "")
        mprob = m.get("Probability", "")

        row = {
            "SequenceID": q,
            "SeqLen": str(r.get("SeqLen", "")),
            "ARG_Prob": str(r.get("ARG_Prob", "")),
            "ARG_Pred": str(r.get("ARG_Pred", "")),
            "ref_label": ref_label,
            "ref_class": ref_class,
            "ref_best_s": ref_best_s,
            "multi_pred_class": mp,
            "multi_prob": mprob,
        }
        merged.append(row)
        ref_counts[ref_label] += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in merged:
            w.writerow(row)

    print("[silver] queries in hits TSV:", len(silver), "matched_to_binary_ids:", q_in_hits, "binary_total:", len(bin_rows))
    print("[silver] label counts:", dict(ref_counts))
    print("[out] merged csv:", args.out)

    # Binary metrics on strict set
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    for row in merged:
        ref_label = row["ref_label"]
        if ref_label in ("ARG", "non-ARG"):
            t = 1 if ref_label == "ARG" else 0
        elif ref_label == "unlabeled" and args.unlabeled == "as_nonarg":
            t = 0
        else:
            continue

        p = safe_int(row["ARG_Pred"], 0)
        s = safe_float(row["ARG_Prob"], 0.0)
        y_true.append(t)
        y_pred.append(p)
        y_score.append(s)

    cm = confusion(y_true, y_pred)
    m = prf(cm)
    print("\n[binary] evaluated_n =", len(y_true), f"(unlabeled={args.unlabeled})")
    print("[binary] cm =", cm)
    print("[binary] metrics =", {k: round(v, 6) for k, v in m.items()})

    # Optional AUCs if sklearn available
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore

        if len(set(y_true)) == 2:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            print("[binary] AUROC =", round(float(auroc), 6))
            print("[binary] AUPRC =", round(float(auprc), 6))
    except Exception:
        pass

    # Multi metrics (conditional on predicted positive & strict ARG)
    y_true_cls: List[str] = []
    y_pred_cls: List[str] = []
    missing_multi = 0

    for row in merged:
        if row["ref_label"] != "ARG":
            continue
        if safe_int(row["ARG_Pred"], 0) != 1:
            continue
        if not row["ref_class"]:
            continue
        if not row["multi_pred_class"]:
            missing_multi += 1
            continue
        y_true_cls.append(row["ref_class"])
        y_pred_cls.append(row["multi_pred_class"])

    print("\n[multi] conditional evaluated_n =", len(y_true_cls), "missing_multi_pred =", missing_multi)
    if y_true_cls:
        mf1 = macro_f1(y_true_cls, y_pred_cls)
        acc = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == p) / len(y_true_cls)
        print("[multi] accuracy =", round(acc, 6))
        print("[multi] macro_f1 =", round(mf1, 6))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
