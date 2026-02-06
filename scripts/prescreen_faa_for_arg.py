#!/usr/bin/env python3
"""
从一个包含大量 .faa 文件的目录中随机抽样 N 个文件，用 DIAMOND 预筛选：
- 统计每个文件中满足 silver standard 阈值的“strict ARG query 数量”（以及 relaxed 命中数量）
- 目的：快速找到“确实含 ARG”的真实样本文件，用于后续 B 真实性能评估（避免抽到全阴性文件只能测假阳性率）

说明：
- 本脚本只负责“挑文件”，不做最终评估。
- 阈值与 `scripts/eval_silver_standard_v0.py` 保持一致（--mode full/orf）。
- 为了加速后续评估，可选保留每个文件的 hits TSV。

依赖：
- Python 标准库
- 外部命令：diamond
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def is_strict_arg(h: Hit, mode: str) -> bool:
    if mode == "orf":
        return (h.evalue <= 1e-10) and (h.pident >= 80.0) and (h.qcov >= 0.80) and (h.scov >= 0.30)
    return (h.evalue <= 1e-10) and (h.pident >= 80.0) and (h.mincov >= 0.80)


def is_relaxed_hit(h: Hit, mode: str) -> bool:
    if mode == "orf":
        return (h.evalue <= 1e-5) and (h.pident >= 30.0) and (h.qcov >= 0.50) and (h.scov >= 0.20)
    return (h.evalue <= 1e-5) and (h.pident >= 30.0) and (h.mincov >= 0.50)


def parse_hits_tsv(path: str) -> Iterable[Hit]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                parts = line.split()
            if len(parts) < 8:
                continue
            q, s = parts[0], parts[1]
            try:
                pident = float(parts[2])
                alnlen = int(float(parts[3]))
                qlen = int(float(parts[4]))
                slen = int(float(parts[5]))
                evalue = float(parts[6])
                bits = float(parts[7])
            except Exception:
                continue
            yield Hit(q, s, pident, alnlen, qlen, slen, evalue, bits)


def list_faa_files(dir_path: str, recursive: bool) -> List[str]:
    root = Path(dir_path)
    if recursive:
        files = [str(p) for p in root.rglob("*.faa")]
    else:
        files = [str(p) for p in root.glob("*.faa")]
    return sorted(files)


def sample_files(files: List[str], n: int, seed: int) -> List[str]:
    if n <= 0:
        return []
    if n >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, n)


def run_diamond(
    query_faa: str,
    db: str,
    out_tsv: str,
    threads: int,
    max_target_seqs: int,
    evalue: float,
) -> None:
    cmd = [
        "diamond",
        "blastp",
        "-q",
        query_faa,
        "-d",
        db,
        "-o",
        out_tsv,
        "-p",
        str(threads),
        "--evalue",
        str(evalue),
        "--max-target-seqs",
        str(max_target_seqs),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="随机抽 N 个 .faa，用 DIAMOND 预筛 strict ARG 数量。")
    p.add_argument("--dir", required=True, help="包含大量 .faa 的目录")
    p.add_argument("--db", required=True, help="DIAMOND 数据库前缀（makedb -d 的值，不含 .dmnd 也可）")
    p.add_argument("--out", required=True, help="输出汇总 CSV 路径")
    p.add_argument("--n", type=int, default=50, help="抽样文件数（默认 50）")
    p.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    p.add_argument("--mode", choices=["full", "orf"], default="orf", help="阈值模式（默认 orf）")
    p.add_argument("--threads", type=int, default=16, help="diamond 线程数（默认 16）")
    p.add_argument("--max-target-seqs", type=int, default=10, help="每条 query 保留的 target 数（默认 10）")
    p.add_argument("--evalue", type=float, default=1e-5, help="diamond 输出的 evalue 阈值（默认 1e-5）")
    p.add_argument("--recursive", action="store_true", help="递归搜索子目录下的 .faa")
    p.add_argument("--keep-hits-dir", default="", help="若提供目录，则保留每个文件的 hits TSV 到该目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    files = list_faa_files(args.dir, recursive=args.recursive)
    if not files:
        raise SystemExit(f"未找到 .faa 文件：{args.dir} (recursive={args.recursive})")

    picked = sample_files(files, args.n, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keep_dir = args.keep_hits_dir
    if keep_dir:
        os.makedirs(keep_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []

    print(f"[scan] total_faa={len(files)} picked={len(picked)} seed={args.seed} mode={args.mode}")

    for idx, fpath in enumerate(picked, 1):
        t0 = time.time()
        tmp_out = f"{args.out}.tmp_{idx}.tsv"
        try:
            run_diamond(
                query_faa=fpath,
                db=args.db,
                out_tsv=tmp_out,
                threads=args.threads,
                max_target_seqs=args.max_target_seqs,
                evalue=args.evalue,
            )
        except subprocess.CalledProcessError as e:
            rows.append(
                {
                    "idx": str(idx),
                    "file": fpath,
                    "strict_q": "0",
                    "relaxed_q": "0",
                    "hits": "0",
                    "seconds": f"{time.time()-t0:.3f}",
                    "error": f"diamond_failed:{e.returncode}",
                }
            )
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            continue

        strict_q = set()
        relaxed_q = set()
        hits = 0
        for h in parse_hits_tsv(tmp_out):
            hits += 1
            if is_relaxed_hit(h, args.mode):
                relaxed_q.add(h.q)
            if is_strict_arg(h, args.mode):
                strict_q.add(h.q)

        if keep_dir:
            base = Path(fpath).name
            keep_path = str(Path(keep_dir) / f"{base}.diamond.tsv")
            os.replace(tmp_out, keep_path)
            saved_hits = keep_path
        else:
            os.remove(tmp_out)
            saved_hits = ""

        rows.append(
            {
                "idx": str(idx),
                "file": fpath,
                "strict_q": str(len(strict_q)),
                "relaxed_q": str(len(relaxed_q)),
                "hits": str(hits),
                "seconds": f"{time.time()-t0:.3f}",
                "saved_hits": saved_hits,
                "error": "",
            }
        )

        if idx % 5 == 0 or idx == len(picked):
            print(f"[progress] {idx}/{len(picked)} last_strict={len(strict_q)} last_relaxed={len(relaxed_q)}")

    # sort by strict desc, then relaxed desc
    def key(r: Dict[str, str]) -> Tuple[int, int]:
        return (int(r.get("strict_q", "0")), int(r.get("relaxed_q", "0")))

    top = sorted(rows, key=key, reverse=True)[:10]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["idx", "file", "strict_q", "relaxed_q", "hits", "seconds", "saved_hits", "error"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[out] summary_csv: {args.out}")
    print("[top10] by strict_q then relaxed_q:")
    for r in top:
        print(f"  strict={r['strict_q']:<4} relaxed={r['relaxed_q']:<4} hits={r['hits']:<6} file={r['file']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

