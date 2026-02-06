#!/usr/bin/env python3
"""
根据 prescreen 输出的 CSV（scripts/prescreen_faa_for_arg.py）批量清理抽样得到的 .faa 文件：
- 去掉每条蛋白序列末尾 '*'（stop codon）

使用场景：
- 先对超大目录做 N=50 预筛（不必全量清理）
- 再只对抽中的这 50 个文件做清理，生成一个小规模 cleaned 子集用于评估/推理

输入 CSV 预期列（至少包含）：
- file: 原始 .faa 文件绝对路径
- strict_q / relaxed_q / error: 用于筛选（可选）
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


FastaRecord = Tuple[str, str]  # (header, seq)


def iter_fasta(path: str) -> Iterator[FastaRecord]:
    header = None
    seq_chunks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def write_fasta(records: Iterator[FastaRecord], out_path: str, wrap: int = 60) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            if wrap <= 0:
                out.write(seq + "\n")
                continue
            for i in range(0, len(seq), wrap):
                out.write(seq[i : i + wrap] + "\n")


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def clean_one_file(in_path: str, out_path: str, wrap: int, internal_policy: str) -> Dict[str, int]:
    n_in = 0
    n_out = 0
    n_drop = 0
    n_internal = 0
    n_trim_trailing = 0

    def gen() -> Iterator[FastaRecord]:
        nonlocal n_in, n_out, n_drop, n_internal, n_trim_trailing
        for header, seq in iter_fasta(in_path):
            n_in += 1
            s = seq.strip().upper()
            trimmed = s.rstrip("*")
            if len(trimmed) != len(s):
                n_trim_trailing += 1
            s = trimmed

            if "*" in s:
                n_internal += 1
                if internal_policy == "error":
                    raise ValueError(f"检测到内部 '*'：{header}")
                if internal_policy == "drop":
                    n_drop += 1
                    continue
                if internal_policy == "replaceX":
                    s = s.replace("*", "X")

            if not s:
                n_drop += 1
                continue

            n_out += 1
            yield header, s

    write_fasta(gen(), out_path, wrap=wrap)
    return {
        "seq_in": n_in,
        "seq_out": n_out,
        "seq_dropped": n_drop,
        "trailing_trimmed": n_trim_trailing,
        "internal_seen": n_internal,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 prescreen CSV 批量清理抽样得到的 .faa 文件（去末尾 *）。")
    p.add_argument("--csv", required=True, help="prescreen 输出 CSV 路径")
    p.add_argument("--out-dir", required=True, help="输出 cleaned 文件目录")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少个文件（0 表示全部）")
    p.add_argument("--strict-min", type=int, default=0, help="只处理 strict_q >= 该值的文件（默认 0 表示不过滤）")
    p.add_argument("--skip-errors", action="store_true", help="跳过 error 列非空的行（推荐开启）")
    p.add_argument("--wrap", type=int, default=60, help="FASTA 输出换行宽度（默认 60；<=0 表示不换行）")
    p.add_argument(
        "--internal-policy",
        choices=["error", "drop", "replaceX"],
        default="error",
        help="内部 '*' 的处理策略：error(默认) / drop / replaceX",
    )
    p.add_argument("--suffix", default="", help="输出文件名后缀（默认空；例如 _clean）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    picked: List[str] = []
    for row in rows:
        if args.skip_errors and row.get("error", "").strip():
            continue
        if safe_int(row.get("strict_q", "0"), 0) < args.strict_min:
            continue
        fpath = row.get("file", "").strip()
        if not fpath:
            continue
        picked.append(fpath)

    if args.limit and args.limit > 0:
        picked = picked[: args.limit]

    if not picked:
        raise SystemExit("未选中任何文件：请检查 --strict-min / --skip-errors / --limit 以及 CSV 内容。")

    total = {"seq_in": 0, "seq_out": 0, "seq_dropped": 0, "trailing_trimmed": 0, "internal_seen": 0}
    failed = 0

    for i, in_path in enumerate(picked, 1):
        p = Path(in_path)
        if not p.is_file():
            failed += 1
            print(f"[fail] missing file: {in_path}")
            continue

        out_name = p.stem + args.suffix + p.suffix
        out_path = str(out_dir / out_name)

        try:
            st = clean_one_file(in_path, out_path, wrap=args.wrap, internal_policy=args.internal_policy)
        except Exception as e:
            failed += 1
            print(f"[fail] {in_path} :: {e}")
            continue

        for k in total:
            total[k] += st[k]

        if i % 10 == 0 or i == len(picked):
            print(f"[progress] {i}/{len(picked)} files; total_out_seqs={total['seq_out']} failed={failed}")

    print(f"[in ] csv: {args.csv}")
    print(f"[out] dir: {args.out_dir}")
    print(f"[cnt] files={len(picked)} failed={failed}")
    print(f"[cnt] seq_in={total['seq_in']} seq_out={total['seq_out']} dropped={total['seq_dropped']}")
    print(f"[cnt] trailing_stop_trimmed={total['trailing_trimmed']} internal_stop_seen={total['internal_seen']} internal_policy={args.internal_policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

