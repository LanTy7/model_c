#!/usr/bin/env python3
"""
清理 Prodigal/翻译得到的蛋白 FASTA/FAA：

- 常见现象：每条序列末尾带有 '*'，表示终止密码子（stop codon）。
- 对模型训练/推理：通常建议去掉末尾 '*'，避免引入额外 token 或扰动长度统计。

默认策略：
- 去掉序列末尾连续的 '*'（只处理 trailing stop）。

可选策略：
- --internal-policy error：若序列内部出现 '*'（非末尾），直接报错退出（默认）
- --internal-policy drop：若内部出现 '*', 丢弃该序列
- --internal-policy replaceX：将内部 '*' 替换为 'X'
"""

from __future__ import annotations

import argparse
import os
from typing import Iterator, List, Tuple


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清理 .faa/.fasta 的终止符 '*'（默认仅去掉末尾 '*')。")
    p.add_argument("--in", dest="in_path", required=True, help="输入 FASTA/FAA 路径")
    p.add_argument("--out", dest="out_path", required=True, help="输出 FASTA 路径")
    p.add_argument("--wrap", type=int, default=60, help="FASTA 输出换行宽度（默认 60；<=0 表示不换行）")
    p.add_argument(
        "--internal-policy",
        choices=["error", "drop", "replaceX"],
        default="error",
        help="内部 '*' 的处理策略：error(默认) / drop / replaceX",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    n_in = 0
    n_out = 0
    n_drop = 0
    n_internal = 0
    n_trim_trailing = 0

    def gen() -> Iterator[FastaRecord]:
        nonlocal n_in, n_out, n_drop, n_internal, n_trim_trailing
        for header, seq in iter_fasta(args.in_path):
            n_in += 1
            s = seq.strip().upper()

            # 1) 去掉末尾连续 '*'
            trimmed = s.rstrip("*")
            if len(trimmed) != len(s):
                n_trim_trailing += 1
            s = trimmed

            # 2) 处理内部 '*'
            if "*" in s:
                n_internal += 1
                if args.internal_policy == "error":
                    raise ValueError(f"检测到内部 '*'：{header}")
                if args.internal_policy == "drop":
                    n_drop += 1
                    continue
                if args.internal_policy == "replaceX":
                    s = s.replace("*", "X")

            if not s:
                n_drop += 1
                continue

            n_out += 1
            yield header, s

    write_fasta(gen(), args.out_path, wrap=args.wrap)

    print(f"[in ] {args.in_path}")
    print(f"[out] {args.out_path}")
    print(f"[cnt] in={n_in} out={n_out} dropped={n_drop}")
    print(f"[cnt] trailing_stop_trimmed={n_trim_trailing} internal_stop_seen={n_internal} internal_policy={args.internal_policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

