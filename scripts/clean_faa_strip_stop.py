#!/usr/bin/env python3
"""
清理 Prodigal/翻译得到的蛋白 FASTA/FAA（去掉终止符 '*'）。

背景：
- 常见现象：每条序列末尾带有 '*'，表示终止密码子（stop codon）。
- 对模型训练/推理/长度统计：通常建议去掉末尾 '*'，避免引入额外 token 或扰动长度统计。

支持两种用法：
1) 单文件：--in FILE --out FILE
2) 目录批处理：--in DIR --out DIR（可选 --recursive；默认输出文件名与输入一致）

默认策略：
- 去掉序列末尾连续的 '*'（只处理 trailing stop）。

可选策略（内部 '*'，即非末尾 stop）：
- --internal-policy error：检测到内部 '*' 直接报错退出（默认）
- --internal-policy drop：丢弃该序列
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
    p = argparse.ArgumentParser(description="清理 .faa/.fasta 的终止符 '*'（默认仅去掉末尾 '*')。支持单文件或目录批处理。")
    p.add_argument("--in", dest="in_path", required=True, help="输入 FASTA/FAA 文件路径，或包含 .faa 的目录")
    p.add_argument("--out", dest="out_path", required=True, help="输出 FASTA 路径，或输出目录")
    p.add_argument("--wrap", type=int, default=60, help="FASTA 输出换行宽度（默认 60；<=0 表示不换行）")
    p.add_argument(
        "--internal-policy",
        choices=["error", "drop", "replaceX"],
        default="error",
        help="内部 '*' 的处理策略：error(默认) / drop / replaceX",
    )
    p.add_argument("--recursive", action="store_true", help="目录模式下：递归处理子目录里的 .faa/.fasta")
    p.add_argument(
        "--ext",
        default=".faa",
        help="目录模式下：匹配的扩展名（默认 .faa；也可设为 .fasta）",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="目录模式下：输出文件名后缀（默认空；例如 _clean）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许在输出目录与输入目录相同的情况下覆盖写入（谨慎使用）",
    )
    return p.parse_args()


def clean_one_file(
    in_path: str,
    out_path: str,
    wrap: int,
    internal_policy: str,
) -> Dict[str, int]:
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

            # 1) 去掉末尾连续 '*'
            trimmed = s.rstrip("*")
            if len(trimmed) != len(s):
                n_trim_trailing += 1
            s = trimmed

            # 2) 处理内部 '*'
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
        "in": n_in,
        "out": n_out,
        "dropped": n_drop,
        "trailing_trimmed": n_trim_trailing,
        "internal_seen": n_internal,
    }


def main() -> int:
    args = parse_args()

    if os.path.isdir(args.in_path):
        in_dir = args.in_path
        out_dir = args.out_path

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        in_dir_abs = os.path.abspath(in_dir)
        out_dir_abs = os.path.abspath(out_dir)
        if in_dir_abs == out_dir_abs and not args.overwrite:
            raise SystemExit("输入目录与输出目录相同。若要原地覆盖处理，请加 --overwrite（不推荐）。")

        ext = args.ext
        if not ext.startswith("."):
            ext = "." + ext

        if args.recursive:
            files = [os.path.join(root, fn) for root, _, fns in os.walk(in_dir) for fn in fns if fn.endswith(ext)]
        else:
            files = [os.path.join(in_dir, fn) for fn in os.listdir(in_dir) if fn.endswith(ext)]

        files.sort()
        if not files:
            raise SystemExit(f"目录下未找到 {ext} 文件：{in_dir}")

        total = {"in": 0, "out": 0, "dropped": 0, "trailing_trimmed": 0, "internal_seen": 0}
        failed = 0

        for i, in_path in enumerate(files, 1):
            rel = os.path.relpath(in_path, in_dir)
            base, e = os.path.splitext(rel)
            out_rel = f"{base}{args.suffix}{e}"
            out_path = os.path.join(out_dir, out_rel)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            try:
                st = clean_one_file(
                    in_path=in_path,
                    out_path=out_path,
                    wrap=args.wrap,
                    internal_policy=args.internal_policy,
                )
            except Exception as e:
                failed += 1
                print(f"[fail] {in_path} :: {e}")
                continue

            for k in total:
                total[k] += st[k]

            if i % 50 == 0 or i == len(files):
                print(f"[progress] {i}/{len(files)} files; total_out_seqs={total['out']} failed={failed}")

        print(f"[in ] dir: {in_dir}")
        print(f"[out] dir: {out_dir}")
        print(f"[cnt] files={len(files)} failed={failed}")
        print(f"[cnt] seq_in={total['in']} seq_out={total['out']} dropped={total['dropped']}")
        print(f"[cnt] trailing_stop_trimmed={total['trailing_trimmed']} internal_stop_seen={total['internal_seen']} internal_policy={args.internal_policy}")
        return 0

    # 单文件模式
    st = clean_one_file(
        in_path=args.in_path,
        out_path=args.out_path,
        wrap=args.wrap,
        internal_policy=args.internal_policy,
    )
    print(f"[in ] {args.in_path}")
    print(f"[out] {args.out_path}")
    print(f"[cnt] in={st['in']} out={st['out']} dropped={st['dropped']}")
    print(f"[cnt] trailing_stop_trimmed={st['trailing_trimmed']} internal_stop_seen={st['internal_seen']} internal_policy={args.internal_policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
