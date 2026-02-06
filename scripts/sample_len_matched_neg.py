#!/usr/bin/env python3
"""
按长度分布匹配（分层抽样）从超大阴性 .faa/.fasta 中采样出训练用阴性集。

设计目标：
- 仅依赖标准库（不需要 biopython/numpy），便于在不同环境快速复现。
- 流式读取阴性大文件，按长度 bin 做 reservoir sampling，避免一次性载入全部序列。
- 长度分布参考阳性集，并与训练时的 max_len 对齐（长度按 min(len, max_len) 计算）。
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Tuple


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


def write_fasta(records: Iterable[FastaRecord], out_path: str, wrap: int = 60) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            if wrap <= 0:
                out.write(seq + "\n")
                continue
            for i in range(0, len(seq), wrap):
                out.write(seq[i : i + wrap] + "\n")


def length_bin(length: int, bin_size: int, max_len: int) -> int:
    length = min(length, max_len)
    return (length // bin_size) * bin_size  # 0, bin_size, 2*bin_size, ...


def compute_pos_bins(pos_fasta: str, bin_size: int, max_len: int) -> Counter:
    bins = Counter()
    for _, seq in iter_fasta(pos_fasta):
        bins[length_bin(len(seq), bin_size, max_len)] += 1
    return bins


def build_targets(pos_bins: Counter, ratio: float) -> Dict[int, int]:
    targets: Dict[int, int] = {}
    for b, c in pos_bins.items():
        k = int(math.floor(c * ratio))
        if c > 0 and ratio > 0 and k == 0:
            k = 1  # 防止极小 bin 在 ratio<1 时被直接归零
        targets[b] = k
    return targets


def reservoir_sample_by_bin(
    neg_fasta: str,
    targets: Dict[int, int],
    bin_size: int,
    max_len: int,
    rng: random.Random,
) -> Tuple[Dict[int, List[FastaRecord]], Counter, Counter]:
    reservoirs: Dict[int, List[FastaRecord]] = {b: [] for b in targets}
    seen = Counter()  # seen records per bin (eligible negatives)
    total_seen = Counter()  # all negatives per bin (even if target is 0)

    for header, seq in iter_fasta(neg_fasta):
        b = length_bin(len(seq), bin_size, max_len)
        total_seen[b] += 1

        k = targets.get(b, 0)
        if k <= 0:
            continue

        seen[b] += 1
        res = reservoirs[b]
        if len(res) < k:
            res.append((header, seq))
            continue

        # Algorithm R (reservoir sampling)
        j = rng.randrange(seen[b])
        if j < k:
            res[j] = (header, seq)

    return reservoirs, seen, total_seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从阴性大文件中按阳性长度分布匹配采样阴性序列（分层 reservoir sampling）。"
    )
    p.add_argument("--pos", required=True, help="阳性 FASTA 路径（ARG 序列）")
    p.add_argument("--neg", required=True, help="阴性 FASTA/FAA 路径（已去污染的非 ARG 序列）")
    p.add_argument("--out", required=True, help="输出采样后的阴性 FASTA 路径")
    p.add_argument("--ratio", type=float, default=3.0, help="阴性/阳性采样比例（默认 3.0）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument("--bin-size", type=int, default=50, help="长度分箱大小（默认 50 aa）")
    p.add_argument("--max-len", type=int, default=1000, help="长度截断上限（应与训练 max_length 一致）")
    p.add_argument("--wrap", type=int, default=60, help="FASTA 输出换行宽度（默认 60；<=0 表示不换行）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    pos_bins = compute_pos_bins(args.pos, args.bin_size, args.max_len)
    targets = build_targets(pos_bins, args.ratio)

    n_pos = sum(pos_bins.values())
    n_target = sum(targets.values())
    print(f"[pos] n={n_pos}")
    print(f"[neg] target ratio={args.ratio} => n_target={n_target}")
    print(f"[cfg] bin_size={args.bin_size} max_len={args.max_len} seed={args.seed}")

    reservoirs, seen, total_seen = reservoir_sample_by_bin(
        args.neg, targets, args.bin_size, args.max_len, rng
    )

    sampled: List[FastaRecord] = []
    missing_bins = 0
    for b in sorted(targets.keys()):
        need = targets[b]
        got = len(reservoirs[b])
        if got < need:
            missing_bins += 1
        sampled.extend(reservoirs[b])

    rng.shuffle(sampled)
    write_fasta(sampled, args.out, wrap=args.wrap)

    print(f"[out] saved: {args.out}")
    print(f"[out] sampled_total={len(sampled)} bins_missing={missing_bins}")

    print("\n[bin] pos_count -> neg_target / neg_got (eligible_seen / total_seen_in_neg)")
    for b in sorted(pos_bins.keys()):
        pc = pos_bins[b]
        need = targets.get(b, 0)
        got = len(reservoirs.get(b, []))
        s = seen.get(b, 0)
        ts = total_seen.get(b, 0)
        print(f"  {b:4d}-{b+args.bin_size-1:4d}: {pc:6d} -> {need:6d} / {got:6d} ({s:7d} / {ts:7d})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

