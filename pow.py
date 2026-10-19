"""Proof of work: targets, mining (single and multiprocess) and retargeting."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chain import Block, Chain

#: Target at difficulty 1: about 2**19 hashes per block on average, which is
#: a few tenths of a second in pure Python.
DIFFICULTY_ONE_TARGET = 1 << 237

#: One retarget may change difficulty by at most this factor either way.
MAX_ADJUSTMENT = 4


# --------------------------------------------------------------------------- #
# Target <-> difficulty
# --------------------------------------------------------------------------- #


def target_from_difficulty(difficulty: int) -> int:
    if difficulty < 1:
        raise ValueError("difficulty must be at least 1")
    return DIFFICULTY_ONE_TARGET // difficulty


def difficulty_from_target(target: int) -> int:
    return max(1, DIFFICULTY_ONE_TARGET // target)


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #


def mine_range(
    block: Block,
    start: int,
    stop: int,
    step: int = 1,
    stop_flag=None,
    check_every: int = 4096,
) -> int | None:
    """Try nonces start, start+step, ... below stop. Return a winning nonce or None.

    stop_flag (a multiprocessing.Event) is checked only every `check_every`
    hashes, because checking it on every hash would cost more than the hash.
    """
    prefix = block.header_prefix()
    target = block.target
    sha = hashlib.sha256
    tried = 0
    for nonce in range(start, stop, step):
        digest = sha(prefix + str(nonce).encode()).digest()
        if int.from_bytes(digest, "big") < target:
            return nonce
        tried += 1
        if stop_flag is not None and tried % check_every == 0 and stop_flag.is_set():
            return None
    return None


def mine(block: Block, chunk: int = 1 << 20) -> Block:
    """Single-process miner: search nonces from 0 upward until one works."""
    start = 0
    while True:
        nonce = mine_range(block, start, start + chunk)
        if nonce is not None:
            return replace(block, nonce=nonce)
        start += chunk


def _worker(block: Block, index: int, workers: int, stop_flag, results) -> None:
    # worker i tries i, i+W, i+2W, ... so the workers never overlap
    nonce = mine_range(block, index, 1 << 62, workers, stop_flag)
    if nonce is not None:
        results.put(nonce)


def mine_parallel(block: Block, workers: int | None = None) -> Block:
    """Multiprocess miner (processes, not threads, because of the GIL).

    Returns once a nonce is found, after every worker process has exited.
    """
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        return mine(block)

    ctx = mp.get_context("spawn")
    stop_flag = ctx.Event()
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(block, i, workers, stop_flag, results), daemon=True)
        for i in range(workers)
    ]
    for p in procs:
        p.start()
    try:
        nonce = results.get()
    finally:
        stop_flag.set()  # tell the losers to stop
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    return replace(block, nonce=nonce)


# --------------------------------------------------------------------------- #
# Difficulty retargeting
# --------------------------------------------------------------------------- #


def retarget(old_difficulty: int, actual_seconds: float, expected_seconds: float) -> int:
    """New difficulty after a window that took actual_seconds instead of expected.

    target_new = target_old * actual / expected, and difficulty = 1 / target,
    so difficulty_new = difficulty_old * expected / actual.
    """
    if actual_seconds <= 0:
        ratio = float(MAX_ADJUSTMENT)
    else:
        ratio = expected_seconds / actual_seconds
    ratio = max(1 / MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, ratio))  # the clamp
    return max(1, round(old_difficulty * ratio))


# --------------------------------------------------------------------------- #
# Interval statistics
# --------------------------------------------------------------------------- #


@dataclass
class MiningStats:
    intervals: list[float] = field(default_factory=list)
    difficulties: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.intervals)

    @property
    def median(self) -> float:
        return statistics.median(self.intervals)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.intervals) if len(self.intervals) > 1 else 0.0

    def report(self, target_seconds: float | None = None) -> str:
        if not self.intervals:
            return "no blocks mined"
        lines = [
            f"blocks           : {len(self.intervals)}",
            f"mean interval    : {self.mean:.4f} s",
            f"median interval  : {self.median:.4f} s",
            f"stdev            : {self.stdev:.4f} s",
            f"min / max        : {min(self.intervals):.4f} s / {max(self.intervals):.4f} s",
            f"median / mean    : {self.median / self.mean:.3f}   (exponential: 0.693)",
            f"stdev / mean     : {self.stdev / self.mean:.3f}   (exponential: 1.000)",
            f"difficulties     : {self.difficulties}",
        ]
        if target_seconds is not None:
            lines.insert(1, f"target interval  : {target_seconds:.2f} s")
        return "\n".join(lines)


def simulate_mining(
    blocks: int = 30,
    workers: int | None = None,
    target_seconds: float = 0.30,
    window: int = 10,
    miner: str = "miner",
) -> tuple[Chain, MiningStats]:
    """Mine `blocks` blocks, retargeting every `window` blocks."""
    from .chain import Chain, make_block  # local import: chain.py imports this module

    chain = Chain()
    stats = MiningStats()
    difficulty = 1
    stats.difficulties.append(difficulty)
    workers = workers or 1

    for i in range(blocks):
        block = make_block(chain, miner=miner, difficulty=difficulty)
        started = time.perf_counter()
        mined = mine_parallel(block, workers) if workers > 1 else mine(block)
        stats.intervals.append(time.perf_counter() - started)
        chain.add_block(mined, check_pow=True)

        if (i + 1) % window == 0 and i + 1 < blocks:
            window_time = sum(stats.intervals[-window:])
            difficulty = retarget(difficulty, window_time, window * target_seconds)
            stats.difficulties.append(difficulty)

    return chain, stats
