"""The ledger: transactions, account state, blocks and the block tree.

The block tree keeps EVERY valid block it has seen, not just the current best
chain. Fork choice picks the branch with the most cumulative work; losing
branches stay stored so that a later block can still make them win.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import cached_property

from .pow import target_from_difficulty

ZERO_HASH = "00" * 32


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Transactions and account state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transaction:
    sender: str
    recipient: str
    amount: int
    nonce: int
    fee: int = 0

    @property
    def txid(self) -> str:
        return sha256_hex(
            f"{self.sender}|{self.recipient}|{self.amount}|{self.nonce}|{self.fee}"
        )


class State:
    """Account balances plus the next expected nonce per account.

    The nonce does two jobs: a transaction can only be applied once (replay
    protection), and an account's transactions apply in a fixed order.
    """

    def __init__(
        self, balances: dict[str, int] | None = None, nonces: dict[str, int] | None = None
    ) -> None:
        self.balances: dict[str, int] = dict(balances or {})
        self.nonces: dict[str, int] = dict(nonces or {})

    def copy(self) -> State:
        return State(self.balances, self.nonces)

    def balance_of(self, account: str) -> int:
        return self.balances.get(account, 0)

    def nonce_of(self, account: str) -> int:
        return self.nonces.get(account, 0)

    def validate(self, tx: Transaction) -> str | None:
        """Return None if tx can be applied, otherwise the reason it can't."""
        if tx.amount < 0 or tx.fee < 0:
            return "negative amount or fee"
        expected = self.nonce_of(tx.sender)
        if tx.nonce != expected:
            return f"bad nonce (expected {expected}, got {tx.nonce})"
        if self.balance_of(tx.sender) < tx.amount + tx.fee:
            return "insufficient balance"
        return None

    def apply(self, tx: Transaction) -> None:
        error = self.validate(tx)
        if error is not None:
            raise ValueError(error)
        # the fee leaves the sender but does NOT go to the recipient
        self.balances[tx.sender] = self.balance_of(tx.sender) - tx.amount - tx.fee
        self.balances[tx.recipient] = self.balance_of(tx.recipient) + tx.amount
        self.nonces[tx.sender] = self.nonce_of(tx.sender) + 1

    def credit(self, account: str, amount: int) -> None:
        self.balances[account] = self.balance_of(account) + amount


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Block:
    height: int
    prev_hash: str
    timestamp: float
    txs: tuple[Transaction, ...] = ()
    difficulty: int = 1
    nonce: int = 0
    miner: str = ""

    @property
    def tx_root(self) -> str:
        # a simple commitment to the transaction list (not a real Merkle tree)
        return sha256_hex("".join(tx.txid for tx in self.txs))

    def header_prefix(self) -> bytes:
        """The header without the nonce. The miner hashes prefix + nonce."""
        return (
            f"{self.height}|{self.prev_hash}|{self.timestamp!r}|{self.tx_root}|"
            f"{self.difficulty}|{self.miner}|"
        ).encode()

    @cached_property
    def hash(self) -> str:
        return hashlib.sha256(self.header_prefix() + str(self.nonce).encode()).hexdigest()

    @property
    def target(self) -> int:
        return target_from_difficulty(self.difficulty)

    @property
    def work(self) -> int:
        # expected number of hashes is proportional to difficulty
        return self.difficulty

    def is_valid_pow(self) -> bool:
        return int(self.hash, 16) < self.target


def genesis_block() -> Block:
    return Block(height=0, prev_hash=ZERO_HASH, timestamp=0.0, miner="genesis")


# --------------------------------------------------------------------------- #
# Block tree with heaviest-chain fork choice
# --------------------------------------------------------------------------- #


@dataclass
class ChainEntry:
    block: Block
    total_work: int  # cumulative work from genesis up to and including this block


class Chain:
    def __init__(self, genesis: Block | None = None) -> None:
        g = genesis or genesis_block()
        self.genesis_hash = g.hash
        self.entries: dict[str, ChainEntry] = {g.hash: ChainEntry(g, g.work)}
        self.tip_hash = g.hash
        self.reorg_depths: list[int] = []

    # -- lookups ------------------------------------------------------------ #

    @property
    def tip(self) -> Block:
        return self.entries[self.tip_hash].block

    @property
    def height(self) -> int:
        return self.tip.height

    @property
    def total_work(self) -> int:
        return self.entries[self.tip_hash].total_work

    @property
    def reorg_count(self) -> int:
        return len(self.reorg_depths)

    def has(self, block_hash: str) -> bool:
        return block_hash in self.entries

    def get(self, block_hash: str) -> Block:
        return self.entries[block_hash].block

    def work_of(self, block_hash: str) -> int:
        return self.entries[block_hash].total_work

    # -- adding blocks ------------------------------------------------------ #

    def add_block(self, block: Block, check_pow: bool = True) -> bool:
        """Store a block. Returns True if it became the new tip.

        Raises KeyError for an unknown parent and ValueError for a block that
        is malformed or has invalid proof of work.
        """
        if block.hash in self.entries:
            return False
        parent = self.entries.get(block.prev_hash)
        if parent is None:
            raise KeyError(f"unknown parent {block.prev_hash[:12]}")
        if block.height != parent.block.height + 1:
            raise ValueError("height does not follow the parent")
        if check_pow and not block.is_valid_pow():
            raise ValueError("invalid proof of work")

        entry = ChainEntry(block, parent.total_work + block.work)
        self.entries[block.hash] = entry  # stored even if it loses

        # strictly more work needed: on a tie the block we saw first stays
        if entry.total_work <= self.total_work:
            return False

        old_tip = self.tip_hash
        if block.prev_hash != old_tip:
            # switching branches: count how many of our blocks get rolled back
            fork_point = self.common_ancestor(old_tip, block.hash)
            depth = self.get(old_tip).height - self.get(fork_point).height
            self.reorg_depths.append(depth)
        self.tip_hash = block.hash
        return True

    # -- walking the tree --------------------------------------------------- #

    def ancestors(self, block_hash: str) -> list[str]:
        """Hashes from block_hash back to genesis (inclusive)."""
        out = []
        current = block_hash
        while True:
            out.append(current)
            if current == self.genesis_hash:
                return out
            current = self.get(current).prev_hash

    def common_ancestor(self, a: str, b: str) -> str:
        seen = set(self.ancestors(a))
        for h in self.ancestors(b):
            if h in seen:
                return h
        return self.genesis_hash

    def path(self, tip_hash: str | None = None) -> list[Block]:
        """Blocks from genesis up to tip_hash (default: the current tip)."""
        hashes = self.ancestors(tip_hash or self.tip_hash)
        return [self.get(h) for h in reversed(hashes)]

    def canonical(self) -> list[Block]:
        return self.path()

    def replay_state(
        self,
        initial_balances: dict[str, int],
        reward: int = 0,
        tip_hash: str | None = None,
    ) -> State:
        """Rebuild the account state by applying every block from genesis."""
        state = State(initial_balances)
        for block in self.path(tip_hash)[1:]:
            fees = 0
            for tx in block.txs:
                state.apply(tx)
                fees += tx.fee
            if block.miner and (reward or fees):
                state.credit(block.miner, reward + fees)
        return state


def make_block(
    chain: Chain,
    txs: tuple[Transaction, ...] | list[Transaction] = (),
    miner: str = "",
    timestamp: float | None = None,
    difficulty: int = 1,
) -> Block:
    """An unmined block on top of the chain's current tip."""
    return Block(
        height=chain.height + 1,
        prev_hash=chain.tip_hash,
        timestamp=time.time() if timestamp is None else float(timestamp),
        txs=tuple(txs),
        difficulty=difficulty,
        miner=miner,
    )
