"""Gossip network: asyncio nodes that flood transactions and blocks.

Transport is in-process: every message is delivered into the receiver's
asyncio.Queue after an artificial latency. A partition simply drops any
message whose sender and receiver are in different groups.

Mining here is NOT real hashing (that's pow.py): a miner waits an exponential
delay and then makes a block, which keeps block arrivals a Poisson process.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from .chain import Block, Chain, State, Transaction, make_block

BLOCK_REWARD = 50
MAX_TXS_PER_BLOCK = 50


@dataclass
class Message:
    kind: str  # "tx", "block", "head", "get_chain", "chain"
    payload: object
    sender: str


# --------------------------------------------------------------------------- #
# Network (transport + partitions)
# --------------------------------------------------------------------------- #


class Network:
    def __init__(self, latency: float = 0.005, seed: int = 0) -> None:
        self.latency = latency
        self.rng = random.Random(seed)
        self.nodes: dict[str, Node] = {}
        self.groups: list[set[str]] | None = None  # None = fully connected
        self.delivered = 0
        self.dropped = 0

    def register(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def partition(self, groups: list[set[str]]) -> None:
        self.groups = [set(g) for g in groups]

    def heal(self) -> None:
        self.groups = None

    def can_reach(self, a: str, b: str) -> bool:
        if self.groups is None:
            return True
        return any(a in g and b in g for g in self.groups)

    def send(self, src: str, dst: str, msg: Message) -> None:
        if not self.can_reach(src, dst):
            self.dropped += 1
            return
        loop = asyncio.get_running_loop()
        loop.call_later(self.latency, self._deliver, src, dst, msg)

    def _deliver(self, src: str, dst: str, msg: Message) -> None:
        # check again: the partition may have started while it was in flight
        if not self.can_reach(src, dst):
            self.dropped += 1
            return
        self.delivered += 1
        self.nodes[dst].inbox.put_nowait(msg)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


@dataclass
class NodeStats:
    blocks_mined: int = 0
    received: int = 0
    duplicates_suppressed: int = 0
    invalid: int = 0
    reorgs: int = 0


class Node:
    def __init__(
        self,
        node_id: str,
        network: Network,
        initial_balances: dict[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.node_id = node_id
        self.network = network
        self.initial_balances = dict(initial_balances or {})
        self.peers: list[str] = []
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.chain = Chain()
        self.mempool: dict[str, Transaction] = {}
        self.known_txs: dict[str, Transaction] = {}  # every valid tx ever seen
        self.seen: set[str] = set()  # dedup filter for gossip
        self.orphans: dict[str, list[Block]] = {}  # parent hash -> waiting blocks
        self.stats = NodeStats()
        self.running = False
        self.rng = random.Random(seed)
        network.register(self)

    # -- convenience -------------------------------------------------------- #

    @property
    def tip_hash(self) -> str:
        return self.chain.tip_hash

    @property
    def height(self) -> int:
        return self.chain.height

    def stop(self) -> None:
        self.running = False

    def state(self) -> State:
        return self.chain.replay_state(self.initial_balances, BLOCK_REWARD)

    def pending_state(self) -> State:
        """Chain state with the mempool applied on top."""
        state = self.state()
        for tx in self.mempool.values():
            state.apply(tx)
        return state

    # -- sending ------------------------------------------------------------ #

    async def broadcast(self, msg: Message, exclude: str | None = None) -> None:
        if msg.kind == "block":
            self.seen.add("block:" + msg.payload.hash)
        for peer in self.peers:
            if peer != exclude:
                self.network.send(self.node_id, peer, Message(msg.kind, msg.payload, self.node_id))

    async def submit_transaction(self, tx: Transaction) -> bool:
        """A user hands a transaction to this node. Invalid ones are not relayed."""
        self.seen.add("tx:" + tx.txid)
        if not self._accept_tx(tx):
            return False
        await self.broadcast(Message("tx", tx, self.node_id))
        return True

    async def announce_head(self) -> None:
        """Tell peers our tip and its work; used to reconcile after a heal."""
        info = (self.chain.tip_hash, self.chain.total_work)
        for peer in self.peers:
            self.network.send(self.node_id, peer, Message("head", info, self.node_id))

    # -- main loops --------------------------------------------------------- #

    async def run(self) -> None:
        while True:
            msg = await self.inbox.get()
            self.stats.received += 1
            await self.handle(msg)

    async def mine_forever(self, mean_block_time: float) -> None:
        while self.running:
            await asyncio.sleep(self.rng.expovariate(1 / mean_block_time))
            if not self.running:
                return
            txs = list(self.mempool.values())[:MAX_TXS_PER_BLOCK]
            block = make_block(
                self.chain, txs=txs, miner=self.node_id,
                timestamp=asyncio.get_running_loop().time(),
            )
            self._add_block(block)
            self.stats.blocks_mined += 1
            await self.broadcast(Message("block", block, self.node_id))

    # -- message handling --------------------------------------------------- #

    async def handle(self, msg: Message) -> None:
        if msg.kind == "tx":
            await self._on_tx(msg)
        elif msg.kind == "block":
            await self._on_block(msg)
        elif msg.kind == "head":
            tip_hash, work = msg.payload
            # only ask for a branch that would beat ours
            if not self.chain.has(tip_hash) and work > self.chain.total_work:
                self.network.send(self.node_id, msg.sender, Message("get_chain", None, self.node_id))
        elif msg.kind == "get_chain":
            self.network.send(
                self.node_id, msg.sender, Message("chain", self.chain.canonical(), self.node_id)
            )
        elif msg.kind == "chain":
            for block in msg.payload:
                if not self.chain.has(block.hash):
                    self._add_block(block)

    async def _on_tx(self, msg: Message) -> None:
        tx: Transaction = msg.payload
        key = "tx:" + tx.txid
        if key in self.seen:
            self.stats.duplicates_suppressed += 1
            return
        self.seen.add(key)
        if self._accept_tx(tx):
            await self.broadcast(msg, exclude=msg.sender)

    async def _on_block(self, msg: Message) -> None:
        block: Block = msg.payload
        key = "block:" + block.hash
        if key in self.seen or self.chain.has(block.hash):
            self.stats.duplicates_suppressed += 1
            return
        self.seen.add(key)
        if not self.chain.has(block.prev_hash):
            # parent missing: park it and ask the sender for its chain
            self.orphans.setdefault(block.prev_hash, []).append(block)
            self.network.send(self.node_id, msg.sender, Message("get_chain", None, self.node_id))
        elif not self._add_block(block):
            return
        await self.broadcast(msg, exclude=msg.sender)

    # -- state changes ------------------------------------------------------ #

    def _accept_tx(self, tx: Transaction) -> bool:
        if tx.txid in self.mempool:
            return False
        if self.pending_state().validate(tx) is not None:
            self.stats.invalid += 1
            return False
        self.mempool[tx.txid] = tx
        self.known_txs[tx.txid] = tx
        return True

    def _add_block(self, block: Block) -> bool:
        """Validate and store a block (plus any orphans waiting for it)."""
        try:
            # every transaction must apply on top of the block's parent
            parent_state = self.chain.replay_state(
                self.initial_balances, BLOCK_REWARD, block.prev_hash
            )
            for tx in block.txs:
                parent_state.apply(tx)
            moved = self.chain.add_block(block, check_pow=False)
        except (KeyError, ValueError):
            self.stats.invalid += 1
            return False

        for tx in block.txs:
            self.known_txs[tx.txid] = tx
        self.seen.add("block:" + block.hash)

        for child in self.orphans.pop(block.hash, []):
            moved = self._add_block(child) or moved

        if moved:
            self.stats.reorgs = self.chain.reorg_count
            self._rebuild_mempool()
        return True

    def _rebuild_mempool(self) -> None:
        """After the tip moves, the mempool = known txs not in the canonical chain
        that are still valid. Txs from an orphaned branch come back this way."""
        included = {tx.txid for b in self.chain.canonical() for tx in b.txs}
        state = self.state()
        pending = sorted(
            (tx for txid, tx in self.known_txs.items() if txid not in included),
            key=lambda tx: (tx.sender, tx.nonce),
        )
        self.mempool = {}
        for tx in pending:
            if state.validate(tx) is None:
                state.apply(tx)
                self.mempool[tx.txid] = tx


# --------------------------------------------------------------------------- #
# Building a network
# --------------------------------------------------------------------------- #


def build_network(
    node_count: int = 8,
    latency: float = 0.005,
    seed: int = 0,
    initial_balances: dict[str, int] | None = None,
) -> tuple[Network, dict[str, Node]]:
    """A ring (so it's always connected) plus some random extra edges."""
    network = Network(latency=latency, seed=seed)
    nodes = {
        f"node{i}": Node(f"node{i}", network, initial_balances, seed=seed * 1000 + i)
        for i in range(node_count)
    }
    names = list(nodes)
    edges: set[tuple[str, str]] = set()
    for i in range(node_count):
        a, b = names[i], names[(i + 1) % node_count]
        if a != b:
            edges.add(tuple(sorted((a, b))))
    for _ in range(node_count // 2):
        a, b = network.rng.sample(names, 2) if node_count > 1 else (names[0], names[0])
        if a != b:
            edges.add(tuple(sorted((a, b))))
    for a, b in sorted(edges):
        nodes[a].peers.append(b)
        nodes[b].peers.append(a)
    return network, nodes


@dataclass
class SimulationResult:
    nodes: dict[str, Node]
    network: Network
    events: list[str] = field(default_factory=list)

    def heights(self) -> dict[str, int]:
        return {name: n.height for name, n in self.nodes.items()}

    def converged(self) -> bool:
        return len({n.tip_hash for n in self.nodes.values()}) == 1

    def report(self) -> str:
        lines = [
            f"{'node':<8}{'height':>6}  {'tip':<14}{'mined':>5} {'recv':>5} {'txs':>4} {'reorgs':>6}",
            "-" * 52,
        ]
        for name, n in self.nodes.items():
            lines.append(
                f"{name:<8}{n.height:>6}  {n.tip_hash[:12]:<14}{n.stats.blocks_mined:>5} "
                f"{n.stats.received:>5} {len(n.mempool):>4} {n.stats.reorgs:>6}"
            )
        lines += [
            "",
            f"messages delivered : {self.network.delivered}",
            f"messages dropped   : {self.network.dropped}  (partitioned away)",
            f"converged          : {self.converged()}",
        ]
        return "\n".join(lines)
