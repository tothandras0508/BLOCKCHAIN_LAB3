"""Command line demos: `python -m consensus [pow|pos|gossip|all]`."""

from __future__ import annotations

import argparse
import asyncio

from .chain import Transaction
from .network import SimulationResult, build_network
from .pos import GWEI, Attestation, BeaconState, Checkpoint, honest_attestations
from .pow import retarget, simulate_mining

LINE = "=" * 78


def heading(title: str) -> None:
    print(LINE)
    print(title)
    print(LINE)


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #


def demo_pow(blocks: int = 30, workers: int | None = None) -> None:
    heading("PART A -- Proof of Work")
    print("Retarget (old=1000, expected window=600 s):")
    for actual in (300, 1200, 6, 60000):
        print(f"  window took {actual:>6} s -> {retarget(1000, actual, 600)}")
    print()

    print(f"Mining {blocks} blocks (retarget every 10, target 0.30 s/block)...")
    chain, stats = simulate_mining(blocks=blocks, workers=workers)
    print(stats.report(target_seconds=0.30))
    print(f"chain height       : {chain.height}")
    print(f"cumulative work    : {chain.total_work}")
    print()


# --------------------------------------------------------------------------- #
# Part B
# --------------------------------------------------------------------------- #


def demo_pos() -> None:
    heading("PART B -- Proof of Stake: Casper FFG and slashing")

    print(">>> Scenario 1: finality")
    state = BeaconState(validator_count=64)
    cps = [state.genesis_checkpoint] + [Checkpoint(e, f"root-{e}") for e in range(1, 4)]
    for epoch in range(1, 4):
        state.attest_many(honest_attestations(state, cps[epoch - 1], cps[epoch]))
        r = state.process_epoch(epoch)
        print(f"  epoch {epoch}: justified={r['justified']}  finalized={r['finalized']}")
    print()

    src = Checkpoint(0, "genesis")
    tgt_a, tgt_b = Checkpoint(1, "chain-A"), Checkpoint(1, "chain-B")

    print(">>> Scenario 2: one validator double votes")
    state = BeaconState(validator_count=64)
    state.attest_many(honest_attestations(state, src, tgt_a))
    state.attest(Attestation(63, src, tgt_b))
    before = state.validators[63].balance
    state.process_epoch(1)
    after = state.validators[63].balance
    print(f"  validator 63 : {before / GWEI:.4f} -> {after / GWEI:.4f} ETH")
    print(f"  slashed      : {state.validators[63].slashed}")
    print(f"  loss         : {(before - after) / GWEI:.4f} ETH")
    print()

    print(">>> Scenario 3: 20 validators double vote together")
    state = BeaconState(validator_count=64)
    state.attest_many(honest_attestations(state, src, tgt_a))
    for i in range(20):
        state.attest(Attestation(i, src, tgt_b))
    before = state.validators[0].balance
    state.process_epoch(1)
    after = state.validators[0].balance
    print(f"  each balance : {before / GWEI:.4f} -> {after / GWEI:.4f} ETH")
    print(f"  loss each    : {(before - after) / GWEI:.4f} ETH")
    print()

    print(">>> Scenario 4: surround vote")
    state = BeaconState(validator_count=32)
    state.attest(Attestation(5, Checkpoint(0, "g"), Checkpoint(6, "far")))
    state.attest(Attestation(5, Checkpoint(2, "b"), Checkpoint(4, "near")))
    state.process_epoch(6)
    print(f"  slashed      : {state.validators[5].slashed}")
    print(f"  rule         : {state.slashings[0].rule}")
    print()

    print(">>> Scenario 5: inactivity leak (24 of 64 online)")
    state = BeaconState(validator_count=64)
    present = list(range(40, 64))
    for epoch in range(1, 12):
        source = Checkpoint(epoch - 1, f"root-{epoch - 1}") if epoch > 1 else state.genesis_checkpoint
        state.attest_many(
            honest_attestations(state, source, Checkpoint(epoch, f"root-{epoch}"), present)
        )
        r = state.process_epoch(epoch)
        if epoch in (1, 5, 8, 11):
            share = sum(state.validators[i].effective_balance for i in present) / state.total_active_balance
            print(
                f"  epoch {epoch:>2}: finalized={str(r['finalized']):<6} "
                f"leaked={r['inactivity_leak'] / GWEI:>7.3f} ETH  voting share={share:.1%}"
            )
    print()
    print(state.summary())
    print()


# --------------------------------------------------------------------------- #
# Part C
# --------------------------------------------------------------------------- #


async def _run_gossip(node_count: int, mean_block_time: float, seed: int) -> SimulationResult:
    balances = {f"user{i}": 1000 for i in range(4)}
    network, nodes = build_network(node_count, latency=0.005, seed=seed, initial_balances=balances)
    result = SimulationResult(nodes=nodes, network=network)
    names = list(nodes)

    tasks = [asyncio.create_task(n.run()) for n in nodes.values()]
    # spread the miners out so the partition separates them
    miners = list(nodes.values())[:: max(1, node_count // 3)]
    for m in miners:
        m.running = True
        tasks.append(asyncio.create_task(m.mine_forever(mean_block_time)))

    async def announcer() -> None:
        while True:
            await asyncio.sleep(0.15)
            for n in nodes.values():
                await n.announce_head()

    tasks.append(asyncio.create_task(announcer()))

    # phase 1: connected
    await asyncio.sleep(0.4)
    for i in range(4):
        await nodes[names[i % node_count]].submit_transaction(
            Transaction("user0", "user1", 10, nonce=i, fee=1)
        )
    await asyncio.sleep(1.2)
    result.events.append(
        f"phase 1 (connected)  heights={sorted(set(result.heights().values()))} "
        f"converged={result.converged()}"
    )

    # phase 2: partition into two halves
    a, b = set(names[: node_count // 2]), set(names[node_count // 2 :])
    network.partition([a, b])
    result.events.append(f"PARTITION  A={sorted(a)}  B={sorted(b)}")
    await asyncio.sleep(1.5)
    result.events.append(
        f"phase 2 (split)      distinct tips={len({n.tip_hash for n in nodes.values()})} "
        f"heights={sorted(set(result.heights().values()))}"
    )

    # phase 3: heal; an equal-work tie is only broken by the next block
    network.heal()
    result.events.append("HEAL")
    await asyncio.sleep(1.0)
    result.events.append(
        f"  just after heal    distinct tips={len({n.tip_hash for n in nodes.values()})}"
    )
    await asyncio.sleep(2.0)
    # stop mining and let the last blocks finish propagating before measuring,
    # otherwise a block that is still in flight looks like a disagreement
    for n in nodes.values():
        n.stop()
    await asyncio.sleep(0.5)
    result.events.append(
        f"phase 3 (healed)     distinct tips={len({n.tip_hash for n in nodes.values()})} "
        f"converged={result.converged()}"
    )

    for n in nodes.values():
        n.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return result


def demo_gossip(node_count: int = 8, seed: int = 7) -> None:
    heading("PART C -- gossip, partition and reorganisation")
    result = asyncio.run(_run_gossip(node_count, mean_block_time=0.25, seed=seed))
    for e in result.events:
        print(f"  {e}")
    print()
    print(result.report())
    print()
    print(f"reorganisations across all nodes: {sum(n.stats.reorgs for n in result.nodes.values())}")
    print("reorg depths:")
    for name, n in result.nodes.items():
        print(f"  {name:<7} {n.chain.reorg_depths}")
    print()


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="consensus")
    parser.add_argument("part", nargs="?", default="all", choices=["all", "pow", "pos", "gossip"])
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.part in ("all", "pow"):
        demo_pow(args.blocks, args.workers)
    if args.part in ("all", "pos"):
        demo_pos()
    if args.part in ("all", "gossip"):
        demo_gossip(args.nodes, args.seed)
    return 0
