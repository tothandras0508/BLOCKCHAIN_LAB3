"""Proof of stake: Casper FFG justification/finality, slashing, inactivity leak.

All balances are in gwei (1 ETH = 10**9 gwei), like on Ethereum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

GWEI = 10**9
MAX_EFFECTIVE_BALANCE = 32 * GWEI

#: On slashing, 1/32 of the effective balance is taken straight away.
INITIAL_SLASHING_QUOTIENT = 32
#: The correlated penalty uses 3 x (slashed stake / total stake).
PROPORTIONAL_SLASHING_MULTIPLIER = 3
#: Epochs without finality before the inactivity leak starts.
MIN_EPOCHS_TO_INACTIVITY_PENALTY = 4
#: Each leaking epoch removes 1/64 of an absent validator's effective balance.
INACTIVITY_PENALTY_QUOTIENT = 64


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    root: str

    def __repr__(self) -> str:
        return f"({self.epoch}, {self.root})"


@dataclass(frozen=True)
class Attestation:
    """A vote for the link source -> target."""

    validator: int
    source: Checkpoint
    target: Checkpoint


@dataclass
class Validator:
    index: int
    balance: int = MAX_EFFECTIVE_BALANCE
    slashed: bool = False

    @property
    def effective_balance(self) -> int:
        return min(self.balance, MAX_EFFECTIVE_BALANCE)

    @property
    def is_active(self) -> bool:
        return not self.slashed and self.balance > 0


@dataclass
class SlashingRecord:
    validator: int
    epoch: int
    rule: str
    initial_penalty: int
    correlated_penalty: int = 0

    @property
    def total(self) -> int:
        return self.initial_penalty + self.correlated_penalty


# --------------------------------------------------------------------------- #
# Slashing conditions
# --------------------------------------------------------------------------- #


def is_double_vote(a: Attestation, b: Attestation) -> bool:
    """Same validator, same target epoch, different target."""
    return (
        a.validator == b.validator
        and a.target.epoch == b.target.epoch
        and a.target.root != b.target.root
    )


def is_surround_vote(a: Attestation, b: Attestation) -> bool:
    """Vote a's source..target span strictly surrounds vote b's."""
    return (
        a.validator == b.validator
        and a.source.epoch < b.source.epoch
        and b.target.epoch < a.target.epoch
    )


def find_violations(
    attestations: list[Attestation],
) -> list[tuple[Attestation, Attestation, str]]:
    found = []
    for a, b in combinations(attestations, 2):
        if a.validator != b.validator:
            continue
        if is_double_vote(a, b):
            found.append((a, b, "double_vote"))
        elif is_surround_vote(a, b) or is_surround_vote(b, a):
            found.append((a, b, "surround_vote"))
    return found


# --------------------------------------------------------------------------- #
# Beacon state
# --------------------------------------------------------------------------- #


class BeaconState:
    def __init__(self, validator_count: int, genesis_root: str = "genesis") -> None:
        self.validators = [Validator(i) for i in range(validator_count)]
        self.genesis_checkpoint = Checkpoint(0, genesis_root)
        self.justified: set[Checkpoint] = {self.genesis_checkpoint}
        self.current_justified = self.genesis_checkpoint
        self.finalized = self.genesis_checkpoint
        self.last_finalized_epoch = 0
        self.current_epoch = 0
        self.attestations: list[Attestation] = []
        self.slashings: list[SlashingRecord] = []
        self.slashed_balance_total = 0

    # -- stake -------------------------------------------------------------- #

    @property
    def total_active_balance(self) -> int:
        return sum(v.effective_balance for v in self.validators if v.is_active)

    def supermajority_threshold(self) -> int:
        """Votes must be STRICTLY greater than this (two thirds of active stake)."""
        return 2 * self.total_active_balance // 3

    # -- voting ------------------------------------------------------------- #

    def attest(self, attestation: Attestation) -> None:
        if self.validators[attestation.validator].is_active:
            self.attestations.append(attestation)

    def attest_many(self, attestations: list[Attestation]) -> None:
        for a in attestations:
            self.attest(a)

    # -- epoch processing --------------------------------------------------- #

    def process_epoch(self, epoch: int) -> dict[str, object]:
        """Slashing first (so cheaters don't count), then justification,
        finalisation and finally the inactivity leak."""
        self.current_epoch = epoch
        slashed = self._slash(epoch)
        justified = self._justify(epoch)
        finalized = self._finalize(epoch)
        leaked = self._leak(epoch)
        return {
            "epoch": epoch,
            "slashed": slashed,
            "justified": justified,
            "finalized": finalized,
            "inactivity_leak": leaked,
        }

    def _justify(self, epoch: int) -> Checkpoint | None:
        weight: dict[tuple[Checkpoint, Checkpoint], int] = {}
        for a in self.attestations:
            if a.target.epoch != epoch:
                continue
            if a.source not in self.justified:
                continue  # links from unjustified sources don't count
            v = self.validators[a.validator]
            if not v.is_active:
                continue
            link = (a.source, a.target)
            weight[link] = weight.get(link, 0) + v.effective_balance

        threshold = self.supermajority_threshold()
        for (_source, target), w in weight.items():
            if w > threshold:
                self.justified.add(target)
                if target.epoch > self.current_justified.epoch:
                    self.current_justified = target
                return target
        return None

    def _finalize(self, epoch: int) -> Checkpoint | None:
        justified_epochs = {c.epoch for c in self.justified}
        for c in sorted(self.justified, key=lambda c: c.epoch):
            if c.epoch <= self.finalized.epoch:
                continue
            if c.epoch + 1 in justified_epochs:  # its direct successor is justified
                self.finalized = c
                self.last_finalized_epoch = epoch
                return c
        return None

    def _slash(self, epoch: int) -> list[SlashingRecord]:
        already = {r.validator for r in self.slashings}
        new_records = []
        for a, _b, rule in find_violations(self.attestations):
            if a.validator in already:
                continue  # a validator is only slashed once
            v = self.validators[a.validator]
            penalty = v.effective_balance // INITIAL_SLASHING_QUOTIENT
            v.balance -= penalty
            v.slashed = True
            self.slashed_balance_total += v.effective_balance
            record = SlashingRecord(a.validator, epoch, rule, penalty)
            self.slashings.append(record)
            new_records.append(record)
            already.add(a.validator)

        if new_records:
            self._correlated_penalty(new_records)
        return new_records

    def _correlated_penalty(self, records: list[SlashingRecord]) -> None:
        """penalty = balance * min(1, 3 * slashed / total).

        One offender out of 64 loses a little; a third of the stake together
        loses everything.
        """
        total = self.total_active_balance + self.slashed_balance_total
        if total == 0:
            return
        fraction = min(
            1.0, PROPORTIONAL_SLASHING_MULTIPLIER * self.slashed_balance_total / total
        )
        for r in records:
            v = self.validators[r.validator]
            extra = min(int(v.effective_balance * fraction), v.balance)
            v.balance -= extra
            r.correlated_penalty = extra

    def _leak(self, epoch: int) -> int:
        """Drain validators that didn't vote this epoch, once finality has stalled."""
        if epoch - self.last_finalized_epoch <= MIN_EPOCHS_TO_INACTIVITY_PENALTY:
            return 0
        voted = {a.validator for a in self.attestations if a.target.epoch == epoch}
        leaked = 0
        for v in self.validators:
            if not v.is_active or v.index in voted:
                continue
            penalty = min(v.effective_balance // INACTIVITY_PENALTY_QUOTIENT, v.balance)
            v.balance -= penalty
            leaked += penalty
        return leaked

    # -- output ------------------------------------------------------------- #

    def summary(self) -> str:
        active = sum(1 for v in self.validators if v.is_active)
        slashed = sum(1 for v in self.validators if v.slashed)
        lines = [
            f"epoch              : {self.current_epoch}",
            f"validators         : {len(self.validators)} ({active} active, {slashed} slashed)",
            f"total active stake : {self.total_active_balance / GWEI:.2f} ETH",
            f"2/3 threshold      : {self.supermajority_threshold() / GWEI:.2f} ETH",
            f"current justified  : {self.current_justified}",
            f"finalized          : {self.finalized}",
        ]
        for r in self.slashings:
            lines.append(
                f"  slashed {r.validator:>3} ({r.rule}, epoch {r.epoch}): "
                f"-{r.total / GWEI:.4f} ETH (initial {r.initial_penalty / GWEI:.4f} "
                f"+ correlated {r.correlated_penalty / GWEI:.4f})"
            )
        return "\n".join(lines)


def honest_attestations(
    state: BeaconState,
    source: Checkpoint,
    target: Checkpoint,
    validators: list[int] | None = None,
) -> list[Attestation]:
    """Everyone listed (default: every active validator) votes source -> target."""
    if validators is None:
        validators = [v.index for v in state.validators if v.is_active]
    return [Attestation(i, source, target) for i in validators]
