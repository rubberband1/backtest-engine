"""Order sending, and the checks that stand between a spec and real money.

Everything in `core.data` is read-only by design; this is the one module that
can move an account. It is therefore also the module that refuses to.

Three refusals, in order of severity:

1. **Dry run is the default.** `LiveBroker(dry_run=True)` computes and logs
   the order it would send and sends nothing. Nothing in this codebase
   constructs a broker with `dry_run=False` on its own; a caller has to pass
   it, explicitly, every time.
2. **The account must be a demo.** Even with `dry_run=False`, a broker whose
   account reports a live trade mode refuses to start and says why. An
   equivalence test that has never been run against a demo has no business
   running against a funded account.
3. **Algo trading must be enabled in the terminal**, or every order comes
   back rejected and the runner would log a wall of retcodes instead of one
   sentence.

Beyond that, the job here is to turn MT5's answer into something the diary
can hold: the retcode by name, the volume actually filled - which is not
always the volume requested - and the price actually filled at, which is
what the expected-versus-realized comparison is built on.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from core.data.provider import SymbolSpec

logger = logging.getLogger(__name__)

try:  # Windows-only, like the rest of the MT5 surface
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

Side = Literal["buy", "sell"]

# Identifies this engine's positions among everything else in the account.
# Reconciliation is by magic number: a position without it was not opened
# here and must never be adopted.
DEFAULT_MAGIC = 770_115

# MT5 account trade modes. 0 is demo, 1 contest, 2 real.
TRADE_MODE_DEMO = 0
TRADE_MODE_CONTEST = 1
TRADE_MODE_REAL = 2

MAX_SEND_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0

_RETCODES: dict[int, str] = {}


class BrokerError(RuntimeError):
    """The broker cannot be used as asked."""


class LiveAccountRefused(BrokerError):
    """The account is not a demo and real sending was requested."""


def retcode_name(code: int) -> str:
    """MT5's numeric retcode as its constant name, for a readable diary."""
    if not _RETCODES and mt5 is not None:
        for attribute in dir(mt5):
            if attribute.startswith("TRADE_RETCODE_"):
                _RETCODES[getattr(mt5, attribute)] = attribute[len("TRADE_RETCODE_"):].lower()
    return _RETCODES.get(int(code), str(code))


@dataclass(frozen=True)
class AccountGuard:
    """What was checked before the runner was allowed to send anything."""

    trade_mode: int
    trade_mode_name: str
    algo_trading_enabled: bool
    dry_run: bool
    allowed_to_send: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderRequest:
    """What the runner wants done, before the broker is involved."""

    symbol: str
    side: Side
    lots: float
    stop_level: float | None
    target_level: float | None
    comment: str = ""
    magic: int = DEFAULT_MAGIC
    deviation_points: int = 20
    # Set when the order closes an existing position. Without it an opposite
    # deal opens a second position on a hedging account instead of closing the
    # first: the account type would silently decide whether "close" means close.
    closes_position: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderResult:
    """What actually happened. `filled_price` is the number that matters."""

    accepted: bool
    dry_run: bool
    retcode: int | None
    retcode_name: str
    requested_lots: float
    filled_lots: float
    requested_price: float | None
    filled_price: float | None
    order_ticket: int | None
    position_ticket: int | None
    comment: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def partial(self) -> bool:
        return self.accepted and self.filled_lots < self.requested_lots

    @property
    def slippage_price(self) -> float | None:
        if self.requested_price is None or self.filled_price is None:
            return None
        return self.filled_price - self.requested_price

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(partial=self.partial, slippage_price=self.slippage_price)
        return payload


@dataclass(frozen=True)
class OpenPosition:
    """A position the broker says exists."""

    ticket: int
    symbol: str
    direction: int
    lots: float
    entry_price: float
    stop_level: float | None
    target_level: float | None
    magic: int
    opened_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveBroker:
    """Order sending on a MetaTrader 5 terminal, guarded.

    `dry_run` defaults to True and has to be turned off by the caller. With
    it on, `send` returns a result marked `dry_run` and no order leaves the
    process.
    """

    def __init__(
        self,
        *,
        dry_run: bool = True,
        magic: int = DEFAULT_MAGIC,
        allow_contest_account: bool = True,
        max_attempts: int = MAX_SEND_ATTEMPTS,
    ) -> None:
        self.dry_run = dry_run
        self.magic = magic
        self.allow_contest_account = allow_contest_account
        self.max_attempts = max_attempts
        self._guard: AccountGuard | None = None

    # -- the guard -------------------------------------------------------

    def check_account(self) -> AccountGuard:
        """Decides whether this process may send orders. Call before trading."""
        if mt5 is None:
            raise BrokerError("MetaTrader5 package not installed (requires Windows)")
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None or terminal is None:
            raise BrokerError(
                "the terminal did not answer account_info/terminal_info: start "
                "MetaTrader 5 and log in before starting the runner"
            )

        mode = int(account.trade_mode)
        mode_name = {
            TRADE_MODE_DEMO: "demo",
            TRADE_MODE_CONTEST: "contest",
            TRADE_MODE_REAL: "real",
        }.get(mode, str(mode))
        algo = bool(getattr(terminal, "trade_allowed", False))

        acceptable = mode == TRADE_MODE_DEMO or (
            mode == TRADE_MODE_CONTEST and self.allow_contest_account
        )

        if self.dry_run:
            reason = (
                f"dry run: the account is {mode_name} and no order will be sent "
                f"whatever it is"
            )
            guard = AccountGuard(mode, mode_name, algo, True, False, reason)
        elif not acceptable:
            reason = (
                f"refusing to send orders: the account reports trade mode "
                f"{mode_name}, not demo. This runner has never been validated "
                f"against a funded account and the replay equivalence test says "
                f"nothing about one. Point the terminal at a demo account, or "
                f"leave dry_run on"
            )
            guard = AccountGuard(mode, mode_name, algo, False, False, reason)
            self._guard = guard
            logger.error(reason)
            raise LiveAccountRefused(reason)
        elif not algo:
            reason = (
                "refusing to start: Algo Trading is disabled in the terminal, so "
                "every order would come back rejected. Enable it and restart"
            )
            guard = AccountGuard(mode, mode_name, algo, False, False, reason)
            self._guard = guard
            logger.error(reason)
            raise BrokerError(reason)
        else:
            reason = f"sending live orders on a {mode_name} account"
            guard = AccountGuard(mode, mode_name, algo, False, True, reason)
            logger.warning(reason)

        self._guard = guard
        logger.info("account guard: %s", guard.reason)
        return guard

    @property
    def guard(self) -> AccountGuard | None:
        return self._guard

    # -- reconciliation --------------------------------------------------

    def open_positions(self, symbol: str) -> list[OpenPosition]:
        """Positions the broker holds on this symbol, whatever opened them."""
        if mt5 is None:
            raise BrokerError("MetaTrader5 package not installed")
        raw = mt5.positions_get(symbol=symbol)
        if raw is None:
            code, text = mt5.last_error()
            raise BrokerError(f"positions_get({symbol}) failed: ({code}) {text}")
        return [
            OpenPosition(
                ticket=int(item.ticket),
                symbol=str(item.symbol),
                direction=1 if int(item.type) == 0 else -1,
                lots=float(item.volume),
                entry_price=float(item.price_open),
                stop_level=float(item.sl) or None,
                target_level=float(item.tp) or None,
                magic=int(item.magic),
                opened_at=datetime.fromtimestamp(int(item.time), tz=timezone.utc),
            )
            for item in raw
        ]

    def own_positions(self, symbol: str) -> list[OpenPosition]:
        return [p for p in self.open_positions(symbol) if p.magic == self.magic]

    def foreign_positions(self, symbol: str) -> list[OpenPosition]:
        return [p for p in self.open_positions(symbol) if p.magic != self.magic]

    # -- sending ---------------------------------------------------------

    def _quote(self, symbol: str, side: Side) -> float:
        assert mt5 is not None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerError(f"no tick for {symbol}: cannot price an order")
        return float(tick.ask if side == "buy" else tick.bid)

    def send(self, request: OrderRequest, spec: SymbolSpec) -> OrderResult:
        """Sends a market order, or describes it and stops if `dry_run`."""
        if mt5 is None:
            raise BrokerError("MetaTrader5 package not installed")
        if self._guard is None:
            raise BrokerError(
                "check_account() has not been called: the runner must not send "
                "an order before the account has been checked"
            )

        price = self._quote(request.symbol, request.side)
        if self.dry_run or not self._guard.allowed_to_send:
            logger.info(
                "DRY RUN: would %s %.2f lots of %s at %.*f (sl=%s tp=%s)",
                request.side, request.lots, request.symbol, spec.digits, price,
                request.stop_level, request.target_level,
            )
            return OrderResult(
                accepted=False,
                dry_run=True,
                retcode=None,
                retcode_name="dry_run",
                requested_lots=request.lots,
                filled_lots=0.0,
                requested_price=price,
                filled_price=None,
                order_ticket=None,
                position_ticket=None,
                comment="dry run: nothing sent",
            )

        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": float(request.lots),
            "type": mt5.ORDER_TYPE_BUY if request.side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": int(request.deviation_points),
            "magic": int(request.magic),
            "comment": request.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if request.stop_level is not None:
            payload["sl"] = round(float(request.stop_level), spec.digits)
        if request.target_level is not None:
            payload["tp"] = round(float(request.target_level), spec.digits)
        if request.closes_position is not None:
            payload["position"] = int(request.closes_position)

        delay = BACKOFF_SECONDS
        last: OrderResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            payload["price"] = self._quote(request.symbol, request.side)
            answer = mt5.order_send(payload)
            last = self._interpret(answer, request, float(payload["price"]))
            if last.accepted:
                if last.partial:
                    logger.warning(
                        "partial fill: %.2f of %.2f lots on %s",
                        last.filled_lots, request.lots, request.symbol,
                    )
                return last
            if last.retcode not in self._retryable():
                logger.error(
                    "order rejected (%s): %s", last.retcode_name, last.comment
                )
                return last
            logger.warning(
                "order attempt %d/%d failed with %s, retrying",
                attempt, self.max_attempts, last.retcode_name,
            )
            if attempt < self.max_attempts:
                time.sleep(delay)
                delay *= 2
        assert last is not None
        return last

    @staticmethod
    def _retryable() -> set[int]:
        """Retcodes worth another attempt: the price moved, not the order is wrong."""
        if mt5 is None:
            return set()
        return {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_CONNECTION,
        }

    def _interpret(
        self, answer: Any, request: OrderRequest, requested_price: float
    ) -> OrderResult:
        assert mt5 is not None
        if answer is None:
            code, text = mt5.last_error()
            return OrderResult(
                accepted=False,
                dry_run=False,
                retcode=None,
                retcode_name="no_answer",
                requested_lots=request.lots,
                filled_lots=0.0,
                requested_price=requested_price,
                filled_price=None,
                order_ticket=None,
                position_ticket=None,
                comment=f"order_send returned nothing: ({code}) {text}",
            )

        code = int(answer.retcode)
        accepted = code in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL)
        filled = float(getattr(answer, "volume", 0.0) or 0.0)
        filled_price = float(getattr(answer, "price", 0.0) or 0.0) or None
        return OrderResult(
            accepted=accepted,
            dry_run=False,
            retcode=code,
            retcode_name=retcode_name(code),
            requested_lots=request.lots,
            filled_lots=filled if accepted else 0.0,
            requested_price=requested_price,
            filled_price=filled_price if accepted else None,
            order_ticket=int(getattr(answer, "order", 0) or 0) or None,
            position_ticket=int(getattr(answer, "deal", 0) or 0) or None,
            comment=str(getattr(answer, "comment", "") or ""),
        )

    def close(self, position: OpenPosition, spec: SymbolSpec) -> OrderResult:
        """Closes a position with an opposite market order."""
        side: Side = "sell" if position.direction > 0 else "buy"
        return self.send(
            OrderRequest(
                symbol=position.symbol,
                side=side,
                lots=position.lots,
                stop_level=None,
                target_level=None,
                comment="close",
                magic=self.magic,
                closes_position=position.ticket,
            ),
            spec,
        )
