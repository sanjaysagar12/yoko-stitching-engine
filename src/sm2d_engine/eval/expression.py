"""Seamly2D formula evaluator (muParser dialect): tokenizer + recursive descent, no ``eval``.

Dialect: ``c ? a : b``, ``&& ||``, comparisons, ``+ - * / ^`` (unary minus binds tighter than ``^``, as in
muParser), ``atan`` etc. in radians, ``sinD`` etc. in degrees, ``min max sum avg sqrt abs sign rint ...``.
Values are plain numbers in the caller's unit; the evaluator does no unit handling.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import lru_cache

from ..errors import FormulaError

Lookup = Callable[[str], float]
Node = Callable[[Lookup], float]

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<id>(?:#|[^\W\d])[\w#]*)"
    r"|(?P<op>&&|\|\||==|!=|<=|>=|[-+*/^<>?:(),!]))"
)


def _rint(x: float) -> float:
    return float(math.floor(x + 0.5))


def _sign(x: float) -> float:
    return float((x > 0) - (x < 0))


_D = math.radians
_FUNCS: dict[str, Callable[..., float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "sinD": lambda x: math.sin(_D(x)), "cosD": lambda x: math.cos(_D(x)), "tanD": lambda x: math.tan(_D(x)),
    "asinD": lambda x: math.degrees(math.asin(x)), "acosD": lambda x: math.degrees(math.acos(x)),
    "atanD": lambda x: math.degrees(math.atan(x)),
    "sqrt": math.sqrt, "abs": abs, "exp": math.exp, "log": math.log, "ln": math.log,
    "log10": math.log10, "log2": math.log2, "rint": _rint, "sign": _sign,
    "min": min, "max": max, "sum": lambda *a: float(sum(a)), "avg": lambda *a: sum(a) / len(a),
}  # fmt: skip
_CONSTS = {"_pi": math.pi, "_e": math.e}


class _Parser:
    def __init__(self, formula: str) -> None:
        self.f = formula
        self.toks: list[tuple[str, str]] = []
        pos = 0
        s = formula.rstrip()
        while pos < len(s):
            m = _TOKEN.match(s, pos)
            if not m or m.end() == pos:
                raise FormulaError(formula, f"unexpected character {s[pos:].lstrip()[:1]!r}")
            kind = m.lastgroup or ""
            self.toks.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def fail(self, why: str) -> FormulaError:
        return FormulaError(self.f, why)

    def peek(self) -> str:
        return self.toks[self.i][1] if self.i < len(self.toks) and self.toks[self.i][0] == "op" else ""

    def eat(self, op: str) -> bool:
        if self.peek() == op:
            self.i += 1
            return True
        return False

    def expect(self, op: str) -> None:
        if not self.eat(op):
            raise self.fail(f"expected {op!r}")

    def parse(self) -> Node:
        if not self.toks:
            raise self.fail("empty formula")
        n = self.ternary()
        if self.i != len(self.toks):
            raise self.fail(f"unexpected token {self.toks[self.i][1]!r}")
        return n

    def ternary(self) -> Node:
        c = self.or_()
        if self.eat("?"):
            a = self.ternary()
            self.expect(":")
            b = self.ternary()
            return lambda lk: a(lk) if c(lk) != 0 else b(lk)
        return c

    def or_(self) -> Node:
        n = self.and_()
        while self.eat("||"):
            n = self._bin(n, self.and_(), lambda x, y: float(x != 0 or y != 0))
        return n

    def and_(self) -> Node:
        n = self.cmp()
        while self.eat("&&"):
            n = self._bin(n, self.cmp(), lambda x, y: float(x != 0 and y != 0))
        return n

    def cmp(self) -> Node:
        ops: dict[str, Callable[[float, float], float]] = {
            "==": lambda x, y: float(x == y), "!=": lambda x, y: float(x != y),
            "<": lambda x, y: float(x < y), ">": lambda x, y: float(x > y),
            "<=": lambda x, y: float(x <= y), ">=": lambda x, y: float(x >= y),
        }  # fmt: skip
        n = self.add()
        while self.peek() in ops:
            fn = ops[self.peek()]
            self.i += 1
            n = self._bin(n, self.add(), fn)
        return n

    def add(self) -> Node:
        n = self.mul()
        while self.peek() in ("+", "-"):
            plus = self.peek() == "+"
            self.i += 1
            n = self._bin(n, self.mul(), (lambda x, y: x + y) if plus else (lambda x, y: x - y))
        return n

    def mul(self) -> Node:
        n = self.pow()
        while self.peek() in ("*", "/"):
            times = self.peek() == "*"
            self.i += 1
            n = self._bin(n, self.pow(), (lambda x, y: x * y) if times else self._div)
        return n

    def _div(self, x: float, y: float) -> float:
        if y == 0:
            raise self.fail("division by zero")
        return x / y

    def pow(self) -> Node:
        n = self.unary()
        while self.eat("^"):
            n = self._bin(n, self.unary(), self._pow)
        return n

    def _pow(self, x: float, y: float) -> float:
        try:
            r = x**y
        except (ZeroDivisionError, OverflowError) as e:
            raise self.fail(str(e)) from None
        if isinstance(r, complex):
            raise self.fail("complex result of power")
        return float(r)

    def unary(self) -> Node:
        if self.eat("-"):
            n = self.unary()
            return lambda lk: -n(lk)
        if self.eat("+"):
            return self.unary()
        if self.eat("!"):
            n = self.unary()
            return lambda lk: float(n(lk) == 0)
        return self.primary()

    def primary(self) -> Node:
        if self.i >= len(self.toks):
            raise self.fail("unexpected end of formula")
        kind, text = self.toks[self.i]
        if kind == "num":
            self.i += 1
            v = float(text)
            return lambda lk: v
        if kind == "id":
            self.i += 1
            if self.peek() == "(":
                return self.call(text)
            if text in _CONSTS:
                c = _CONSTS[text]
                return lambda lk: c
            return lambda lk: float(lk(text))
        if text == "(":
            self.i += 1
            n = self.ternary()
            self.expect(")")
            return n
        raise self.fail(f"unexpected token {text!r}")

    def call(self, name: str) -> Node:
        fn = _FUNCS.get(name)
        if fn is None:
            raise self.fail(f"unknown function {name!r}")
        self.expect("(")
        args = [self.ternary()]
        while self.eat(","):
            args.append(self.ternary())
        self.expect(")")

        def run(lk: Lookup) -> float:
            vals = [a(lk) for a in args]
            try:
                return float(fn(*vals))
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                raise FormulaError(self.f, f"{name}({', '.join(map(str, vals))}): {e}") from None

        return run

    @staticmethod
    def _bin(a: Node, b: Node, fn: Callable[[float, float], float]) -> Node:
        return lambda lk: fn(a(lk), b(lk))


@lru_cache(maxsize=4096)
def _compile(formula: str) -> Node:
    return _Parser(formula).parse()


def evaluate(formula: str, lookup: Lookup) -> float:
    """Evaluate ``formula``; ``lookup(name)`` supplies variable values (same unit as the caller's data)."""
    return _compile(formula.strip())(lookup)
