# Basic calculator: functions + simple CLI and REPL
import argparse
import sys


def add(a: float, b: float) -> float:
    return a + b


def sub(a: float, b: float) -> float:
    return a - b


def mul(a: float, b: float) -> float:
    return a * b


def div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def power(a: float, b: float) -> float:
    return a**b


def run_operation(op: str, a: float, b: float) -> float:
    ops = {
        "add": add,
        "sub": sub,
        "mul": mul,
        "div": div,
        "pow": power,
        "+": add,
        "-": sub,
        "*": mul,
        "/": div,
        "**": power,
    }
    func = ops.get(op)
    if func is None:
        raise ValueError(f"unknown operation: {op}")
    return func(a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic calculator")
    parser.add_argument(
        "operation", nargs="?", help="Operation (add, sub, mul, div, pow)"
    )
    parser.add_argument("a", nargs="?", type=float, help="First number")
    parser.add_argument("b", nargs="?", type=float, help="Second number")
    return parser.parse_args()


def repl() -> None:
    print(
        'Basic calculator REPL. Enter `op a b` (e.g. "add 1 2"). Type "quit" to exit.'
    )
    while True:
        try:
            line = input("calc> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break
        parts = line.split()
        if len(parts) < 3:
            print("Usage: <op> <a> <b>")
            continue
        op, a_str, b_str = parts[0], parts[1], parts[2]
        try:
            a = float(a_str)
            b = float(b_str)
            res = run_operation(op, a, b)
        except (ValueError, ZeroDivisionError) as e:
            print("Error:", e)
        else:
            print(res)


if __name__ == "__main__":
    args = parse_args()
    if args.operation is None:
        repl()
    else:
        if args.a is None or args.b is None:
            print("Please provide two numbers", file=sys.stderr)
            sys.exit(2)
        try:
            result = run_operation(args.operation, args.a, args.b)
        except (ValueError, ZeroDivisionError) as e:
            print("Error:", e, file=sys.stderr)
            sys.exit(1)
        else:
            print(result)
