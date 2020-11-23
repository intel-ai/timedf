#!/usr/bin/env python3
from subprocess import Popen, PIPE
import tempfile
import pandas as pd


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-ht", default=False, action="store_true", help="do not use Hyper Threading"
    )
    parser.add_argument(
        "--n0-only", default=False, action="store_true", help="operate within single node"
    )
    parser.add_argument(
        "--each", default=False, action="store_true", help="print each number without ranges"
    )
    parser.add_argument(
        "--total", default=False, action="store_true", help="print total number of selected cpus"
    )
    parser.add_argument("--sep", "-s", default=",", help="output separator")
    parser.add_argument(
        "threads",
        default="all",
        help='number of threads to list. Accepts "all", "half", "/{number}", "{number}" ',
    )
    args = parser.parse_args()

    with tempfile.SpooledTemporaryFile(mode="w+") as tmp, Popen(
        "lscpu -p", shell=True, stdout=PIPE
    ) as pipe:
        for line in iter(pipe.stdout.readline, b""):
            decoded_line = line.decode()
            if decoded_line.startswith("#") and not decoded_line.startswith("# CPU"):
                continue
            tmp.write(decoded_line)
        tmp.seek(0)
        df = pd.read_csv(tmp)

    if args.no_ht:
        df = df.groupby("Core").min()

    nd0 = df[df["Node"] == 0]["# CPU"].values
    if args.n0_only:
        cpus = list(nd0)
    else:
        nd1 = df[df["Node"] == 1]["# CPU"].values
        cpus = [v for p in zip(nd0, nd1) for v in p]

    n = len(cpus)
    if args.threads == "all":
        pass
    elif args.threads == "half":
        n = int(n / 2)
    elif args.threads.startswith("/"):
        n = int(n / int(args.threads[1:]))
    else:
        n = min(int(args.threads), n)
    cpus = cpus[0:n]

    if args.total:
        print(len(cpus))
    elif args.each:
        print(*cpus, sep=args.sep)
    else:  # ranges
        cpus.sort()
        dash = False
        print(cpus[0], end="")
        for i in range(1, len(cpus) - 1):
            if cpus[i - 1] == cpus[i] - 1:  # print range
                dash = True
                continue
            if dash:
                print("-", cpus[i - 1], end="", sep="")
            dash = False
            print(args.sep, cpus[i], end="", sep="")
        if len(cpus) > 1:
            print("-" if dash else args.sep, cpus[-1], sep="")
        else:
            print("")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
