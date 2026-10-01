"""Every number in the report must trace to a logged measurement.

The point is not tidiness. Once in this project a conclusion was written from a
half-remembered figure rather than from a run, so the rule is that the ledger is
the only source and this script is what enforces it. It accepts the legitimate
re-renderings of a logged value -- percent vs fraction, and rounding to fewer
decimals -- and nothing else.
"""
import re, sys

def nums(t):
    return [m.group(1) for m in re.finditer(r'(?<![\w.])(\d*\.\d+|\d{2,})(?![\w])', t)]

def forms(n):
    """every spelling of one logged value that is still that value"""
    out = {n}
    try:
        f = float(n)
    except ValueError:
        return out
    for v in (f, f * 100, f / 100):          # fraction <-> percent
        for d in range(0, 6):                 # rounding to fewer decimals
            out.add(f"{v:.{d}f}")
        out.add(f"{v:g}")
        if v == int(v):
            out.add(str(int(v)))
    return out

# structural numbers: competition rules, ALC weights, IDs, section refs
SKIP = {'0.5','0.25','0.1','0.2','2026','10','30','23','59','17828','29785','31','15',
        '0.0','1.0','100','50','25','2.5','3.44','0.3377','80','8000','12','20',
        '4.1','4.2','4.3','4.4','5.1','5.2','5.3','3.1','3.2','2.1','2.2','2.3','1.1','1.2','1.3'}

def main(report, ledger):
    E = set()
    for n in nums(open(ledger).read().replace(',', '')):
        E |= forms(n)
    miss = []
    for ln, line in enumerate(open(report).read().splitlines(), 1):
        if line.strip().startswith('#'):
            continue
        for n in nums(line.replace(',', '')):
            if n in SKIP:
                continue
            if n not in E:
                miss.append((ln, n, line.strip()[:84]))
    print(f"{report}: {len(miss)} numeric tokens with no logged measurement")
    for ln, n, t in miss:
        print(f"  L{ln} {n}  {t}")
    return 1 if miss else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "report.md",
                  sys.argv[2] if len(sys.argv) > 2 else "evidence.md"))
