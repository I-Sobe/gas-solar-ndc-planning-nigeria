"""TASK 86 B.4: determine, for each limitations-register entry that cites a stored result file (by path or by
hash8), which result generation the cited files belong to. Generation = the token `gas_vintage_dispatch=True`
in the file's recorded arm tag (ARMS.json, the `# arm_tag:` header of a CSV, or the tag in the file name).
No token = blended dispatch (pre-vintage). Read-only unless --write, which appends a Generation line to the end
of each tagged entry in docs/limitations_register.md (idempotent: skips entries already carrying one)."""
import re, sys, glob, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "docs/limitations_register.md"
TOK = re.compile(r"gas_vintage_dispatch(?:=|\"?\s*:\s*)\"?(True|true)")
ANY = re.compile(r"gas_vintage_dispatch")
allfiles = [p for p in (ROOT / "results").rglob("*") if p.is_file()]

def gen_of_file(p):
    """'V' vintage, 'B' blended (no token), None undeterminable."""
    n = p.name
    if TOK.search(n) or "gas_vintage_dispatch=True" in str(p): return "V"
    try:
        if n == "ARMS.json" or n.endswith(".json"):
            t = p.read_text(errors="ignore")[:200000]
            if n == "ARMS.json" or "arm_tag" in t:
                return "V" if TOK.search(t) else "B"
        if n.endswith(".csv"):
            first = p.open(errors="ignore").readline()
            if first.startswith("# arm_tag:"):
                return "V" if TOK.search(first) else "B"
    except Exception:
        return None
    # sibling ARMS.json
    a = p.parent / "ARMS.json"
    if a.exists(): return gen_of_file(a)
    # arm tag spelled in the file name (long `__`-joined tag) and no token
    if n.count("__") >= 6: return "B"
    return None

def resolve(ref):
    """ref is a results/ path (may contain * or fragments) or hash8. Returns (exists, set(gens), example)."""
    if re.fullmatch(r"[0-9a-f]{8}", ref):
        fs = [p for p in allfiles if ref in p.as_posix()]
    else:
        ref = ref.rstrip(".,;:)`")
        g = glob.glob(str(ROOT / ref)) if any(c in ref for c in "*?[") else ([str(ROOT / ref)] if (ROOT / ref).exists() else [])
        fs = []
        for x in g:
            xp = Path(x)
            fs += [q for q in xp.rglob("*") if q.is_file()] if xp.is_dir() else [xp]
        if not fs and not any(c in ref for c in "*?["):
            # long name written with a truncated tail: prefix match
            pre = ROOT / ref
            fs = [p for p in allfiles if p.as_posix().startswith(pre.as_posix())]
    gens = {gen_of_file(p) for p in fs[:60]}
    return bool(fs), gens, (fs[0].relative_to(ROOT).as_posix() if fs else None)

def entries():
    t = REG.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=#{2,3} L\d+\b)", t)
    return parts[0], parts[1:]

def classify_entry(txt):
    """Only hash-keyed citations (arms_<h8>, <name>__<h8>) are tied to one generation; fixed-name, directory and
    pattern citations are overwritten by later runs (L142) and are reported as not established."""
    hashes = sorted(set(re.findall(r"(?:arms_|__)([0-9a-f]{8})\b", txt)))
    anyres = bool(re.search(r"results/", txt)) or bool(hashes)
    gens, ev, missing = [], [], []
    for h in hashes:
        ex, g, eg = resolve(h)
        if not ex: missing.append(h); continue
        g = g - {None}
        if g: gens.append(g); ev.append((h, "".join(sorted(g)), eg))
    return anyres, gens, ev, missing

def label(gens):
    flat = set().union(*gens) if gens else set()
    if not flat: return None
    if flat == {"V"}: return "vintage"
    if flat == {"B"}: return "blended"
    return "mixed"

if __name__ == "__main__":
    head, ents = entries()
    out = []
    from collections import Counter
    c = Counter()
    new = [head]
    for e in ents:
        m = re.match(r"#{2,3} (L\d+)\b", e)
        refs, gens, ev, missing = classify_entry(e)
        lab = label(gens)
        c[(bool(refs), lab)] += 1
        if refs: out.append((m.group(1), int(refs), lab, [x[0] for x in ev][:3], missing[:2]))
        if "--write" in sys.argv and refs and "**Generation (TASK 86):**" not in e:
            if lab:
                ex = "; ".join(f"`{x[0]}` ({'vintage' if 'V' in x[1] and 'B' not in x[1] else 'blended' if x[1]=='B' else 'both'})" for x in ev[:4])
                text = {"vintage": "vintage (`gas_vintage_dispatch=True`)", "blended": "blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records)",
                        "mixed": "MIXED: some cited files are vintage, some blended; each is named where its figure is quoted"}[lab]
                line = f"> **Generation (TASK 86):** the hash-keyed result files this entry cites are {text}. Checked: {ex}. Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142."
            else:
                line = ("> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations "
                        "(overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142).")
            e = e.rstrip("\n") + "\n\n" + line + "\n\n"
        new.append(e)
    for r in out: print(r)
    print(c)
    if "--write" in sys.argv:
        REG.write_text("".join(new), encoding="utf-8")
        print("written")
