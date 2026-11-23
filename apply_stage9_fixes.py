#!/usr/bin/env python3
"""
Apply the STAGE 8 / STAGE 9 fixes to src/normalizer.py and src/evaluator.py.

Usage (from the repository root):
    python apply_stage9_fixes.py --dry-run     # show what would change, write nothing
    python apply_stage9_fixes.py               # patch the files (makes .bak_stage9_fix backups)
    python apply_stage9_fixes.py --run         # patch, then back up the JSON and run Stage 8 + 9

Properties:
  * Idempotent: edits that are already applied are skipped.
  * All-or-nothing: if any edit cannot be located or the result does not
    parse as Python, nothing is written.
  * Preserves CRLF/LF line endings.
"""
import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

WARNINGS = []


class PatchError(Exception):
    pass


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_text(path: Path):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def write_text(path: Path, text: str, crlf: bool):
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def find_def_start(text: str, name: str) -> int:
    m = re.search(rf"^[ \t]*def {re.escape(name)}\(", text, re.M)
    if not m:
        raise PatchError(f"could not find 'def {name}(' in file")
    return m.start()


# --------------------------------------------------------------------------
# normalizer.py
# --------------------------------------------------------------------------
def patch_normalizer(t: str):
    log = []
    marker = 'rec[f"{field}_norm_flags"] = field_flags'
    if marker in t:
        log.append("[1] normalizer: per-engine flags already preserved - skipped")
        return t, log

    pat = re.compile(r'^(?P<i>[ \t]*)rec\[f"\{field\}_norm"\] = norm[ \t]*\n', re.M)
    ms = list(pat.finditer(t))
    if len(ms) != 1:
        raise PatchError(
            f'[1] normalizer: expected exactly 1 match of rec[f"{{field}}_norm"] = norm, found {len(ms)}'
        )
    m = ms[0]
    if not re.search(r"\bfield_flags\b", t):
        WARNINGS.append(
            "[1] normalizer: variable 'field_flags' is not defined anywhere in normalizer.py; "
            "check that normalize_records() really names the flags variable that way."
        )
    t = t[: m.end()] + f"{m.group('i')}{marker}\n" + t[m.end():]
    log.append("[1] normalizer: added rec[f\"{field}_norm_flags\"] = field_flags")
    return t, log


# --------------------------------------------------------------------------
# evaluator.py
# --------------------------------------------------------------------------
HELPER = '''def _normalized_text_and_flags(
    rec: Dict[str, Any], field: str
) -> Tuple[Optional[str], List[str]]:
    """Read Stage 8 output and preserve flags from the original raw text."""
    norm_field = f"{field}_norm"
    field_flags = f"{field}_norm_flags"

    if norm_field in rec:
        text = rec.get(norm_field)
    else:
        text = normalize_arabic(rec.get(field))

    saved_flags = rec.get(field_flags)
    if isinstance(saved_flags, list):
        flags = saved_flags
    else:
        # Recompute from raw text, not normalized text: decimal_number
        # disappears once a decimal has been expanded into Arabic words.
        _, raw_flags = normalize_with_flags(rec.get(field))
        legacy_flags = rec.get("norm_flags", [])
        if not isinstance(legacy_flags, list):
            legacy_flags = []
        flags = list(set(raw_flags) | set(legacy_flags))

    return text, sorted(set(flags))'''

QUANTILE = '''def quantile(q: float) -> Optional[float]:
    if not wers:
        return None
    if len(wers) == 1:
        return round(wers[0], 4)
    return round(
        statistics.quantiles(wers, n=100, method="inclusive")[int(q * 100) - 1],
        4,
    )'''

WER_ENTRY_BODY = [
    '"n": len(wers),',
    '"median": None if not wers else round(statistics.median(wers), 4),',
    '"p75": quantile(0.75),',
    '"p90": quantile(0.90),',
    '"mean": None if not wers else round(statistics.mean(wers), 4),',
]

NEW_DOC_SENTENCE = (
    'Engine A is the reference and Engine B is the hypothesis, so the numbers\n'
    'read as "how much does B differ from A".'
)


def patch_docstring(t: str, log: list):
    if "how much does B differ from A" in t:
        log.append("[2] evaluator: docstring already updated - skipped")
        return t
    m = re.search(r"Engine B[^.]*?\bMMS\b[^.]*\.", t)
    if not m:
        WARNINGS.append(
            "[2] evaluator: could not find the 'Engine B ... MMS' sentence; edit the "
            "module docstring by hand (see fix #2)."
        )
        return t
    log.append(f"[2] evaluator: docstring sentence replaced (was: {' '.join(m.group(0).split())!r})")
    start = m.start()
    line_start = t.rfind("\n", 0, start) + 1
    if t[line_start:start].strip():
        WARNINGS.append(
            "[2] evaluator: text precedes the replaced sentence on its line "
            f"({t[line_start:start].strip()!r}); review the docstring."
        )
    return t[:start] + NEW_DOC_SENTENCE + t[m.end():]


def patch_helper(t: str, log: list):
    if "def _normalized_text_and_flags(" in t:
        log.append("[3] evaluator: helper already present - skipped")
        return t
    lines = t.split("\n")
    start_off = find_def_start(t, "compute_wer_cer")
    start = t.count("\n", 0, start_off)
    # find the next top-level statement after the function
    j = start + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() and indent_of(ln) == 0 and not ln.lstrip().startswith(")"):
            break
        j += 1
    end = j
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    lines[end:end] = ["", ""] + HELPER.split("\n")
    log.append("[3] evaluator: inserted _normalized_text_and_flags() after compute_wer_cer()")
    return "\n".join(lines)


def patch_triage_call(t: str, log: list):
    new_marker = '_normalized_text_and_flags(rec, "asr_a_text")'
    if new_marker in t and "_normalized_text_and_flags(rec, \"asr_b_text\")" in t:
        log.append("[4] evaluator: triage_clip already uses helper - skipped")
        return t
    pat = re.compile(
        r'(?P<i>[ \t]*)text_a, flags_a = normalize_with_flags\(rec\.get\("asr_a_text_norm", rec\.get\("asr_a_text"\)\)\)[ \t]*\n'
        r'[ \t]*raw_b = rec\.get\("asr_b_text_norm", rec\.get\("asr_b_text"\)\)[ \t]*\n'
        r'[ \t]*text_b, flags_b = normalize_with_flags\(raw_b\)'
    )
    ms = list(pat.finditer(t))
    if len(ms) != 1:
        raise PatchError(f"[4] evaluator: expected exactly 1 match of the triage_clip block, found {len(ms)}")
    m = ms[0]
    i = m.group("i")
    repl = (
        f'{i}text_a, flags_a = _normalized_text_and_flags(rec, "asr_a_text")\n'
        f'{i}text_b, flags_b = _normalized_text_and_flags(rec, "asr_b_text")'
    )
    t = t[: m.start()] + repl + t[m.end():]
    if re.search(r"\braw_b\b", t):
        WARNINGS.append("[4] evaluator: 'raw_b' is still referenced elsewhere; check triage_clip().")
    log.append("[4] evaluator: triage_clip now uses _normalized_text_and_flags()")
    return t


def patch_empty_b(t: str, log: list):
    off = find_def_start(t, "triage_clip")
    m = re.compile(r"\bif text_b is None:").search(t, off)
    if m:
        t = t[: m.start()] + "if not text_b:" + t[m.end():]
        log.append("[5] evaluator: 'if text_b is None:' -> 'if not text_b:'")
        return t
    if re.compile(r"\bif not text_b:").search(t, off):
        log.append("[5] evaluator: empty Engine B check already applied - skipped")
        return t
    raise PatchError("[5] evaluator: could not find 'if text_b is None:' inside/after triage_clip()")


def patch_quantile(t: str, log: list):
    if 'method="inclusive"' in t:
        log.append("[6a] evaluator: quantile() already updated - skipped")
        return t
    s_off = find_def_start(t, "summarize")
    s_line = t.count("\n", 0, s_off)
    lines = t.split("\n")
    qi = None
    for k in range(s_line, len(lines)):
        if re.match(r"\s*def quantile\(", lines[k]):
            qi = k
            break
    if qi is None:
        raise PatchError("[6a] evaluator: could not find 'def quantile(' inside summarize()")
    n = indent_of(lines[qi])
    j = qi + 1
    while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > n):
        j += 1
    end = j
    while end > qi + 1 and not lines[end - 1].strip():
        end -= 1
    pad = lines[qi][:n]
    lines[qi:end] = [(pad + l) if l else l for l in QUANTILE.split("\n")]
    log.append("[6a] evaluator: replaced quantile() helper")
    return "\n".join(lines)


def patch_wer_dict(t: str, log: list):
    if '"median": None if not wers' in t:
        log.append("[6b] evaluator: 'wer' summary entry already updated - skipped")
        return t
    s_off = find_def_start(t, "summarize")
    m = re.compile(r'"wer"\s*:\s*\{').search(t, s_off)
    if not m:
        raise PatchError('[6b] evaluator: could not find the "wer": { entry in summarize()')
    depth = 0
    k = m.end() - 1
    while k < len(t):
        c = t[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    else:
        raise PatchError('[6b] evaluator: unbalanced braces in the "wer" entry')
    line_start = t.rfind("\n", 0, m.start()) + 1
    pad = t[line_start:m.start()]
    if pad.strip():
        pad = pad[: indent_of(pad)]
    body_pad = pad + "    "
    new = '"wer": {\n' + "\n".join(body_pad + b for b in WER_ENTRY_BODY) + f"\n{pad}}}"
    log.append('[6b] evaluator: replaced "wer" summary entry')
    return t[: m.start()] + new + t[k + 1:]


def ensure_imports(t: str, log: list):
    names = ["Any", "Dict", "List", "Optional", "Tuple"]
    m = re.search(r"^from typing import ([^\n(]+)$", t, re.M)
    if m:
        have = {x.strip() for x in m.group(1).split(",")}
        missing = [n for n in names if n not in have]
        if missing:
            merged = sorted(have | set(missing))
            t = t[: m.start()] + "from typing import " + ", ".join(merged) + t[m.end():]
            log.append(f"[imports] evaluator: added to typing import: {', '.join(missing)}")
    elif not re.search(r"^from typing import \(", t, re.M):
        WARNINGS.append("evaluator: no 'from typing import ...' line found; make sure Any, Dict, List, Optional, Tuple are imported.")
    else:
        WARNINGS.append("evaluator: multi-line typing import found; make sure Any, Dict, List, Optional, Tuple are included.")
    if not re.search(r"^\s*import statistics\b|^\s*import .*\bstatistics\b", t, re.M):
        WARNINGS.append("evaluator: 'import statistics' not found; summarize() needs it.")
    if len(re.findall(r"\bnormalize_arabic\b", t)) < 2:
        WARNINGS.append("evaluator: 'normalize_arabic' does not appear to be imported/defined; the helper calls it.")
    if "normalize_with_flags" not in t.replace("def _normalized_text_and_flags", ""):
        WARNINGS.append("evaluator: 'normalize_with_flags' not found; the helper calls it.")
    return t


def patch_evaluator(t: str):
    log = []
    t = patch_docstring(t, log)
    t = patch_helper(t, log)
    t = patch_triage_call(t, log)
    t = patch_empty_b(t, log)
    t = patch_quantile(t, log)
    t = patch_wer_dict(t, log)
    t = ensure_imports(t, log)
    return t, log


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(cmd, cwd):
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"Command failed with exit code {r.returncode}: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--video-id", default="OiDI2dckckc")
    ap.add_argument("--dry-run", action="store_true", help="report changes but write nothing")
    ap.add_argument("--run", action="store_true", help="after patching, back up the JSON and run Stage 8 and Stage 9")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    norm_path = root / "src" / "normalizer.py"
    eval_path = root / "src" / "evaluator.py"
    for p in (norm_path, eval_path):
        if not p.exists():
            sys.exit(f"Missing file: {p}\nRun this from the repository root (or pass --root).")

    results = []
    try:
        for path, patcher in ((norm_path, patch_normalizer), (eval_path, patch_evaluator)):
            text, crlf = read_text(path)
            new, log = patcher(text)
            try:
                ast.parse(new)
            except SyntaxError as e:
                raise PatchError(f"{path.name}: patched result is not valid Python ({e}); nothing written")
            results.append((path, text, new, crlf, log))
    except PatchError as e:
        sys.exit(f"ERROR: {e}\nNo files were modified.")

    for path, old, new, crlf, log in results:
        print(f"\n== {path.relative_to(root)} ==")
        for line in log:
            print("  " + line)
        if old == new:
            print("  (no changes needed)")

    for w in WARNINGS:
        print(f"WARNING: {w}")

    if args.dry_run:
        print("\nDry run: no files written.")
        return

    for path, old, new, crlf, _ in results:
        if old == new:
            continue
        bak = path.with_suffix(path.suffix + ".bak_stage9_fix")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"Backup: {bak.relative_to(root)}")
        write_text(path, new, crlf)
        print(f"Patched: {path.relative_to(root)}")

    if args.run:
        meta = root / "metadata" / f"{args.video_id}_transcripts.json"
        if not meta.exists():
            sys.exit(f"Missing transcript JSON: {meta}")
        backup = meta.with_name(f"{args.video_id}_transcripts.before_stage9_fix.json")
        if backup.exists():
            print(f"Backup already exists, keeping the original: {backup.relative_to(root)}")
        else:
            shutil.copy2(meta, backup)
            print(f"Backup: {backup.relative_to(root)}")
        run([sys.executable, "src/normalizer.py", "--video-id", args.video_id], root)
        run([sys.executable, "src/evaluator.py", "--video-id", args.video_id], root)
        print("\nDone. Keep the backup JSON until the results are checked.")
    else:
        print(
            "\nPatched. To regenerate Stage 8 and run Stage 9, re-run with --run, or manually:\n"
            f"  python src/normalizer.py --video-id {args.video_id}\n"
            f"  python src/evaluator.py --video-id {args.video_id}"
        )


if __name__ == "__main__":
    main()
