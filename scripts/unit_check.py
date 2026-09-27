#!/usr/bin/env python3
# coding=utf-8
"""C / PyTorch 单元对比驱动器。

跑 ./rnn_unit 和 examples/rnn_unit.py，按输出协议配对，逐项对比并高亮差异。
不负责编译（自己先 make rnn_unit）。

本仓库用 uv 管理 python 环境，直接 uv run 即可（它会自动发现上一级的 .venv）：

用法:
    uv run scripts/unit_check.py                  # 跑 scripts/unit_cases 下所有单元
    uv run scripts/unit_check.py conv1d           # 只跑指定单元
    uv run scripts/unit_check.py -t 1e-3 conv1d   # 覆盖容差
    uv run scripts/unit_check.py -v conv1d        # 附带双方原始输出
    uv run scripts/unit_check.py --py /path/to/python conv1d

用例数据：scripts/unit_cases/<unit>.json。两侧程序自己按约定去这个路径找，
驱动器只负责把路径显式传下去并核对「双方是不是读了同一份数据」。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_DIR = os.path.join(ROOT, "scripts", "unit_cases")
C_BIN = os.path.join(ROOT, "rnn_unit")
PY_SCRIPT = os.path.join(ROOT, "examples", "rnn_unit.py")

DEFAULT_TOL = 1e-3
WARN_RATIO = 0.1          # max|Δ| 超过 tol*WARN_RATIO 就标黄预警

LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")

# --------------------------- 颜色 ---------------------------
# 非终端（比如重定向到文件）默认关色；想强行开就设 FORCE_COLOR=1
_COLOR = (sys.stdout.isatty() or os.environ.get("FORCE_COLOR")) and not os.environ.get("NO_COLOR")


def paint(text, code):
    if not _COLOR:
        return str(text)
    return "\033[%sm%s\033[0m" % (code, text)


def green(t):
    return paint(t, "32")


def red(t):
    return paint(t, "1;31")


def yellow(t):
    return paint(t, "33")


def dim(t):
    return paint(t, "2")


def bold(t):
    return paint(t, "1")


# --------------------------- 解释器探测 ---------------------------

_PY = None


def find_python(forced=None):
    """找一个能 import torch 的解释器。"""
    global _PY
    if _PY:
        return _PY
    cands = []
    if forced:
        cands.append(forced)
    env_py = os.environ.get("RNN_PYTHON")
    if env_py:
        cands.append(env_py)
    cands.append(sys.executable)
    d = ROOT
    for _ in range(4):                       # 自己目录及上级目录的 .venv
        cands.append(os.path.join(d, ".venv", "bin", "python"))
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    cands.append(shutil.which("python3"))
    cands.append("/usr/bin/python3")

    # 必须在 /tmp 下探测：仓库根目录里有个 torch/ 目录（上游代码），
    # 在根目录下跑 import torch 会命中它，产生假阳性
    probe_cwd = tempfile.gettempdir()
    seen = set()
    for cand in cands:
        if not cand or cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        try:
            r = subprocess.run([cand, "-I", "-c", "import torch"],
                               capture_output=True, cwd=probe_cwd)
        except OSError:
            continue
        if r.returncode == 0:
            _PY = cand
            return _PY
    return None


# --------------------------- 跑 + 解析 ---------------------------

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def parse(text):
    """把协议行解析成 {key: [float, ...]}。"""
    items = {}
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        nums = m.group(2).split()
        if not nums:
            continue
        try:
            items[m.group(1)] = [float(v) for v in nums]
        except ValueError:
            continue
    return items


def sort_key(key):
    name, _, frame = key.partition("#")
    return (name, int(frame) if frame.isdigit() else 0)


def data_source(stderr_text):
    for line in stderr_text.splitlines():
        if "case data:" in line:
            return line.split("case data:", 1)[1].strip()
    return "?"


def last_line(text):
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def short_src(src):
    """显示用：去掉冗长的路径。"""
    return src.split(" path=")[0]


def all_builtin(src):
    """三个字段是否全部退回了内置保底数据。

    注意要先去掉 path —— 用例文件名本身就以 .json 结尾，会干扰判断。
    """
    return "json" not in short_src(src)


# --------------------------- 比对 ---------------------------

def compare_item(cv, pv, tol):
    """返回 (状态, max_abs, max_rel, 超阈值下标)"""
    if len(cv) != len(pv):
        return "SIZE", float("nan"), float("nan"), []
    max_abs, max_rel, bad = 0.0, 0.0, []
    for i, (a, b) in enumerate(zip(cv, pv)):
        d = abs(a - b)
        max_abs = max(max_abs, d)
        denom = max(abs(a), abs(b), 1e-30)
        max_rel = max(max_rel, d / denom)
        if d > tol:
            bad.append(i)
    return ("PASS" if not bad else "FAIL"), max_abs, max_rel, bad


def render_unit(unit, c_items, p_items, tol, verbose, src_c, src_py,
                case_path=None, out=sys.stdout):
    keys = sorted(set(c_items) | set(p_items), key=sort_key)

    out.write("\n")
    out.write("%s  %s\n" % (bold(unit), dim("tol=%.1e" % tol)))
    out.write(dim("  case: %s" % (os.path.relpath(case_path, ROOT) if case_path else "-")))
    out.write(dim("   src : %s\n" % short_src(src_c)))
    out.write(dim("  %-24s %4s %13s %13s  %s\n" % ("item", "n", "max|Δ|", "max|Δ|rel", "result")))

    n_pass = n_fail = 0
    details = []
    for key in keys:
        cv, pv = c_items.get(key), p_items.get(key)
        if cv is None:
            n_fail += 1
            out.write("  %-24s %4s %13s %13s  %s\n" % (key, "-", "-", "-", red("仅 py 有，C 缺")))
            continue
        if pv is None:
            n_fail += 1
            out.write("  %-24s %4s %13s %13s  %s\n" % (key, "-", "-", "-", red("仅 C 有，py 缺")))
            continue

        status, max_abs, max_rel, bad = compare_item(cv, pv, tol)
        if status == "PASS":
            n_pass += 1
            res = green("PASS")
            diffs = dim("%13.3e" % max_abs)
            diffs_rel = dim("%13.3e" % max_rel)
            # 接近阈值就预警，别等它炸
            if max_abs > tol * WARN_RATIO:
                diffs = yellow("%13.3e" % max_abs)
        elif status == "SIZE":
            n_fail += 1
            res = red("SIZE")
            diffs = diffs_rel = "%13s" % "-"
        else:
            n_fail += 1
            res = red("FAIL")
            diffs = red("%13.3e" % max_abs)
            diffs_rel = red("%13.3e" % max_rel)
            details.append((key, cv, pv, max_abs, bad))

        out.write("  %-24s %4d %s %s  %s\n" % (key, len(cv), diffs, diffs_rel, res))

    for key, cv, pv, max_abs, bad in details:
        out.write("\n")
        out.write("  %s %s   %s\n" % (red(">>> " + key),
                                       dim("逐元素对比，标红的是超过 %.1e 的位置" % tol),
                                       dim("max|Δ|=%.3e" % max_abs)))
        out.write(dim("    %6s  %17s  %17s  %11s\n" % ("idx", "C", "PyTorch", "|Δ|")))
        bad_set = set(bad)
        for i, (a, b) in enumerate(zip(cv, pv)):
            d = abs(a - b)
            line = "    %6d  %17.8e  %17.8e  %11.2e" % (i, a, b, d)
            if i in bad_set:
                out.write(red(line + "  <-- 超阈值") + "\n")
            elif d > tol * WARN_RATIO:
                out.write(yellow(line) + "\n")
            else:
                out.write(line + "\n")

    if verbose:
        out.write("\n" + dim("  ---- C 原始输出 ----"))
        for key in keys:
            if key in c_items:
                out.write("\n  " + dim(key) + " " + " ".join("%.8e" % v for v in c_items[key]))
        out.write("\n" + dim("  ---- PyTorch 原始输出 ----"))
        for key in keys:
            if key in p_items:
                out.write("\n  " + dim(key) + " " + " ".join("%.8e" % v for v in p_items[key]))
        out.write("\n")

    return n_pass, n_fail


# --------------------------- 主流程 ---------------------------

def discover_units():
    if not os.path.isdir(CASES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CASES_DIR) if f.endswith(".json"))


def main(argv):
    ap = argparse.ArgumentParser(description="C / PyTorch 单元输出对比")
    ap.add_argument("units", nargs="*", help="要跑的单元名；缺省＝scripts/unit_cases 下全部")
    ap.add_argument("-t", "--tol", type=float, default=None, help="覆盖容差（缺省读 JSON 的 tolerance，再缺省 %.0e）" % DEFAULT_TOL)
    ap.add_argument("-v", "--verbose", action="store_true", help="附带打印双方原始输出")
    ap.add_argument("--py", default=None, help="指定 python 解释器（需能 import torch）；缺省自动探测 uv 环境")
    args = ap.parse_args(argv[1:])

    if not os.path.exists(C_BIN):
        print(red("找不到 %s，先跑 make rnn_unit" % C_BIN))
        return 2

    py = find_python(args.py)
    if py is None:
        print(red("找不到能 import torch 的 python 解释器。"))
        print(dim("  本仓库用 uv 管理环境，直接 `uv run scripts/unit_check.py` 即可；"))
        print(dim("  也可以 --py <解释器路径> 或设置 RNN_PYTHON。"))
        return 2

    units = args.units or discover_units()
    if not units:
        print(red("scripts/unit_cases 下没有 .json 用例；用命令行显式指定单元名"))
        return 2

    print(dim("C      : %s" % C_BIN))
    print(dim("python : %s (%s)" % (py, "examples/rnn_unit.py")))

    total_pass = total_fail = n_builtin = 0
    for unit in units:
        case_path = os.path.join(CASES_DIR, "%s.json" % unit)
        has_case = os.path.isfile(case_path)

        tol = args.tol
        if tol is None and has_case:
            try:
                with open(case_path, "r", encoding="utf-8") as fh:
                    tol = float(json.load(fh).get("tolerance", DEFAULT_TOL))
            except (OSError, ValueError, TypeError):
                tol = DEFAULT_TOL
        if tol is None:
            tol = DEFAULT_TOL

        rc = run([C_BIN, unit] + ([case_path] if has_case else []))
        rp = run([py, PY_SCRIPT, unit] + ([case_path] if has_case else []))

        c_items = parse(rc.stdout)
        p_items = parse(rp.stdout)

        src_c = data_source(rc.stderr)
        src_py = data_source(rp.stderr)

        # 某一侧整个没跑起来，就直接报错，不要伪装成"N 项都缺"
        if not c_items or not p_items:
            print("\n" + red("%s: 输出缺失，无法对比" % unit))
            if not c_items:
                print("  " + red("C ") + "退出码 %d  %s" % (rc.returncode, last_line(rc.stderr)))
            if not p_items:
                print("  " + red("py") + "退出码 %d  %s" % (rp.returncode, last_line(rp.stderr)))
            total_fail += 1
            continue

        # 逐字段来源必须完全一致，否则比的是两份不同的数据
        if src_c != src_py:
            print(yellow("\n%s: 数据来源不一致！C=%s / py=%s" % (unit, src_c, src_py)))

        # 两侧都退回保底：结论是空的，必须显眼地说出来，不能让人以为"测过了"
        if all_builtin(src_c) and all_builtin(src_py):
            n_builtin += 1
            print("\n" + red("=" * 60))
            print(red("%s: 两侧都在用内置保底数据，本次对比不验证真实用例数据！" % unit))
            if not has_case:
                print(red("  没找到用例文件: %s" % case_path))
                print(dim("  新建该文件后重跑即可；字段约定见 doc/补充学习手册.md 第一节"))
            else:
                print(red("  用例文件存在但没被读进去: %s" % case_path))
            print(red("=" * 60))
        elif all_builtin(src_c) or all_builtin(src_py):
            print(yellow("\n%s: 有一侧退回了内置保底数据，请确认这是预期的" % unit))

        n_pass, n_fail = render_unit(unit, c_items, p_items, tol, args.verbose,
                                     src_c, src_py, case_path)
        total_pass += n_pass
        total_fail += n_fail

    print("\n" + ("=" * 60))
    if total_fail == 0:
        print(green("全部通过：%d 项" % total_pass))
    else:
        print(red("失败 %d 项，通过 %d 项" % (total_fail, total_pass)))
    if n_builtin:
        print(red("其中 %d 个单元跑在内置保底数据上，这些单元的结论不算数" % n_builtin))
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
