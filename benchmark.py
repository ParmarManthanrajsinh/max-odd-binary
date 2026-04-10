#!/usr/bin/env python3
"""
Benchmark: Maximum Odd Binary Number
Profiles solutions across array languages and C++.
Generates a horizontal bar chart with language logos.
"""

import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ─────────────────────────── Configuration ──────────────────────

N_ITERS = 50
N_RUNS = 5
INPUT_LEN = 1_000
INPUT_SIZES = [100, 1_000, 10_000, 100_000]

ROOT = Path(__file__).parent
BUILD_DIR = ROOT / "build"
LOGOS_DIR = ROOT / "logos"
OUTPUT_PNG = ROOT / "img" / "benchmark_results.png"
CACHE_FILE = ROOT / "benchmark_cache.json"


# ─────────────────────────── Helpers ────────────────────────────


def find_cmd(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def parse_number(text):
    """Extract the last float from text, handling BQN ¯, J _, and TinyAPL ⏨."""
    if not text:
        return None
    text = text.replace("\u00af", "-")  # BQN high minus
    text = text.replace("\u2a28", "e")  # TinyAPL ⏨ scientific notation
    text = text.replace("\u23e8", "e")  # TinyAPL ⏨ (alternate codepoint)
    for line in reversed(text.strip().splitlines()):
        cleaned = line.strip().replace("_", "-")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        if m:
            try:
                return float(m.group())
            except ValueError:
                continue
    return None


def run_cmd(cmd, *, input_text=None, env=None, timeout=120):
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=input_text,
            env=env,
            timeout=timeout,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except Exception as e:
        return "", str(e), -1


def write_temp(suffix, content):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ─────────────────────────── Benchmark: BQN ─────────────────────


def bench_bqn():
    interp = find_cmd("cbqn", "bqn")
    if not interp:
        return None
    script = f'\u2022Out \u2022Fmt {N_ITERS}(1\u233d\u2228)\u2022_timed {INPUT_LEN}\u294a"01"'
    out, _, rc = run_cmd([interp, "-e", script])
    return parse_number(out) if rc == 0 else None


def bench_bqn_count():
    interp = find_cmd("cbqn", "bqn")
    if not interp:
        return None
    fn = '{n\u2190+\u00b4\u00271\u0027=\U0001d569\u22c4("1"/\u02dcn-1)\u223e("0"/\u02dc(\u2260\U0001d569)-n)\u223e\u00271\u0027}'
    script = f'\u2022Out \u2022Fmt {N_ITERS}({fn})\u2022_timed {INPUT_LEN}\u294a"01"'
    out, _, rc = run_cmd([interp, "-e", script])
    return parse_number(out) if rc == 0 else None


def bench_bqn_sort_alt():
    interp = find_cmd("cbqn", "bqn")
    if not interp:
        return None
    fn = '"01"\u228f\u02dc1\u233d\u00b7\u2228\'1\'\u22b8='
    script = f'\u2022Out \u2022Fmt {N_ITERS}({fn})\u2022_timed {INPUT_LEN}\u294a"01"'
    out, _, rc = run_cmd([interp, "-e", script])
    return parse_number(out) if rc == 0 else None


def bench_bqn_tacit():
    interp = find_cmd("cbqn", "bqn")
    if not interp:
        return None
    fn = '"101"/\u02dc(\u2260(1\u223e\u02dc(1-\u02dc\u22a2)\u223e-)(+\u00b4\u00271\u0027\u22b8=))'
    script = f'\u2022Out \u2022Fmt {N_ITERS}({fn})\u2022_timed {INPUT_LEN}\u294a"01"'
    out, _, rc = run_cmd([interp, "-e", script])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: Uiua ────────────────────


def bench_uiua():
    interp = find_cmd("uiua")
    if not interp:
        return None
    code = (
        f'S \u2190 \u21af{INPUT_LEN} "01"\n'
        f"T \u2190 now\n"
        f"\u2365(\u25cc\u21bb1\u21cc\u2346 S){N_ITERS}\n"
        f"&p \u00f7{N_ITERS} -T now\n"
    )
    path = write_temp(".ua", code)
    try:
        out, err, rc = run_cmd([interp, "run", path])
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


def bench_uiua_count():
    interp = find_cmd("uiua")
    if not interp:
        return None
    # \u02dc=˜ \u25bd=▽ \u2282=⊂ \u2283=⊃ \u2081=₁ \u29e7=⧻
    code = (
        f'S \u2190 \u21af{INPUT_LEN} "01"\n'
        f"T \u2190 now\n"
        f'\u2365(\u25cc\u02dc\u25bd"101"\u02dc\u22821\u2282\u2283-\u2081-\u2283/+\u29fb=@1 S){N_ITERS}\n'
        f"&p \u00f7{N_ITERS} -T now\n"
    )
    path = write_temp(".ua", code)
    try:
        out, err, rc = run_cmd([interp, "run", path])
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: J ───────────────────────


def bench_j():
    interp = find_cmd("jconsole", "ijconsole")
    if not interp:
        return None
    code = (
        f"input =: {INPUT_LEN} $ '01'\n"
        f"n =: {N_ITERS}\n"
        "bench =: 3 : '1 |. \\:~ input'\n"
        "echo (6!:2 'bench\"0 i. n') % n\n"
        "exit ''\n"
    )
    path = write_temp(".ijs", code)
    try:
        out, _, rc = run_cmd([interp, path])
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


def bench_j_count():
    interp = find_cmd("jconsole", "ijconsole")
    if not interp:
        return None
    code = (
        f"input =: {INPUT_LEN} $ '01'\n"
        f"n =: {N_ITERS}\n"
        "Mob =: '101'#~1,~#(<:@],-)([:+/'1'=[)\n"
        "bench =: 3 : 'Mob input'\n"
        "echo (6!:2 'bench\"0 i. n') % n\n"
        "exit ''\n"
    )
    path = write_temp(".ijs", code)
    try:
        out, _, rc = run_cmd([interp, path])
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: Dyalog APL ──────────────


def bench_apl():
    interp = find_cmd("dyalog")
    if not interp:
        return None
    apl_n = N_ITERS * 200
    code = (
        f"input\u2190{INPUT_LEN}\u2374'01'\n"
        f"n\u2190{apl_n}\n"
        "t\u21903\u2283\u2395AI\n"
        "_\u2190{(1\u233d\u2282\u2364\u2352\u235b\u2337)input}\u00a8\u2373n\n"
        "\u2395\u2190'RESULT ',(\u2355((3\u2283\u2395AI)-t)\u00f71000\u00d7n)\n"
        "\u2395OFF\n"
    )
    env = os.environ.copy()
    env["RIDE_INIT"] = "SERVE::0"
    out, _, _ = run_cmd([interp], input_text=code, env=env)
    if not out:
        return None
    for line in out.splitlines():
        if "RESULT" in line:
            return parse_number(line.split("RESULT")[-1])
    return None


def bench_apl_count():
    interp = find_cmd("dyalog")
    if not interp:
        return None
    apl_n = N_ITERS * 200
    code = (
        f"input\u2190{INPUT_LEN}\u2374'01'\n"
        f"n\u2190{apl_n}\n"
        "Mob\u2190'101'/\u23681,\u2368\u2262((1-\u2368\u22a2),-)( +/'1'\u2218=)\n"
        "t\u21903\u2283\u2395AI\n"
        "_\u2190{Mob input}\u00a8\u2373n\n"
        "\u2395\u2190'RESULT ',(\u2355((3\u2283\u2395AI)-t)\u00f71000\u00d7n)\n"
        "\u2395OFF\n"
    )
    env = os.environ.copy()
    env["RIDE_INIT"] = "SERVE::0"
    out, _, _ = run_cmd([interp], input_text=code, env=env)
    if not out:
        return None
    for line in out.splitlines():
        if "RESULT" in line:
            return parse_number(line.split("RESULT")[-1])
    return None


# ─────────────────────────── Benchmark: TinyAPL ─────────────────


def bench_tinyapl(solution_code):
    """TinyAPL: uses ⎕_Measure for native timing (returns seconds)."""
    interp = find_cmd("tinyapl")
    if not interp:
        return None
    code = f"\u2395\u2190({solution_code}) \u2395_Measure {INPUT_LEN}\u2374'01'\n"
    path = write_temp(".apl", code)
    try:
        out, err, rc = run_cmd([interp, path])
        combined = (out or "") + (err or "")
        if "error" in combined.lower():
            return None
        return parse_number(out)
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: Kap ─────────────────────


def bench_kap():
    """Kap: uses time:measureTime for native timing."""
    kap_path = Path.home() / "Downloads/kap/gui2/bin/kap-jvm"
    if not kap_path.exists():
        return None
    expr = (
        f'input \u2190 {INPUT_LEN}\u2374"01" \u25ca '
        f"time:measureTime {{ {{ 1\u233d\u2228 input }} \u00a8 \u2373{N_ITERS} }}"
    )
    out, err, rc = run_cmd([str(kap_path), "-n", "-E", expr], timeout=30)
    if rc != 0:
        return None
    for line in (out or "").splitlines():
        if "Total time:" in line:
            m = re.search(r"Total time:\s*([\d.]+)", line)
            if m:
                return float(m.group(1)) / N_ITERS
    return None


# ─────────────────────────── Benchmark: C++ ─────────────────────

_cpp_bin_cache = {}


def bench_cpp(name, func_body):
    compiler = find_cmd("g++")
    if not compiler:
        return None

    if name not in _cpp_bin_cache:
        BUILD_DIR.mkdir(exist_ok=True)
        src = (
            "#include <algorithm>\n"
            "#include <chrono>\n"
            "#include <cstring>\n"
            "#include <iostream>\n"
            "#include <string>\n\n"
            "auto maximum_odd_binary(std::string s) -> std::string {\n"
            f"  {func_body}\n"
            "  return s;\n"
            "}\n\n"
            "int main() {\n"
            "  std::string input;\n"
            f'  for (int i = 0; i < {INPUT_LEN // 2}; ++i) input += "01";\n'
            f"  constexpr int n = {N_ITERS};\n"
            "  std::string result;\n"
            "  auto start = std::chrono::high_resolution_clock::now();\n"
            "  for (int i = 0; i < n; ++i)\n"
            "    result = maximum_odd_binary(input);\n"
            "  auto end = std::chrono::high_resolution_clock::now();\n"
            "  std::cout << std::chrono::duration<double>(end - start).count() / n\n"
            "            << '\\n';\n"
            "  if (result.empty()) return 1;\n"
            "}\n"
        )
        src_path = BUILD_DIR / f"bench_{name}.cpp"
        bin_path = BUILD_DIR / f"bench_{name}"
        src_path.write_text(src)

        for std_flag in ["-std=c++23", "-std=c++20"]:
            _, cerr, crc = run_cmd(
                [compiler, std_flag, "-O2", "-o", str(bin_path), str(src_path)]
            )
            if crc == 0:
                break
        else:
            _cpp_bin_cache[name] = None
            return None
        _cpp_bin_cache[name] = str(bin_path)

    bin_path = _cpp_bin_cache[name]
    if bin_path is None:
        return None
    out, _, rc = run_cmd([bin_path])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: D  ─────────────────────

_d_bin_cache = {}


def bench_d(name, func_body):
    compiler = find_cmd("ldc2")
    if not compiler:
        ldc_home = Path.home() / "dlang"
        if ldc_home.is_dir():
            candidates = sorted(ldc_home.glob("ldc-*/bin/ldc2"), reverse=True)
            if candidates and candidates[0].exists():
                compiler = str(candidates[0])
    if not compiler:
        return None

    if name not in _d_bin_cache:
        BUILD_DIR.mkdir(exist_ok=True)
        src = (
            "import core.time;\n"
            "import std.algorithm;\n"
            "import std.datetime.stopwatch;\n"
            "import std.stdio;\n"
            "import std.range;\n"
            "import std.array;\n"
            "import std.string : representation;\n\n"
            "auto maximumOddBinary(char[] s){\n"
            f"  {func_body}\n"
            "  return s;\n"
            "}\n\n"
            "int main() {\n"
            f'  char[] input = cast(char[])"01".repeat({INPUT_LEN // 2}).join.array;\n'
            f"  enum n = {N_ITERS};\n"
            "  char[] result;\n"
            "  auto sw = StopWatch(AutoStart.yes);\n"
            "  foreach(i; 0 .. n)\n"
            "    result = maximumOddBinary(input.dup);\n"
            "  sw.stop();\n"
            "  double dur = sw.peek.total!`nsecs`;\n"
            "  writeln(dur / (1e9 * n));\n"
            "  if (result.length == 0) return 1;\n"
            "  return 0;\n"
            "}\n"
        )
        src_path = BUILD_DIR / f"bench_{name}.d"
        bin_path = BUILD_DIR / f"bench_{name}"
        src_path.write_text(src)
        _, cerr, crc = run_cmd(
            [
                compiler,
                "-O3",
                f"-of={str(bin_path)}",
                str(src_path),
            ]
        )
        if crc != 0:
            _d_bin_cache[name] = None
            return None
        _d_bin_cache[name] = str(bin_path)

    bin_path = _d_bin_cache[name]
    if bin_path is None:
        return None
    out, _, rc = run_cmd([bin_path])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: Rust ────────────────────

_rust_bin_cache = {}


def bench_rust(name, func_body):
    compiler = find_cmd("rustc")
    if not compiler:
        return None

    if name not in _rust_bin_cache:
        BUILD_DIR.mkdir(exist_ok=True)
        src = (
            "use std::time::Instant;\n\n"
            "fn maximum_odd_binary(mut s: Vec<u8>) -> Vec<u8> {\n"
            f"    {func_body}\n"
            "    s\n"
            "}\n\n"
            "fn main() {\n"
            f'    let input: Vec<u8> = "01".repeat({INPUT_LEN // 2}).into_bytes();\n'
            f"    let n = {N_ITERS};\n"
            "    let mut result = Vec::new();\n"
            "    let start = Instant::now();\n"
            "    for _ in 0..n {\n"
            "        result = maximum_odd_binary(input.clone());\n"
            "    }\n"
            "    let elapsed = start.elapsed().as_secs_f64();\n"
            '    println!("{}", elapsed / n as f64);\n'
            "    if result.is_empty() { std::process::exit(1); }\n"
            "}\n"
        )
        src_path = BUILD_DIR / f"bench_{name}.rs"
        bin_path = BUILD_DIR / f"bench_{name}"
        src_path.write_text(src)
        _, cerr, crc = run_cmd([compiler, "-O", "-o", str(bin_path), str(src_path)])
        if crc != 0:
            _rust_bin_cache[name] = None
            return None
        _rust_bin_cache[name] = str(bin_path)

    bin_path = _rust_bin_cache[name]
    if bin_path is None:
        return None
    out, _, rc = run_cmd([bin_path])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: Rust nightly ────────────

_rust_nightly_bin_cache = {}


def bench_rust_nightly(name, func_body):
    rustup = find_cmd("rustup")
    if not rustup:
        return None

    if name not in _rust_nightly_bin_cache:
        BUILD_DIR.mkdir(exist_ok=True)
        src = (
            "#![feature(iter_partition_in_place)]\n\n"
            "use std::time::Instant;\n\n"
            "fn maximum_odd_binary(mut s: Vec<u8>) -> Vec<u8> {\n"
            f"    {func_body}\n"
            "    s\n"
            "}\n\n"
            "fn main() {\n"
            f'    let input: Vec<u8> = "01".repeat({INPUT_LEN // 2}).into_bytes();\n'
            f"    let n = {N_ITERS};\n"
            "    let mut result = Vec::new();\n"
            "    let start = Instant::now();\n"
            "    for _ in 0..n {\n"
            "        result = maximum_odd_binary(input.clone());\n"
            "    }\n"
            "    let elapsed = start.elapsed().as_secs_f64();\n"
            '    println!("{}", elapsed / n as f64);\n'
            "    if result.is_empty() { std::process::exit(1); }\n"
            "}\n"
        )
        src_path = BUILD_DIR / f"bench_{name}.rs"
        bin_path = BUILD_DIR / f"bench_{name}"
        src_path.write_text(src)
        _, cerr, crc = run_cmd(
            [
                rustup,
                "run",
                "nightly",
                "rustc",
                "-O",
                "-o",
                str(bin_path),
                str(src_path),
            ]
        )
        if crc != 0:
            _rust_nightly_bin_cache[name] = None
            return None
        _rust_nightly_bin_cache[name] = str(bin_path)

    bin_path = _rust_nightly_bin_cache[name]
    if bin_path is None:
        return None
    out, _, rc = run_cmd([bin_path])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: Nim ─────────────────────

_nim_bin_cache = {}


def bench_nim(name, func_body):
    compiler = find_cmd("nim")
    if not compiler:
        compiler_alt = Path.home() / ".nimble/bin/nim"
        if compiler_alt.exists():
            compiler = str(compiler_alt)
        else:
            return None

    if name not in _nim_bin_cache:
        BUILD_DIR.mkdir(exist_ok=True)
        src = (
            "import std/algorithm\n"
            "import std/monotimes\n"
            "import std/times\n"
            "import std/strutils\n\n"
            "proc maximumOddBinary(s: var string) =\n"
            f"  {func_body}\n\n"
            "proc main() =\n"
            f'  let base = repeat("01", {INPUT_LEN // 2})\n'
            f"  let n = {N_ITERS}\n"
            "  var result = base\n"
            "  let start = getMonoTime()\n"
            "  for i in 0 ..< n:\n"
            "    result = base\n"
            "    maximumOddBinary(result)\n"
            "  let elapsed = (getMonoTime() - start).inNanoseconds.float / 1e9\n"
            "  echo elapsed / n.float\n\n"
            "main()\n"
        )
        src_path = BUILD_DIR / f"bench_{name}.nim"
        bin_path = BUILD_DIR / f"bench_{name}_nim"
        src_path.write_text(src)
        _, cerr, crc = run_cmd(
            [
                compiler,
                "c",
                "-d:release",
                "--hints:off",
                f"--out:{bin_path}",
                str(src_path),
            ],
            timeout=60,
        )
        if crc != 0:
            _nim_bin_cache[name] = None
            return None
        _nim_bin_cache[name] = str(bin_path)

    bin_path = _nim_bin_cache[name]
    if bin_path is None:
        return None
    out, _, rc = run_cmd([bin_path])
    return parse_number(out) if rc == 0 else None


# ─────────────────────────── Benchmark: Julia ───────────────────


def bench_julia():
    interp = find_cmd("julia")
    if not interp:
        return None
    code = (
        "function maximum_odd_binary(s)\n"
        "    v = sort(s, rev=true)\n"
        "    circshift(v, -1)\n"
        "end\n\n"
        f'input = collect(repeat("01", {INPUT_LEN // 2}))\n'
        "maximum_odd_binary(input)\n"
        f"n = {N_ITERS}\n"
        "t = @elapsed for _ in 1:n\n"
        "    maximum_odd_binary(input)\n"
        "end\n"
        "println(t / n)\n"
    )
    path = write_temp(".jl", code)
    try:
        out, _, rc = run_cmd([interp, "--startup-file=no", path], timeout=60)
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: Julia (bool) ─────────────


def bench_julia_bool():
    interp = find_cmd("julia")
    if not interp:
        return None
    code = (
        "function maximum_odd_binary(s)\n"
        "    v = circshift(sort(collect(s) .== '1', rev=true), -1)\n"
        "    String(map(b -> b ? '1' : '0', v))\n"
        "end\n\n"
        f'input = repeat("01", {INPUT_LEN // 2})\n'
        "maximum_odd_binary(input)\n"
        f"n = {N_ITERS}\n"
        "t = @elapsed for _ in 1:n\n"
        "    maximum_odd_binary(input)\n"
        "end\n"
        "println(t / n)\n"
    )
    path = write_temp(".jl", code)
    try:
        out, _, rc = run_cmd([interp, "--startup-file=no", path], timeout=60)
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: Julia (count) ────────────


def bench_julia_count():
    interp = find_cmd("julia")
    if not interp:
        return None
    code = (
        "function maximum_odd_binary(s)\n"
        "    n = count(==('1'), s)\n"
        "    '1'^(n-1) * '0'^(length(s)-n) * \"1\"\n"
        "end\n\n"
        f'input = repeat("01", {INPUT_LEN // 2})\n'
        "maximum_odd_binary(input)\n"
        f"n = {N_ITERS}\n"
        "t = @elapsed for _ in 1:n\n"
        "    maximum_odd_binary(input)\n"
        "end\n"
        "println(t / n)\n"
    )
    path = write_temp(".jl", code)
    try:
        out, _, rc = run_cmd([interp, "--startup-file=no", path], timeout=60)
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Benchmark: Python ──────────────────


def bench_python_sort():
    interp = find_cmd("python3", "python")
    if not interp:
        return None
    code = (
        "import time\n"
        f'input_str = "01" * {INPUT_LEN // 2}\n'
        "def maximum_odd_binary(s):\n"
        "    s = list(s)\n"
        "    s.sort(reverse=True)\n"
        "    s.append(s.pop(0))\n"
        '    return "".join(s)\n\n'
        "maximum_odd_binary(input_str)\n"
        f"n = {N_ITERS}\n"
        "start = time.perf_counter()\n"
        "for _ in range(n):\n"
        "    maximum_odd_binary(input_str)\n"
        "elapsed = time.perf_counter() - start\n"
        "print(elapsed / n)\n"
    )
    path = write_temp(".py", code)
    try:
        out, _, rc = run_cmd([interp, path], timeout=60)
        return parse_number(out) if rc == 0 else None
    finally:
        os.unlink(path)


# ─────────────────────────── Logo: C++ ──────────────────────────


def _make_cpp_logo():
    cpp_logo = LOGOS_DIR / "cpp.png"
    if cpp_logo.exists():
        return
    try:
        from PIL import Image, ImageDraw, ImageFont

        size = 128
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(
            [4, 4, size - 4, size - 4],
            radius=20,
            fill="#659ad2",
            outline="#ffffff",
            width=3,
        )
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44
            )
        except Exception:
            font = ImageFont.load_default()
        draw.text(
            (size // 2, size // 2),
            "C++",
            fill="white",
            font=font,
            anchor="mm",
        )
        img.save(str(cpp_logo))
    except Exception:
        pass


# ─────────────────────────── Chart ──────────────────────────────


def generate_chart(results, output_path=OUTPUT_PNG):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    _make_cpp_logo()

    BQN_FONT = Path.home() / ".local/share/fonts/BQN386.ttf"
    from matplotlib import font_manager as fm

    label_fp = None
    if BQN_FONT.exists():
        fm.fontManager.addfont(str(BQN_FONT))
        label_fp = FontProperties(fname=str(BQN_FONT))

    results.sort(key=lambda x: x["time"])
    n = len(results)
    times_us = [r["time"] * 1e6 for r in results]
    max_t = max(times_us) if times_us else 1

    BG = "#0d1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    TEXT = "#e6edf3"
    SUBTEXT = "#8b949e"

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": SUBTEXT,
            "ytick.color": SUBTEXT,
            "font.size": 11,
        }
    )

    fig_h = max(4.5, 0.85 * n + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.subplots_adjust(left=0.36, right=0.88, top=0.86, bottom=0.10)

    y = np.arange(n)
    bars = ax.barh(
        y,
        times_us,
        height=0.55,
        color=[r["color"] for r in results],
        edgecolor="none",
        alpha=0.92,
        zorder=3,
    )

    for i, (bar, t_us) in enumerate(zip(bars, times_us)):
        if t_us < 1:
            lbl = f"{t_us * 1000:.0f} ns"
        elif t_us < 1000:
            lbl = f"{t_us:.1f} \u03bcs"
        else:
            lbl = f"{t_us / 1000:.2f} ms"
        ax.text(
            bar.get_width() + max_t * 0.015,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            ha="left",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )

    labels = []
    for r in results:
        labels.append(f"{r['name']}  {r['code']}")
    ax.set_yticks(y)
    tick_kw = {}
    if label_fp:
        tick_kw["fontproperties"] = label_fp
    ax.set_yticklabels(labels, fontsize=11, **tick_kw)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=45)

    for i, r in enumerate(results):
        logo_file = LOGOS_DIR / f"{r.get('logo', '')}.png"
        if not logo_file.exists():
            continue
        try:
            img = mpimg.imread(str(logo_file))
            zoom = 22 / img.shape[0]
            oi = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(
                oi,
                (0, i),
                xybox=(-18, 0),
                xycoords=("axes fraction", "data"),
                boxcoords="offset points",
                frameon=False,
                pad=0,
            )
            ax.add_artist(ab)
        except Exception:
            pass

    ax.set_xlabel("Time per call (\u03bcs)", fontsize=12, labelpad=10)
    ax.set_title(
        "Maximum Odd Binary Number \u2014 Benchmark\n"
        f'Input: {INPUT_LEN}-char "010101\u2026"  \u00b7  '
        f"{N_ITERS:,} iters  \u00b7  median of {N_RUNS} runs",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#ffffff",
    )
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=GRID, zorder=0)
    ax.set_xlim(0, max_t * 1.22)

    plt.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  Chart saved to {output_path}")


# ─────────────────────────── Syntax Highlighting ────────────────


def _shiki_highlight(snippets):
    """Highlight code snippets with Shiki (VS Code engine). Falls back to Pygments."""
    import json as _json

    node = find_cmd("node")
    shiki_ok = node and (ROOT / "node_modules" / "shiki").is_dir()

    if shiki_ok:
        input_data = [{"code": code, "lang": lang} for code, lang in snippets]
        in_path = BUILD_DIR / "_shiki_in.json"
        out_path = BUILD_DIR / "_shiki_out.json"
        BUILD_DIR.mkdir(exist_ok=True)
        in_path.write_text(_json.dumps(input_data, ensure_ascii=False))

        script = (
            "import{codeToHtml}from'shiki';"
            "import{readFileSync,writeFileSync}from'fs';"
            f"const d=JSON.parse(readFileSync('{in_path}','utf8'));"
            "const r=[];"
            "for(const{code,lang}of d)"
            "  r.push(await codeToHtml(code,{lang,theme:'dracula'}));"
            f"writeFileSync('{out_path}',JSON.stringify(r));"
        )
        _, err, rc = run_cmd([node, "--input-type=module", "-e", script], timeout=30)
        if rc == 0 and out_path.exists():
            results = _json.loads(out_path.read_text())
            results = [re.sub(r"font-style:\s*italic;?\s*", "", h) for h in results]
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            print("  Syntax highlighting: Shiki (VS Code engine)")
            return results

    try:
        from pygments import highlight as pyg_highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter

        fmt = HtmlFormatter(style="dracula", noclasses=True, nowrap=False)
        results = []
        for code, lang in snippets:
            html = pyg_highlight(code, get_lexer_by_name(lang), fmt)
            html = re.sub(r"font-style:\s*italic;?\s*", "", html)
            results.append(html)
        print("  Syntax highlighting: Pygments (fallback)")
        return results
    except Exception:
        return [""] * len(snippets)


# ─────────────────────────── Interactive HTML ───────────────────


def generate_html(all_results, output_path):
    """Create a self-contained interactive HTML benchmark page with multi-size support."""
    import base64
    import json

    def _logo_b64(logo_key):
        for ext in (".png", ".svg"):
            p = LOGOS_DIR / f"{logo_key}{ext}"
            if p.exists():
                data = p.read_bytes()
                mime = "image/png" if ext == ".png" else "image/svg+xml"
                return f"data:{mime};base64,{base64.b64encode(data).decode()}"
        return ""

    sizes = sorted(all_results.keys())
    first_results = all_results[sizes[0]]

    HIGHLIGHT_LANGS = {
        "C++": "cpp",
        "Rust": "rust",
        "Nim": "nim",
        "D": "d",
        "Python": "python",
        "Julia": "julia",
    }

    sol_map = {}
    for size in sizes:
        for r in all_results[size]:
            key = r["name"] + "\x00" + r["code"]
            if key not in sol_map:
                sol_map[key] = dict(
                    name=r["name"],
                    code=r["code"],
                    bytes=r.get("bytes"),
                    color=r["color"],
                    logo=_logo_b64(r.get("logo", "")),
                    script=r.get("script", ""),
                    source_code=r.get("source_code", ""),
                    highlight_lang=HIGHLIGHT_LANGS.get(r["name"], ""),
                    by_size={},
                )
            sol_map[key]["by_size"][str(size)] = dict(
                median_s=r["time"],
                median_display=_fmt_time(r["time"]),
                all_times_us=[round(t * 1e6, 2) for t in r.get("all_times", [])],
            )

    snippets = []
    for sol in sol_map.values():
        hl = sol.get("highlight_lang", "")
        if hl:
            snippets.append((sol, hl))
        else:
            sol["highlighted_html"] = ""

    highlighted = _shiki_highlight([(s["source_code"], lang) for s, lang in snippets])
    for (sol, _), html in zip(snippets, highlighted):
        sol["highlighted_html"] = html or ""

    json_payload = dict(
        sizes=sizes,
        solutions=list(sol_map.values()),
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Max Odd Binary \u2014 Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2"></script>
<link href="https://fonts.cdnfonts.com/css/star-jedi-rounded" rel="stylesheet">
<link href="https://fonts.cdnfonts.com/css/sf-distant-galaxy" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px 32px;
    background: #0d1117; color: #e6edf3;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
  .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 16px; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 0; margin-bottom: 20px; }}
  .tab {{
    padding: 8px 20px; cursor: pointer; font-size: 0.9rem;
    border: 1px solid #30363d; background: transparent; color: #8b949e;
  }}
  .tab:first-child {{ border-radius: 8px 0 0 8px; }}
  .tab:last-child {{ border-radius: 0 8px 8px 0; }}
  .tab.active {{ background: #21262d; color: #e6edf3; border-color: #58a6ff; }}
  .view {{ display: none; }}
  .view.active {{ display: block; }}

  /* Controls row */
  .controls-row {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; flex: 1; }}
  .chip {{
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px 5px 6px; border-radius: 20px;
    border: 2px solid var(--c); background: transparent;
    color: #e6edf3; cursor: pointer; font-size: 0.85rem;
    transition: background 0.15s;
    font-family: 'BQN386 Unicode', 'DejaVu Sans Mono', monospace;
  }}
  .chip:hover {{ background: color-mix(in srgb, var(--c) 25%, transparent); }}
  .chip.active {{ background: color-mix(in srgb, var(--c) 40%, transparent); }}
  .chip img {{ width: 22px; height: 22px; border-radius: 4px; }}
  .chip .check {{ width: 14px; text-align: center; font-size: 0.75rem; }}
  select {{
    padding: 6px 12px; border-radius: 8px; font-size: 0.9rem;
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
  }}

  .chart-wrap {{
    position: relative; width: 100%;
    background: #161b22; border-radius: 12px; padding: 20px;
    margin-bottom: 24px;
  }}
  table {{
    width: 100%; border-collapse: collapse;
    font-size: 0.85rem; background: #161b22; border-radius: 12px;
    overflow: hidden;
  }}
  th {{ text-align: left; padding: 10px 12px; color: #8b949e; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  .lang-cell {{ display: flex; align-items: center; gap: 8px; }}
  .lang-cell img {{ width: 20px; height: 20px; }}
  .code-cell {{ font-family: 'BQN386 Unicode', 'DejaVu Sans Mono', monospace; white-space: pre-wrap; }}
  .time-val {{ font-weight: 600; }}
  .runs {{ color: #8b949e; font-size: 0.8rem; font-family: monospace; }}
  .script-cell {{
    font-family: 'BQN386 Unicode', 'DejaVu Sans Mono', monospace;
    font-size: 0.78rem; white-space: pre-wrap; color: #8b949e;
    max-width: 500px;
  }}
  .bar-swatch {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 6px;
  }}
  .legend-row {{
    display: flex; align-items: center; gap: 6px; margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .legend-mode {{
    padding: 4px 14px; border-radius: 6px; cursor: pointer;
    font-size: 0.8rem; border: 1px solid #30363d; background: transparent;
    color: #8b949e; transition: all 0.15s;
  }}
  .legend-mode.active {{ background: #21262d; color: #e6edf3; border-color: #58a6ff; }}
  .approach-filters {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
  }}
  .approach-filters .label {{ color: #8b949e; font-size: 0.8rem; margin-right: 4px; }}
  .approach-filter {{
    padding: 5px 16px; border-radius: 20px; cursor: pointer;
    font-size: 0.8rem; border: 2px solid var(--ab);
    background: color-mix(in srgb, var(--ab) 35%, transparent);
    color: #e6edf3; transition: all 0.15s;
  }}
  .approach-filter:not(.active) {{
    background: transparent; opacity: 0.5;
  }}
  .legend-tag {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 600; margin: 0 2px;
  }}

  /* Title slide */
  #title-slide {{
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 10000; background: #000;
    display: none; flex-direction: column;
    align-items: center; justify-content: center;
    cursor: pointer; overflow: hidden;
  }}
  #title-slide.active {{ display: flex; }}
  #title-slide .top-logos {{
    display: flex; align-items: center; justify-content: center;
    gap: clamp(24px, 4vw, 60px); margin-bottom: clamp(30px, 5vh, 60px);
  }}
  #title-slide .top-logos .vs-label {{
    font-family: 'Star Jedi', 'SF Distant Galaxy', sans-serif;
    font-size: clamp(24px, 3.5vw, 56px);
    color: #fff;
  }}
  #title-slide .top-logos .big-logo {{
    width: clamp(120px, 18vw, 280px); height: clamp(120px, 18vw, 280px);
    object-fit: contain;
  }}
  #title-slide .array-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: clamp(10px, 1.5vw, 24px);
  }}
  #title-slide .array-grid img {{
    width: clamp(80px, 10vw, 160px); height: clamp(80px, 10vw, 160px);
    object-fit: contain;
  }}
  #title-slide .title-main {{
    font-family: 'SF Distant Galaxy', 'Star Jedi', 'SF Distant Galaxy', sans-serif;
    font-size: clamp(80px, 18vw, 300px);
    color: #FFE81F;
    letter-spacing: 0.06em;
    line-height: 1;
    white-space: nowrap;
  }}
  #title-slide .title-sub {{
    font-family: 'Arial', 'Helvetica Neue', sans-serif;
    font-size: clamp(14px, 2.5vw, 36px);
    font-weight: 400;
    color: #58a6ff;
    letter-spacing: 0.5em;
    text-transform: uppercase;
    margin-top: 20px; text-align: center;
  }}
  #title-slide .bottom-logos {{
    display: flex; align-items: center; justify-content: center;
    gap: clamp(30px, 5vw, 80px); margin-top: clamp(30px, 5vh, 60px);
  }}
  #title-slide .bottom-logos img {{
    width: clamp(70px, 9vw, 140px); height: clamp(70px, 9vw, 140px);
    object-fit: contain;
  }}

  /* Slideshow */
  #slide-container {{
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 10000; background: #000;
    display: none;
  }}
  #slide-container.active {{ display: block; }}
  .slide {{
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    display: none; flex-direction: column;
    align-items: center; justify-content: center;
    background: #000; overflow: hidden;
  }}
  .slide.active {{ display: flex; }}
  .slide-section {{
    text-align: center; color: #fff;
  }}
  .slide-section .sec-top {{
    font-family: 'Star Jedi', 'SF Distant Galaxy', sans-serif;
    font-size: clamp(16px, 2.5vw, 40px);
    letter-spacing: 0.2em;
    margin-bottom: 8px;
  }}
  .slide-section .sec-bar {{
    width: min(70vw, 800px); height: 2px;
    background: #fff; margin: 12px auto;
  }}
  .slide-section .sec-part {{
    font-family: 'Cinzel', 'Times New Roman', serif;
    font-size: clamp(50px, 10vw, 160px);
    letter-spacing: 0.15em; line-height: 1.1;
    font-weight: 400;
  }}
  .slide-section .sec-subtitle {{
    font-family: 'Cinzel', 'Times New Roman', serif;
    font-size: clamp(20px, 3.5vw, 56px);
    letter-spacing: 0.2em; text-transform: uppercase;
    font-weight: 400;
  }}
  .slide-text {{
    font-family: 'Arial', 'Helvetica Neue', sans-serif;
    font-size: clamp(36px, 6vw, 100px);
    color: #e6edf3;
    text-align: center; letter-spacing: 0.05em;
    font-weight: 300;
  }}
  .slide-logos {{
    display: flex; align-items: center; justify-content: center;
    gap: clamp(30px, 5vw, 80px);
  }}
  .slide-logos img {{
    width: clamp(100px, 20vw, 320px); height: clamp(100px, 20vw, 320px);
    object-fit: contain;
  }}
  .slide-logos.solo img {{
    width: clamp(200px, 40vw, 640px); height: clamp(200px, 40vw, 640px);
    object-fit: contain;
  }}
  .slide-fullimg img {{
    width: 100vw; height: 100vh; object-fit: contain;
  }}
  .slide-code {{
    background: #282a36;
  }}
  .slide-code .pres-logo {{
    position: absolute; top: 48px; left: 48px;
    max-width: min(14vw, 260px); max-height: min(14vw, 260px);
    min-width: 140px; object-fit: contain;
  }}
  .pres-logo-wrap {{
    position: absolute; top: 48px; left: 48px;
    display: inline-block;
  }}
  .pres-logo-wrap .pres-logo {{
    position: static; display: block;
  }}
  .pres-badge {{
    position: absolute; bottom: -16px; right: -16px;
    width: 96px; height: 96px;
    border-radius: 50%; border: none;
    object-fit: cover;
  }}
  .slide-code .pres-code-wrap {{
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; padding: 60px 6vw 60px 12vw;
  }}
  .slide-code pre {{
    margin: 0; padding: 0; background: transparent !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: clamp(28px, 3vw, 52px); line-height: 1.75;
    font-style: normal !important; font-feature-settings: 'liga' 0, 'calt' 0;
  }}
  .slide-code code {{ font-family: inherit; }}

  /* Star Wars wipe transitions */
  .slide.wipe-reveal {{ z-index: 10002; }}

  .slide.wipe-down {{
    -webkit-mask-image: linear-gradient(to bottom, #000 46%, transparent 54%);
    mask-image: linear-gradient(to bottom, #000 46%, transparent 54%);
    -webkit-mask-size: 100% 300vh; mask-size: 100% 300vh;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    animation: wipeDown 1800ms ease-in-out forwards;
  }}
  @keyframes wipeDown {{
    from {{ -webkit-mask-position: 0 -200vh; mask-position: 0 -200vh; }}
    to   {{ -webkit-mask-position: 0 0; mask-position: 0 0; }}
  }}

  .slide.wipe-up {{
    -webkit-mask-image: linear-gradient(to bottom, transparent 46%, #000 54%);
    mask-image: linear-gradient(to bottom, transparent 46%, #000 54%);
    -webkit-mask-size: 100% 300vh; mask-size: 100% 300vh;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    animation: wipeUp 1800ms ease-in-out forwards;
  }}
  @keyframes wipeUp {{
    from {{ -webkit-mask-position: 0 0; mask-position: 0 0; }}
    to   {{ -webkit-mask-position: 0 -200vh; mask-position: 0 -200vh; }}
  }}

  .slide.wipe-rtl {{
    -webkit-mask-image: linear-gradient(to right, transparent 46%, #000 54%);
    mask-image: linear-gradient(to right, transparent 46%, #000 54%);
    -webkit-mask-size: 300vw 100%; mask-size: 300vw 100%;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    animation: wipeRtl 1800ms ease-in-out forwards;
  }}
  @keyframes wipeRtl {{
    from {{ -webkit-mask-position: 0 0; mask-position: 0 0; }}
    to   {{ -webkit-mask-position: -200vw 0; mask-position: -200vw 0; }}
  }}

  /* Magic Move token animation */
  #magic-move-overlay {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    z-index: 10001; pointer-events: none;
    background: #282a36; display: none;
  }}
  #magic-move-overlay span {{
    position: absolute; white-space: pre;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'DejaVu Sans Mono', monospace;
    font-feature-settings: 'liga' 0, 'calt' 0;
  }}

  /* Presentation overlay */
  #presentation-overlay {{
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 9999; background: #282a36;
    display: none; cursor: pointer; overflow: auto;
  }}
  #presentation-overlay.active {{ display: block; }}
  #presentation-overlay .pres-logo {{
    position: absolute; top: 48px; left: 48px;
    max-width: min(14vw, 260px); max-height: min(14vw, 260px);
    min-width: 140px;
    object-fit: contain;
  }}
  #presentation-overlay .pres-code-wrap {{
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; padding: 60px 6vw 60px 12vw;
  }}
  #presentation-overlay pre {{
    margin: 0; padding: 0; background: transparent !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: clamp(28px, 3vw, 52px); line-height: 1.75;
    font-style: normal !important; font-feature-settings: 'liga' 0, 'calt' 0;
  }}
  #presentation-overlay code {{ font-family: inherit; }}
</style>
</head>
<body>

<h1>Maximum Odd Binary Number \u2014 Benchmark</h1>
<div class="subtitle">
  {N_ITERS:,} iterations &middot; median of {N_RUNS} runs &middot;
  sizes: {", ".join(f"{s:,}" for s in sizes)} chars
</div>

<div class="tabs">
  <button class="tab active" onclick="switchTab('detail')">Detail</button>
  <button class="tab" onclick="switchTab('summary')">Summary</button>
</div>

<div class="controls-row">
  <div class="controls" id="controls"></div>
  <select id="sizeSelect" onchange="onSizeChange()">
    {chr(10).join(f'    <option value="{s}"' + (" selected" if s == 1000 else "") + f">{s:,} chars</option>" for s in sizes)}
  </select>
</div>

<div class="legend-row">
  <button class="legend-mode active" onclick="setColorMode('language')">By Language</button>
  <button class="legend-mode" onclick="setColorMode('approach')">By Approach</button>
  <span id="legend-tags"></span>
</div>

<div class="approach-filters">
  <span class="label">Filter:</span>
  <button class="approach-filter active" style="--ab:#58a6ff" onclick="toggleApproachFilter('sort', this)">Sort + Rotate</button>
  <button class="approach-filter active" style="--ab:#3fb950" onclick="toggleApproachFilter('partition', this)">Partition</button>
  <button class="approach-filter active" style="--ab:#f85149" onclick="toggleApproachFilter('count', this)">Count + Construct</button>
</div>

<!-- Detail view -->
<div class="view active" id="view-detail">
  <div class="chart-wrap">
    <canvas id="barChart"></canvas>
    <button id="resetZoom" style="display:none; position:absolute; top:8px; right:12px;
      padding:4px 12px; border-radius:6px; border:1px solid #30363d;
      background:#21262d; color:#e6edf3; cursor:pointer; font-size:0.8rem;"
      onclick="barChart.resetZoom(); this.style.display='none';">Reset zoom</button>
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Language</th><th>Solution</th><th>Code</th>
        <th>Median</th><th>Script</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<!-- Summary view -->
<div class="view" id="view-summary">
  <div class="chart-wrap" style="height:500px">
    <canvas id="lineChart"></canvas>
  </div>
</div>

<!-- Slideshow -->
<div id="slide-container">
  <div id="magic-move-overlay"></div>
  <!-- 0: Title -->
  <div class="slide" id="title-slide" data-slide="0">
    <div class="top-logos" id="top-logos"></div>
    <div class="title-main">PERF WARS</div>
    <div class="title-sub">Episode I: Maximum Odd Binary</div>
    <div class="bottom-logos" id="bottom-logos"></div>
  </div>
  <!-- 1: Problem description image -->
  <div class="slide slide-fullimg" data-slide="1" data-wipe="down">
    <img src="img/problem.png" alt="Max Odd Binary problem">
  </div>
  <!-- 2: Section (Episode I style) -->
  <div class="slide" data-slide="2" data-wipe="clock">
    <div class="slide-section">
      <div class="sec-top">PERF WARS</div>
      <div class="sec-bar"></div>
      <div class="sec-part">PART I</div>
      <div class="sec-bar"></div>
      <div class="sec-subtitle">Sort + Rotate</div>
    </div>
  </div>
  <div class="slide" data-slide="3" data-wipe="clock">
    <div class="slide-text">Array Solutions</div>
  </div>
  <div class="slide" data-slide="4">
    <div class="slide-text">Array Comparison</div>
  </div>
  <!-- 5: C++ logo -->
  <div class="slide" data-slide="5">
    <div class="slide-logos solo" id="slide-logos-cpp"></div>
  </div>
  <!-- 6: C++ code -->
  <div class="slide slide-code" data-slide="6" id="slide-code-cpp-sort"></div>
  <!-- 7: Rust/D/Julia/Nim/Python logos -->
  <div class="slide" data-slide="7">
    <div class="slide-logos" id="slide-logos-others"></div>
  </div>
  <!-- 8-12: Code slides -->
  <div class="slide slide-code" data-slide="8" id="slide-code-rust-sort"></div>
  <div class="slide slide-code" data-slide="9" id="slide-code-julia-sort"></div>
  <div class="slide slide-code" data-slide="10" id="slide-code-nim-sort"></div>
  <div class="slide slide-code" data-slide="11" id="slide-code-python-sort"></div>
  <div class="slide slide-code" data-slide="12" id="slide-code-d-sort"></div>
  <!-- Part II -->
  <div class="slide" data-slide="13" data-wipe="down">
    <div class="slide-section">
      <div class="sec-top">PERF WARS</div>
      <div class="sec-bar"></div>
      <div class="sec-part">PART II</div>
      <div class="sec-bar"></div>
      <div class="sec-subtitle">Partition + Rotate</div>
    </div>
  </div>
  <div class="slide slide-code" data-slide="14" data-wipe="down" id="slide-code-cpp-part"></div>
  <div class="slide slide-code" data-slide="15" id="slide-code-rust-part"></div>
  <div class="slide slide-code" data-slide="16" id="slide-code-d-part"></div>
  <!-- Part III -->
  <div class="slide" data-slide="17" data-wipe="up">
    <div class="slide-section">
      <div class="sec-top">PERF WARS</div>
      <div class="sec-bar"></div>
      <div class="sec-part">PART III</div>
      <div class="sec-bar"></div>
      <div class="sec-subtitle">Count + Construct</div>
    </div>
  </div>
  <div class="slide" data-slide="18" data-wipe="up">
    <div class="slide-text">Array Solutions</div>
  </div>
  <div class="slide slide-code" data-slide="19" id="slide-code-cpp-count"></div>
  <div class="slide slide-code" data-slide="20" id="slide-code-rust-count"></div>
  <div class="slide slide-code" data-slide="21" id="slide-code-d-count"></div>
  <!-- Part IV -->
  <div class="slide" data-slide="22" data-wipe="rtl">
    <div class="slide-section">
      <div class="sec-top">PERF WARS</div>
      <div class="sec-bar"></div>
      <div class="sec-part">PART IV</div>
      <div class="sec-bar"></div>
      <div class="sec-subtitle">Twitter++</div>
    </div>
  </div>
  <div class="slide slide-code" data-slide="23" data-wipe="rtl" id="slide-code-cpp-part-iv"></div>
  <div class="slide slide-code" data-slide="24" id="slide-code-cpp-partpoint"></div>
  <div class="slide slide-code" data-slide="25" id="slide-code-cpp-partonly"></div>
  <div class="slide slide-code" data-slide="26" id="slide-code-cpp-cc2"></div>
  <div class="slide slide-code" data-slide="27" id="slide-code-cpp-cc3"></div>
  <!-- Part V -->
  <div class="slide" data-slide="28" data-wipe="clock">
    <div class="slide-section">
      <div class="sec-top">PERF WARS</div>
      <div class="sec-bar"></div>
      <div class="sec-part">PART V</div>
      <div class="sec-bar"></div>
      <div class="sec-subtitle">Performance</div>
    </div>
  </div>
</div>

<!-- Presentation overlay (bar click) -->
<div id="presentation-overlay" onclick="closePresentation()">
  <img class="pres-logo" id="pres-logo" src="" alt="">
  <div class="pres-code-wrap" id="pres-code-wrap"></div>
</div>

<script>
const BENCH = {json.dumps(json_payload, ensure_ascii=False)};
const SIZES = BENCH.sizes;
const SOLS  = BENCH.solutions;
const enabled = new Set(SOLS.map((_, i) => i));
const enabledApproaches = new Set(['sort', 'partition', 'count']);
let currentSize = SIZES.includes(1000) ? '1000' : String(SIZES[0]);
let barChart, lineChart;

function fmtTime(s) {{
  const us = s * 1e6;
  if (us < 1) return (us * 1000).toFixed(0) + ' ns';
  if (us < 1000) return us.toFixed(1) + ' \\u03bcs';
  return (us / 1000).toFixed(2) + ' ms';
}}

function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${{name}}"]`).classList.add('active');
  document.getElementById('view-' + name).classList.add('active');
  document.getElementById('sizeSelect').style.display = name === 'detail' ? '' : 'none';
  if (name === 'summary') updateLineChart();
}}

function onSizeChange() {{
  currentSize = document.getElementById('sizeSelect').value;
  updateBarChart();
  updateTable();
}}

// ── Controls ──
const enabledLangs = new Set();
function buildControls() {{
  const el = document.getElementById('controls');
  const langs = [];
  const seen = new Set();
  SOLS.forEach((d, i) => {{
    if (!seen.has(d.name)) {{
      seen.add(d.name);
      langs.push({{ name: d.name, color: d.color, logo: d.logo, indices: [] }});
    }}
    langs.find(l => l.name === d.name).indices.push(i);
  }});
  langs.forEach(lang => {{
    enabledLangs.add(lang.name);
    const chip = document.createElement('button');
    chip.className = 'chip active';
    chip.style.setProperty('--c', lang.color);
    chip.innerHTML =
      (lang.logo ? `<img src="${{lang.logo}}" alt="">` : '') +
      `<span class="check">\\u2713</span>` +
      `<span>${{lang.name}}</span>`;
    chip.onclick = () => {{
      const on = enabledLangs.has(lang.name);
      if (on) {{
        enabledLangs.delete(lang.name);
        lang.indices.forEach(i => enabled.delete(i));
        chip.classList.remove('active');
      }} else {{
        enabledLangs.add(lang.name);
        lang.indices.forEach(i => enabled.add(i));
        chip.classList.add('active');
      }}
      chip.querySelector('.check').textContent = enabledLangs.has(lang.name) ? '\\u2713' : '';
      updateBarChart(); updateTable(); updateLineChart();
    }};
    el.appendChild(chip);
  }});
}}

// ── Bar chart (detail view) ──
function buildBarChart() {{
  barChart = new Chart(document.getElementById('barChart'), {{
    type: 'bar',
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderWidth: 0 }}] }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      onClick: (event, elements) => {{
        if (elements.length > 0) {{
          const d = barChart._visibleData[elements[0].index];
          if (d && d.highlight_lang) openPresentation(d);
        }}
      }},
      onHover: (event, elements) => {{
        const target = event.native ? event.native.target : event.chart.canvas;
        if (elements.length > 0) {{
          const d = barChart._visibleData[elements[0].index];
          target.style.cursor = (d && d.highlight_lang) ? 'pointer' : 'default';
        }} else {{
          target.style.cursor = 'default';
        }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          displayColors: false,
          titleFont: {{ family: "'BQN386 Unicode','DejaVu Sans Mono',monospace", size: 18 }},
          bodyFont: {{ family: "'BQN386 Unicode','DejaVu Sans Mono',monospace", size: 16 }},
          padding: 16,
          maxWidth: 600,
          callbacks: {{
            title: ctx => {{
              const d = barChart._visibleData[ctx[0].dataIndex];
              return d ? `${{d.name}}  ${{d.code}}` : '';
            }},
            beforeBody: ctx => {{
              const d = barChart._visibleData[ctx[0].dataIndex];
              return d && d.source_code ? '\\n' + d.source_code : '';
            }},
            label: ctx => '\\n' + fmtTime(ctx.raw),
          }}
        }},
        zoom: {{
          zoom: {{
            drag: {{
              enabled: true,
              backgroundColor: 'rgba(88,166,255,0.15)',
              borderColor: 'rgba(88,166,255,0.6)',
              borderWidth: 1,
            }},
            mode: 'y',
            onZoom: ({{chart}}) => {{
              document.getElementById('resetZoom').style.display = '';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Time per call', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
        }},
        y: {{
          ticks: {{ color: '#e6edf3', font: {{ family: "'BQN386 Unicode','DejaVu Sans Mono',monospace", size: 12 }} }},
          grid: {{ display: false }},
        }}
      }}
    }}
  }});
  updateBarChart();
}}

function updateBarChart() {{
  const vis = SOLS
    .map((d, i) => ({{ ...d, idx: i }}))
    .filter(d => enabled.has(d.idx) && d.by_size[currentSize] && enabledApproaches.has(getApproach(d)))
    .sort((a, b) => a.by_size[currentSize].median_s - b.by_size[currentSize].median_s);
  const labels = vis.map(d => {{
    return `${{d.name}}  ${{d.code}}`;
  }});
  const values = vis.map(d => d.by_size[currentSize].median_s * 1e6);
  const colors = vis.map(d => getColor(d));
  barChart._visibleData = vis;
  barChart.data.labels = labels;
  barChart.data.datasets[0].data = values;
  barChart.data.datasets[0].backgroundColor = colors;
  barChart.canvas.parentElement.style.height = Math.max(200, vis.length * 52 + 60) + 'px';
  barChart.resize();
  barChart.resetZoom();
  barChart.update();
  document.getElementById('resetZoom').style.display = 'none';
}}

// ── Table (detail view) ──
function updateTable() {{
  const tbody = document.getElementById('tbody');
  const vis = SOLS
    .map((d, i) => ({{ ...d, idx: i }}))
    .filter(d => enabled.has(d.idx) && d.by_size[currentSize] && enabledApproaches.has(getApproach(d)))
    .sort((a, b) => a.by_size[currentSize].median_s - b.by_size[currentSize].median_s);
  tbody.innerHTML = '';
  vis.forEach((d, rank) => {{
    const sd = d.by_size[currentSize];
    const tr = document.createElement('tr');
    const scriptHtml = d.script
      ? d.script.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      : '';
    const rawCode = d.source_code || d.code || '';
    const codeHtml = rawCode.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    tr.innerHTML = `
      <td>${{rank + 1}}</td>
      <td><div class="lang-cell">
        <span class="bar-swatch" style="background:${{getColor(d)}}"></span>
        ${{d.logo ? `<img src="${{d.logo}}" alt="">` : ''}}
        ${{d.name}}
      </div></td>
      <td class="code-cell">${{d.code}}</td>
      <td class="code-cell">${{codeHtml}}</td>
      <td class="time-val">${{sd.median_display}}</td>
      <td class="script-cell">${{scriptHtml}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

// ── Line chart (summary view) ──
function buildLineChart() {{
  lineChart = new Chart(document.getElementById('lineChart'), {{
    type: 'line',
    data: {{ labels: SIZES.map(s => s.toLocaleString()), datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: true, labels: {{ color: '#e6edf3', usePointStyle: true, padding: 16,
          font: {{ family: "'BQN386 Unicode','DejaVu Sans Mono',monospace" }} }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{ctx.dataset.label}}: ${{fmtTime(ctx.raw)}}`
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Input size (chars)', color: '#8b949e' }},
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
        }},
        y: {{
          type: 'logarithmic',
          title: {{ display: true, text: 'Time per call (seconds, log)', color: '#8b949e' }},
          ticks: {{ color: '#8b949e', callback: (v) => fmtTime(v) }},
          grid: {{ color: '#21262d' }},
        }}
      }}
    }}
  }});
}}

function updateLineChart() {{
  if (!lineChart) return;
  const datasets = [];
  SOLS.forEach((d, i) => {{
    if (!enabled.has(i) || !enabledApproaches.has(getApproach(d))) return;
    const data = SIZES.map(s => {{
      const sd = d.by_size[String(s)];
      return sd ? sd.median_s : null;
    }});
    datasets.push({{
      label: `${{d.name}} ${{d.code}}`,
      data,
      borderColor: getColor(d),
      backgroundColor: getColor(d) + '33',
      pointBackgroundColor: getColor(d),
      pointRadius: 5,
      borderWidth: 2.5,
      tension: 0.2,
    }});
  }});
  lineChart.data.datasets = datasets;
  lineChart.update();
}}

// ── Color modes ──
const APPROACH_COLORS = {{
  sort:      '#58a6ff',
  partition: '#3fb950',
  count:     '#f85149',
}};

const APPROACH_LABELS = [
  {{ key: 'sort',      label: 'Sort',      bg: '#58a6ff' }},
  {{ key: 'partition',  label: 'Partition',  bg: '#3fb950' }},
  {{ key: 'count',     label: 'Count',     bg: '#f85149' }},
];

function getApproach(d) {{
  const c = d.code.toLowerCase();
  if (c.includes('count') || c.includes('tacit count') || c.includes('construct')) return 'count';
  if (c.includes('partition')) return 'partition';
  if (c.includes('sort') || c.includes('circshift') || c.includes('rotate')
      || c === '1\u233d\u2228' || c === '\u21bb1\u21cc\u2346' || c === '1|.\\\\:~'
      || c === '1\u233d\u2282\u2364\u2352\u235b\u2337'
      || c.startsWith('1\u2218\u2296') || c.startsWith('1\u00ab\u2296')
      || c.startsWith('\u2985')) return 'sort';
  return 'other';
}}

let colorMode = 'language';

function getColor(d) {{
  return colorMode === 'language' ? d.color : (APPROACH_COLORS[getApproach(d)] || APPROACH_COLORS.other);
}}

function setColorMode(mode) {{
  colorMode = mode;
  document.querySelectorAll('.legend-mode').forEach(b => b.classList.toggle('active', b.textContent.toLowerCase().includes(mode)));
  const tagsEl = document.getElementById('legend-tags');
  if (mode === 'approach') {{
    tagsEl.innerHTML = APPROACH_LABELS.map(a =>
      `<span class="legend-tag" style="background:${{a.bg}};color:#0d1117">${{a.label}}</span>`
    ).join('');
  }} else {{
    tagsEl.innerHTML = '';
  }}
  updateBarChart();
  updateTable();
  updateLineChart();
}}

function toggleApproachFilter(approach, btn) {{
  if (enabledApproaches.has(approach)) {{
    enabledApproaches.delete(approach);
    btn.classList.remove('active');
  }} else {{
    enabledApproaches.add(approach);
    btn.classList.add('active');
  }}
  updateBarChart();
  updateTable();
  updateLineChart();
}}

// ── Presentation overlay ──
function openPresentation(d) {{
  const overlay = document.getElementById('presentation-overlay');
  document.getElementById('pres-logo').src = d.logo;
  document.getElementById('pres-code-wrap').innerHTML = d.highlighted_html;
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
}}

function closePresentation() {{
  document.getElementById('presentation-overlay').classList.remove('active');
  document.body.style.overflow = '';
}}

// ── Slideshow engine ──
let slideIndex = 0;
let slideshowActive = false;
const slideEls = document.querySelectorAll('#slide-container .slide');
const totalSlides = slideEls.length;

function getTokens(slideEl) {{
  const spans = slideEl.querySelectorAll('.pres-code-wrap code span.line span[style]');
  const tokens = [];
  spans.forEach(el => {{
    const text = el.textContent;
    if (!text.trim()) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0)
      tokens.push({{ text, style: el.getAttribute('style'), x: r.left, y: r.top, w: r.width, h: r.height }});
  }});
  return tokens;
}}

function magicMove(oldSlide, newSlide, duration) {{
  const oldTokens = getTokens(oldSlide);

  const oldLogo = oldSlide.querySelector('.pres-logo');
  let oldLogoRect, oldLogoSrc;
  if (oldLogo) {{
    oldLogoRect = oldLogo.getBoundingClientRect();
    oldLogoSrc = oldLogo.src;
  }}

  const overlay = document.getElementById('magic-move-overlay');
  overlay.innerHTML = '';
  overlay.style.display = 'block';
  overlay.style.opacity = '1';
  overlay.style.transition = '';

  slideEls.forEach((el, i) => el.classList.toggle('active', el === newSlide));

  if (oldLogoSrc) {{
    const newLogo = newSlide.querySelector('.pres-logo');
    const newLogoRect = newLogo ? newLogo.getBoundingClientRect() : oldLogoRect;

    const oldImg = document.createElement('img');
    oldImg.src = oldLogoSrc;
    oldImg.style.cssText = `position:absolute;left:${{oldLogoRect.left}}px;top:${{oldLogoRect.top}}px;width:${{oldLogoRect.width}}px;height:${{oldLogoRect.height}}px;object-fit:contain;z-index:2;transition:opacity ${{duration}}ms ease;`;
    overlay.appendChild(oldImg);

    if (newLogo) {{
      const newImg = document.createElement('img');
      newImg.src = newLogo.src;
      newImg.style.cssText = `position:absolute;left:${{newLogoRect.left}}px;top:${{newLogoRect.top}}px;width:${{newLogoRect.width}}px;height:${{newLogoRect.height}}px;object-fit:contain;z-index:1;opacity:0;transition:opacity ${{duration}}ms ease;`;
      overlay.appendChild(newImg);
      requestAnimationFrame(() => {{
        oldImg.style.opacity = '0';
        newImg.style.opacity = '1';
      }});
    }}
  }}

  const newTokens = getTokens(newSlide);

  const pre = newSlide.querySelector('pre') || newSlide;
  const cs = getComputedStyle(pre);

  const usedOld = new Set();
  const matchMap = new Map();
  newTokens.forEach((nt, ni) => {{
    for (let i = 0; i < oldTokens.length; i++) {{
      if (!usedOld.has(i) && oldTokens[i].text === nt.text) {{
        usedOld.add(i);
        matchMap.set(ni, oldTokens[i]);
        return;
      }}
    }}
  }});

  function mkSpan(text, style, x, y, w, h) {{
    const span = document.createElement('span');
    span.textContent = text;
    span.setAttribute('style', style);
    span.style.fontSize = cs.fontSize;
    span.style.lineHeight = h + 'px';
    span.style.height = h + 'px';
    span.style.width = w + 'px';
    span.style.letterSpacing = cs.letterSpacing;
    span.style.left = x + 'px';
    span.style.top = y + 'px';
    span.style.overflow = 'hidden';
    return span;
  }}

  newTokens.forEach((nt, ni) => {{
    if (matchMap.has(ni)) {{
      const o = matchMap.get(ni);
      const span = mkSpan(nt.text, nt.style, nt.x, nt.y, nt.w, nt.h);
      const dx = o.x - nt.x;
      const dy = o.y - nt.y;
      span.style.transform = `translate(${{dx}}px, ${{dy}}px)`;
      span.style.transition = `transform ${{duration}}ms cubic-bezier(0.4, 0, 0.2, 1)`;
      overlay.appendChild(span);
      requestAnimationFrame(() => {{ span.style.transform = 'translate(0, 0)'; }});
    }} else {{
      const span = mkSpan(nt.text, nt.style, nt.x, nt.y, nt.w, nt.h);
      span.style.opacity = '0';
      span.style.transition = `opacity ${{duration * 0.5}}ms ease ${{duration * 0.3}}ms`;
      overlay.appendChild(span);
      requestAnimationFrame(() => {{ span.style.opacity = '1'; }});
    }}
  }});

  oldTokens.forEach((o, i) => {{
    if (usedOld.has(i)) return;
    const span = mkSpan(o.text, o.style, o.x, o.y, o.w, o.h);
    span.style.transition = `opacity ${{duration * 0.4}}ms ease`;
    overlay.appendChild(span);
    requestAnimationFrame(() => {{ span.style.opacity = '0'; }});
  }});

  setTimeout(() => {{
    overlay.style.transition = `opacity ${{duration * 0.2}}ms ease`;
    overlay.style.opacity = '0';
  }}, duration * 0.85);

  setTimeout(() => {{
    overlay.innerHTML = '';
    overlay.style.display = '';
    overlay.style.opacity = '';
    overlay.style.transition = '';
  }}, duration + 80);
}}

const MM_DURATION = 900;

let wipeInProgress = false;
const WIPE_DURATION = 1800;
const WIPE_CLASSES = ['wipe-reveal', 'wipe-down', 'wipe-up', 'wipe-rtl'];

function wipeEase(t) {{
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}}

function animateClockWipe(slide, duration, onDone) {{
  const start = performance.now();
  function tick() {{
    const t = Math.min((performance.now() - start) / duration, 1);
    const angle = wipeEase(t) * 380;
    const solid = Math.max(0, angle - 20);
    const edge = Math.min(angle, 360);
    const g = `conic-gradient(from -90deg, #000 ${{solid}}deg, transparent ${{edge}}deg, transparent)`;
    slide.style.maskImage = g;
    slide.style.webkitMaskImage = g;
    if (t < 1) requestAnimationFrame(tick);
    else {{ slide.style.maskImage = ''; slide.style.webkitMaskImage = ''; onDone(); }}
  }}
  requestAnimationFrame(tick);
}}

function showSlide(n) {{
  if (wipeInProgress) return;
  const newIndex = Math.max(0, Math.min(n, totalSlides - 1));
  if (newIndex === slideIndex && slideEls[slideIndex].classList.contains('active')) return;

  const oldSlide = slideEls[slideIndex];
  const newSlide = slideEls[newIndex];
  const bothCode = oldSlide.classList.contains('slide-code') && newSlide.classList.contains('slide-code');
  const wipeType = newSlide.dataset.wipe;

  if (wipeType && slideshowActive) {{
    wipeInProgress = true;
    if (wipeType === 'clock') {{
      const g0 = 'conic-gradient(from -90deg, transparent, transparent)';
      newSlide.style.maskImage = g0;
      newSlide.style.webkitMaskImage = g0;
      newSlide.classList.add('active', 'wipe-reveal');
      animateClockWipe(newSlide, WIPE_DURATION, () => {{
        oldSlide.classList.remove('active');
        newSlide.classList.remove('wipe-reveal');
        wipeInProgress = false;
      }});
    }} else {{
      WIPE_CLASSES.forEach(c => newSlide.classList.remove(c));
      void newSlide.offsetWidth;
      newSlide.classList.add('active', 'wipe-reveal', 'wipe-' + wipeType);
      setTimeout(() => {{
        oldSlide.classList.remove('active');
        WIPE_CLASSES.forEach(c => newSlide.classList.remove(c));
        wipeInProgress = false;
      }}, WIPE_DURATION + 50);
    }}
  }} else if (bothCode && slideshowActive) {{
    magicMove(oldSlide, newSlide, MM_DURATION);
  }} else {{
    slideEls.forEach((el, i) => el.classList.toggle('active', i === newIndex));
  }}
  slideIndex = newIndex;
}}

function enterPresentation(startAt) {{
  slideshowActive = true;
  showSlide(startAt != null ? startAt : 0);
  document.getElementById('slide-container').classList.add('active');
  document.body.style.overflow = 'hidden';
}}

function exitPresentation() {{
  slideshowActive = false;
  document.getElementById('slide-container').classList.remove('active');
  slideEls.forEach(el => el.classList.remove('active'));
  document.body.style.overflow = '';
}}

function togglePresentation() {{
  if (slideshowActive) exitPresentation();
  else enterPresentation(0);
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    if (slideshowActive) {{ exitPresentation(); return; }}
    closePresentation();
  }}
  if (e.ctrlKey && e.key === 'p') {{
    e.preventDefault();
    togglePresentation();
    return;
  }}
  if (!slideshowActive) return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'PageDown') {{
    e.preventDefault();
    showSlide(slideIndex + 1);
  }}
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'PageUp') {{
    e.preventDefault();
    showSlide(slideIndex - 1);
  }}
}});

// ── Build slides ──
function buildSlides() {{
  const byName = {{}};
  const solByKey = {{}};
  SOLS.forEach(d => {{
    if (d.logo && !byName[d.name]) byName[d.name] = d.logo;
    solByKey[d.name + '|' + d.code] = d;
  }});
  const mk = (src, cls) => {{
    const img = document.createElement('img');
    img.src = src;
    if (cls) img.className = cls;
    return img;
  }};

  // Title slide logos
  const topEl = document.getElementById('top-logos');
  const botEl = document.getElementById('bottom-logos');
  const arrayLangs = ['BQN', 'Uiua', 'J', 'APL', 'Kap', 'TinyAPL'];
  const bottomLangs = ['Julia', 'D', 'Nim', 'Python'];

  if (byName['C++']) topEl.appendChild(mk(byName['C++'], 'big-logo'));
  const vs1 = document.createElement('span');
  vs1.className = 'vs-label'; vs1.textContent = 'vs';
  topEl.appendChild(vs1);
  if (byName['Rust']) topEl.appendChild(mk(byName['Rust'], 'big-logo'));
  const vs2 = document.createElement('span');
  vs2.className = 'vs-label'; vs2.textContent = 'vs';
  topEl.appendChild(vs2);
  const grid = document.createElement('div');
  grid.className = 'array-grid';
  arrayLangs.forEach(name => {{
    if (byName[name]) grid.appendChild(mk(byName[name]));
  }});
  topEl.appendChild(grid);
  bottomLangs.forEach(name => {{
    if (byName[name]) botEl.appendChild(mk(byName[name]));
  }});

  // Slide 5: C++ logo
  const cppLogos = document.getElementById('slide-logos-cpp');
  if (byName['C++']) cppLogos.appendChild(mk(byName['C++']));

  // Slide 7: Rust, Julia, Nim, Python logos
  const othLogos = document.getElementById('slide-logos-others');
  ['Rust', 'D', 'Julia', 'Nim', 'Python'].forEach(name => {{
    if (byName[name]) othLogos.appendChild(mk(byName[name]));
  }});

  // Code slides
  const codeSlides = [
    ['slide-code-cpp-sort', 'C++', 'sort+rotate'],
    ['slide-code-rust-sort', 'Rust', 'sort+rotate'],
    ['slide-code-julia-sort', 'Julia', 'sort+circshift'],
    ['slide-code-nim-sort', 'Nim', 'sort+rotate'],
    ['slide-code-python-sort', 'Python', 'sort+rotate'],
    ['slide-code-d-sort', 'D', 'sort+rotate'],
    ['slide-code-cpp-part', 'C++', 'partition+rotate'],
    ['slide-code-rust-part', 'Rust', 'partition+rotate'],
    ['slide-code-d-part', 'D', 'partition+rotate'],
    ['slide-code-cpp-count', 'C++', 'count+construct'],
    ['slide-code-rust-count', 'Rust', 'count+construct'],
    ['slide-code-d-count', 'D', 'count+construct'],
    ['slide-code-cpp-part-iv', 'C++', 'partition+rotate'],
    ['slide-code-cpp-partpoint', 'C++', 'partition+partition_point'],
    ['slide-code-cpp-partonly', 'C++', 'partition'],
    ['slide-code-cpp-cc2', 'C++', 'count+construct 2'],
    ['slide-code-cpp-cc3', 'C++', 'count+construct 3'],
  ];
  codeSlides.forEach(([id, name, code]) => {{
    const sol = solByKey[name + '|' + code];
    if (!sol) return;
    const el = document.getElementById(id);
    const logo = document.createElement('img');
    logo.className = 'pres-logo'; logo.src = sol.logo;
    let badgeSrc = null;
    if (name === 'D') badgeSrc = 'img/cyrus_msk.jpg';
    if (code === 'count+construct 2') badgeSrc = 'img/nekrolm.jpg';
    if (code === 'count+construct 3') badgeSrc = 'img/barryrevzin.jpg';
    if (badgeSrc) {{
      const lw = document.createElement('div');
      lw.className = 'pres-logo-wrap';
      lw.appendChild(logo);
      const badge = document.createElement('img');
      badge.className = 'pres-badge';
      badge.src = badgeSrc;
      lw.appendChild(badge);
      el.appendChild(lw);
    }} else {{
      el.appendChild(logo);
    }}
    const wrap = document.createElement('div');
    wrap.className = 'pres-code-wrap';
    wrap.innerHTML = sol.highlighted_html;
    el.appendChild(wrap);
  }});
}}

buildControls();
buildBarChart();
updateTable();
buildLineChart();
updateLineChart();
setColorMode('language');
buildSlides();
</script>
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"  Interactive page saved to {output_path}")


# ─────────────────────────── Solutions ──────────────────────────

SOLUTIONS = [
    dict(
        name="BQN",
        code="1\u233d\u2228",
        bytes=3,
        color="#2b7067",
        logo="bqn",
        source_code="1\u233d\u2228",
        bench=bench_bqn,
        script=(
            '  \u2022Out \u2022Fmt N(1\u233d\u2228)\u2022_timed L\u294a"01"\n'
            "              ^^^^^^^^^^^^  \u2190 only this is timed\n"
            "  \u2022_timed creates the input once, then runs (1\u233d\u2228) on it N times"
        ),
    ),
    dict(
        name="BQN",
        code="sort+rotate alt",
        bytes=None,
        color="#2b7067",
        logo="bqn",
        source_code='"01"\u228f\u02dc1\u233d\u00b7\u2228\'1\'\u22b8=',
        bench=bench_bqn_sort_alt,
        script=(
            '  \u2022Out \u2022Fmt N("01"\u228f\u02dc1\u233d\u00b7\u2228\'1\'\u22b8=)\u2022_timed L\u294a"01"\n'
            "  sort+rotate on boolean mask, then index back into \"01\""
        ),
    ),
    dict(
        name="BQN",
        code="count+construct",
        bytes=None,
        color="#2b7067",
        logo="bqn",
        source_code="{n\u2190+\u00b4'1'=\U0001d569\n (\"1\"/\u02dcn-1)\u223e(\"0\"/\u02dc(\u2260\U0001d569)-n)\u223e'1'}",
        bench=bench_bqn_count,
        script=(
            "  {n\u2190+\u00b4'1'=\U0001d569 \u22c4 (\"1\"/\u02dcn-1)\u223e(\"0\"/\u02dc(\u2260\U0001d569)-n)\u223e'1'}\n"
            "  O(n) counting approach \u2014 no sort, builds result from char counts"
        ),
    ),
    dict(
        name="BQN",
        code="tacit count",
        bytes=None,
        color="#2b7067",
        logo="bqn",
        source_code="\"101\"/\u02dc(\u2260(1\u223e\u02dc(1-\u02dc\u22a2)\u223e-)(+\u00b4'1'\u22b8=))",
        bench=bench_bqn_tacit,
        script=(
            "  \"101\"/\u02dc(\u2260(1\u223e\u02dc(1-\u02dc\u22a2)\u223e-)(+\u00b4'1'\u22b8=))\n"
            "  O(n) tacit counting \u2014 same approach, point-free style"
        ),
    ),
    dict(
        name="Kap",
        code="1\u233d\u2228",
        bytes=3,
        color="#ffffff",
        logo="kap",
        source_code="1\u233d\u2228",
        bench=bench_kap,
        script=(
            '  input \u2190 L\u2374"01"                          \u2190 not timed\n'
            "  time:measureTime { { 1\u233d\u2228 input } \u00a8 \u2373N }  \u2190 timed: apply fn N times"
        ),
    ),
    dict(
        name="Uiua",
        code="\u21bb1\u21cc\u2346",
        bytes=4,
        color="#ea1999",
        logo="uiua",
        source_code="\u21bb1\u21cc\u2346",
        bench=bench_uiua,
        script=(
            '  S \u2190 \u21afL "01"              \u2190 not timed\n'
            "  T \u2190 now\n"
            "  \u2365(\u25cc\u21bb1\u21cc\u2346 S)N            \u2190 timed: apply fn N times, pop result\n"
            "  &p \u00f7N -T now"
        ),
    ),
    dict(
        name="Uiua",
        code="count+construct",
        bytes=None,
        color="#ea1999",
        logo="uiua",
        source_code='\u02dc\u25bd"101"\u02dc\u22821\u2282\u2283-\u2081-\u2283/+\u29fb=@1',
        bench=bench_uiua_count,
        script=(
            '  \u02dc\u25bd"101"\u02dc\u22821\u2282\u2283-\u2081-\u2283/+\u29fb=@1\n'
            "  O(n) counting approach \u2014 no sort"
        ),
    ),
    dict(
        name="TinyAPL",
        code="1\u2218\u2296\u2218\u22b5",
        bytes=5,
        color="#8cc63f",
        logo="tinyapl",
        source_code="1\u2218\u2296\u2218\u22b5",
        bench=lambda: bench_tinyapl("1\u2218\u2296\u2218\u22b5"),
        script=(
            "  \u2395\u2190(1\u2218\u2296\u2218\u22b5) \u2395_Measure L\u2374'01'\n"
            "  \u2395_Measure times F applied to the argument (1 call)"
        ),
    ),
    dict(
        name="TinyAPL",
        code="1\u00ab\u2296\u00bb\u22b5",
        bytes=5,
        color="#8cc63f",
        logo="tinyapl",
        source_code="1\u00ab\u2296\u00bb\u22b5",
        bench=lambda: bench_tinyapl("1\u00ab\u2296\u00bb\u22b5"),
        script="  \u2395\u2190(1\u00ab\u2296\u00bb\u22b5) \u2395_Measure L\u2374'01'",
    ),
    dict(
        name="TinyAPL",
        code="\u29851\u2296\u22b5\u2986",
        bytes=5,
        color="#8cc63f",
        logo="tinyapl",
        source_code="\u29851\u2296\u22b5\u2986",
        bench=lambda: bench_tinyapl("\u29851\u2296\u22b5\u2986"),
        script="  \u2395\u2190(\u29851\u2296\u22b5\u2986) \u2395_Measure L\u2374'01'",
    ),
    dict(
        name="J",
        code="1|.\\:~",
        bytes=6,
        color="#2ad4ff",
        logo="j",
        source_code="1|.\\:~",
        bench=bench_j,
        script=(
            "  input =: L $ '01'                    \u2190 not timed\n"
            "  bench =: 3 : '1 |. \\:~ input'        \u2190 define fn\n"
            "  echo (6!:2 'bench\"0 i. N') % N       \u2190 timed: 6!:2 runs bench N times"
        ),
    ),
    dict(
        name="J",
        code="tacit count",
        bytes=None,
        color="#2ad4ff",
        logo="j",
        source_code="'101'#~1,~#(<:@],-)\n ([:+/'1'=[)",
        bench=bench_j_count,
        script=(
            "  bench =: 3 : '''101''#~1,~#(<:@],-)([:+/''1''=y)'\n"
            "  echo (6!:2 'bench\"0 i. N') % N       \u2190 O(n) counting, no sort"
        ),
    ),
    dict(
        name="APL",
        code="1\u233d\u2282\u2364\u2352\u235b\u2337",
        bytes=7,
        color="#24a148",
        logo="apl",
        source_code="1\u233d\u2282\u2364\u2352\u235b\u2337",
        bench=bench_apl,
        script=(
            "  input\u2190L\u2374'01'                         \u2190 not timed\n"
            "  t\u21903\u2283\u2395AI\n"
            "  _\u2190{(1\u233d\u2282\u2364\u2352\u235b\u2337)input}\u00a8\u2373N        \u2190 timed: apply fn N times\n"
            "  \u2395\u2190((3\u2283\u2395AI)-t)\u00f71000\u00d7N"
        ),
    ),
    dict(
        name="APL",
        code="tacit count",
        bytes=None,
        color="#24a148",
        logo="apl",
        source_code="'101'/\u23681,\u2368\u2262\n ((1-\u2368\u22a2),-)(+/'1'\u2218=)",
        bench=bench_apl_count,
        script=(
            "  Mob\u2190'101'/\u23681,\u2368\u2262((1-\u2368\u22a2),-)( +/'1'\u2218=)\n"
            "  _\u2190{Mob input}\u00a8\u2373N        \u2190 O(n) counting, no sort"
        ),
    ),
    dict(
        name="C++",
        code="partition+rotate",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  std::ranges::partition(s, [](auto c){ return c=='1'; });\n  std::ranges::rotate(s, std::next(s.begin()));\n  return s;\n}",
        bench=lambda: bench_cpp(
            "partition_rotate",
            "std::ranges::partition(s, [](auto c) { return c == '1'; });\n"
            "  std::ranges::rotate(s, std::next(s.begin()));",
        ),
        script=(
            "  // input created before timing\n"
            "  auto start = high_resolution_clock::now();\n"
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);  // pass-by-value copy + partition + rotate\n"
            "  auto end = high_resolution_clock::now();"
        ),
    ),
    dict(
        name="C++",
        code="sort+rotate",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  std::ranges::sort(s, std::greater{});\n  std::ranges::rotate(s, std::next(s.begin()));\n  return s;\n}",
        bench=lambda: bench_cpp(
            "sort_rotate",
            "std::ranges::sort(s, std::greater{});\n"
            "  std::ranges::rotate(s, std::next(s.begin()));",
        ),
        script=(
            "  // input created before timing\n"
            "  auto start = high_resolution_clock::now();\n"
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);  // pass-by-value copy + sort + rotate\n"
            "  auto end = high_resolution_clock::now();"
        ),
    ),
    dict(
        name="C++",
        code="count+construct",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  auto n = std::ranges::count(s, '1');\n  return std::string(n-1,'1')\n    + std::string(s.size()-n,'0') + '1';\n}",
        bench=lambda: bench_cpp(
            "count_construct",
            "auto n = std::ranges::count(s, '1');\n"
            "  s = std::string(n - 1, '1') + std::string(s.size() - n, '0') + '1';",
        ),
        script=(
            "  // O(n) count + construct, no sort\n"
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);"
        ),
    ),
    dict(
        name="C++",
        code="partition+partition_point",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  std::ranges::partition(s, [](auto c) { return c == '1'; });\n  auto it = std::ranges::partition_point(s, [](auto c) { return c == '1'; });\n  std::iter_swap(std::prev(it), std::prev(s.end()));\n  return s;\n}",
        bench=lambda: bench_cpp(
            "partition_point",
            "std::ranges::partition(s, [](auto c) { return c == '1'; });\n"
            "  auto it = std::ranges::partition_point(s, [](auto c) { return c == '1'; });\n"
            "  std::iter_swap(std::prev(it), std::prev(s.end()));",
        ),
        script=(
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);  // partition + partition_point + swap"
        ),
    ),
    dict(
        name="C++",
        code="partition",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  auto r = std::ranges::partition(s, [](auto c) { return c == '1'; });\n  std::iter_swap(std::prev(r.begin()), std::prev(s.end()));\n  return s;\n}",
        bench=lambda: bench_cpp(
            "partition_only",
            "auto r = std::ranges::partition(s, [](auto c) { return c == '1'; });\n"
            "  std::iter_swap(std::prev(r.begin()), std::prev(s.end()));",
        ),
        script=(
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);  // partition + swap"
        ),
    ),
    dict(
        name="C++",
        code="count+construct 2",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  auto n = std::ranges::count(s, '1');\n  std::string out;\n  out.reserve(s.length());\n  out.resize(n-1, '1');\n  out.resize(s.length() - 1, '0');\n  out.push_back('1');\n  return out;\n}",
        bench=lambda: bench_cpp(
            "count_construct2",
            "auto n = std::ranges::count(s, '1');\n"
            "  std::string out;\n"
            "  out.reserve(s.length());\n"
            "  out.resize(n - 1, '1');\n"
            "  out.resize(s.length() - 1, '0');\n"
            "  out.push_back('1');\n"
            "  s = std::move(out);",
        ),
        script=(
            "  // O(n) count + reserve/resize construct\n"
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);"
        ),
    ),
    dict(
        name="C++",
        code="count+construct 3",
        bytes=None,
        color="#659ad2",
        logo="cpp_logo",
        source_code="auto mob(std::string s) -> std::string {\n  auto n = std::ranges::count(s, '1');\n  std::string out;\n  out.resize_and_overwrite(s.length(),\n    [n](char* buf, size_t size) noexcept {\n      std::memset(buf, '1', n - 1);\n      std::memset(buf + n - 1, '0', size - n);\n      buf[size - 1] = '1';\n      return size;\n    });\n  return out;\n}",
        bench=lambda: bench_cpp(
            "count_construct3",
            "auto n = std::ranges::count(s, '1');\n"
            "  std::string out;\n"
            "  out.resize_and_overwrite(s.length(), [n](char* buf, size_t size) noexcept {\n"
            "    std::memset(buf, '1', n - 1);\n"
            "    std::memset(buf + n - 1, '0', size - n);\n"
            "    buf[size - 1] = '1';\n"
            "    return size;\n"
            "  });\n"
            "  s = std::move(out);",
        ),
        script=(
            "  // O(n) count + resize_and_overwrite + memset\n"
            "  for (int i = 0; i < N; ++i)\n"
            "    result = maximum_odd_binary(input);"
        ),
    ),
    dict(
        name="Rust",
        code="sort+rotate",
        bytes=None,
        color="#dea584",
        logo="rust_logo_darkmode",
        source_code="fn mob(mut s: Vec<u8>) -> Vec<u8> {\n  s.sort_unstable_by(|a,b| b.cmp(a));\n  s.rotate_left(1);\n  s\n}",
        bench=lambda: bench_rust(
            "rust_sort",
            "s.sort_unstable_by(|a, b| b.cmp(a));\n    s.rotate_left(1);",
        ),
        script=(
            "  let start = Instant::now();\n"
            "  for _ in 0..N {\n"
            "      result = maximum_odd_binary(input.clone());  // clone + sort + rotate\n"
            "  }"
        ),
    ),
    dict(
        name="Rust",
        code="partition+rotate",
        bytes=None,
        color="#dea584",
        logo="rust_logo_darkmode",
        source_code="fn mob(mut s: Vec<u8>) -> Vec<u8> {\n  s.iter_mut()\n    .partition_in_place(|c| *c == b'1');\n  s.rotate_left(1);\n  s\n}",
        bench=lambda: bench_rust_nightly(
            "rust_partition",
            "s.iter_mut().partition_in_place(|c| *c == b'1');\n    s.rotate_left(1);",
        ),
        script=(
            "  #![feature(iter_partition_in_place)]  // nightly\n"
            "  for _ in 0..N {\n"
            "      result = maximum_odd_binary(input.clone());  // clone + partition + rotate\n"
            "  }"
        ),
    ),
    dict(
        name="Rust",
        code="count+construct",
        bytes=None,
        color="#dea584",
        logo="rust_logo_darkmode",
        source_code="fn mob(mut s: Vec<u8>) -> Vec<u8> {\n  let n = s.iter()\n    .filter(|&&c| c==b'1').count();\n  let len = s.len();\n  s.clear();\n  s.extend(repeat(b'1').take(n-1));\n  s.extend(repeat(b'0').take(len-n));\n  s.push(b'1');\n  s\n}",
        bench=lambda: bench_rust(
            "rust_count",
            "let n = s.iter().filter(|&&c| c == b'1').count();\n"
            "    let len = s.len();\n"
            "    s.clear();\n"
            "    s.extend(std::iter::repeat(b'1').take(n - 1));\n"
            "    s.extend(std::iter::repeat(b'0').take(len - n));\n"
            "    s.push(b'1');",
        ),
        script=(
            "  // O(n) count + construct, no sort\n"
            "  for _ in 0..N {\n"
            "      result = maximum_odd_binary(input.clone());\n"
            "  }"
        ),
    ),
    dict(
        name="Nim",
        code="sort+rotate",
        bytes=None,
        color="#ffe953",
        logo="nim_logo",
        source_code="proc mob(s: var string) =\n  sort(s, order = Descending)\n  rotateLeft(s, 1)",
        bench=lambda: bench_nim(
            "nim_sort",
            "sort(s, order = Descending)\n  rotateLeft(s, 1)",
        ),
        script=(
            "  let start = getMonoTime()\n"
            "  for i in 0 ..< N:\n"
            "      result = base; maximumOddBinary(result)  // copy + sort + rotate\n"
            "  let elapsed = ..."
        ),
    ),
    dict(
        name="D",
        code="sort+rotate",
        bytes=None,
        color="#b63838",
        logo="dlang_logo",
        source_code='auto mob(char[] s) {\n  s.representation.sort!"a > b";\n  bringToFront(s[0..1], s[1..$]);\n  return s;\n}',
        bench=lambda: bench_d(
            "d_sort",
            's.representation.sort!"a > b";\n    bringToFront(s[0..1],s[1..$]);',
        ),
        script=(
            "  auto start = StopWatch(AutoStart.yes);\n"
            "  foreach(i;0 .. N) {\n"
            "      result = maximumOddBinary(input); // sort + rotate\n"
            "  }"
        ),
    ),
    dict(
        name="D",
        code="partition+rotate",
        bytes=None,
        color="#b63838",
        logo="dlang_logo",
        source_code="auto mob(char[] s) {\n  s.representation.partition!(c => c == '1');\n  bringToFront(s[0 .. 1], s[1 .. $]);\n  return s;\n}",
        bench=lambda: bench_d(
            "d_partition",
            "s.representation.partition!(c => c == '1');\n     bringToFront(s[0 .. 1], s[1 .. $]);",
        ),
        script=(
            "  foreach(i;0 .. N) {\n"
            "      result = maximumOddBinary(input); // partition + rotate\n"
            "  }"
        ),
    ),
    dict(
        name="D",
        code="count+construct",
        bytes=None,
        color="#b63838",
        logo="dlang_logo",
        source_code="auto mob(char[] s) {\n  auto n = s.count('1') - 1;\n  s[0 .. n].fill('1');\n  s[n .. $ - 1].fill('0');\n  s[$ - 1] = '1';\n  return s;\n}",
        bench=lambda: bench_d(
            "d_count",
            "    auto n = s.count('1') - 1;\n"
            "    s[0 .. n].fill('1');\n"
            "    s[n .. $ - 1].fill('0');\n"
            "    s[$ - 1] = '1';",
        ),
        script=(
            "  // O(n) count + construct, no sort\n"
            "  foreach(i;0 .. N) {\n"
            "      result = maximumOddBinary(input);\n"
            "  }"
        ),
    ),
    dict(
        name="Julia",
        code="sort+circshift",
        bytes=None,
        color="#9558b2",
        logo="julia_logo_darkmode",
        source_code="function mob(s)\n  v = sort(s, rev=true)\n  circshift(v, -1)\nend",
        bench=bench_julia,
        script=(
            "  maximum_odd_binary(input)  # warmup\n"
            "  t = @elapsed for _ in 1:N\n"
            "      maximum_odd_binary(input)  # sort + circshift (allocates)\n"
            "  end"
        ),
    ),
    dict(
        name="Julia",
        code="bool sort+circshift",
        bytes=None,
        color="#9558b2",
        logo="julia_logo_darkmode",
        source_code="function mob(s)\n  v = circshift(sort(collect(s) .== '1', rev=true), -1)\n  String(map(b -> b ? '1' : '0', v))\nend",
        bench=bench_julia_bool,
        script=(
            "  maximum_odd_binary(input)  # warmup\n"
            "  t = @elapsed for _ in 1:N\n"
            "      maximum_odd_binary(input)  # str→Bool[], sort, circshift, →str\n"
            "  end"
        ),
    ),
    dict(
        name="Julia",
        code="count+construct",
        bytes=None,
        color="#9558b2",
        logo="julia_logo_darkmode",
        source_code="function mob(s)\n  n = count(==('1'), s)\n  '1'^(n-1) * '0'^(length(s)-n) * \"1\"\nend",
        bench=bench_julia_count,
        script=(
            "  maximum_odd_binary(input)  # warmup\n"
            "  t = @elapsed for _ in 1:N\n"
            "      maximum_odd_binary(input)  # O(n) count + string concat\n"
            "  end"
        ),
    ),
    dict(
        name="Python",
        code="sort+rotate",
        bytes=None,
        color="#3776ab",
        logo="python_logo",
        source_code="def mob(s):\n  s = list(s)\n  s.sort(reverse=True)\n  s.append(s.pop(0))\n  return ''.join(s)",
        bench=bench_python_sort,
        script=(
            "  start = time.perf_counter()\n"
            "  for _ in range(N):\n"
            "      maximum_odd_binary(input)  # list() + sort + pop/append + join\n"
            "  elapsed = ..."
        ),
    ),
]


# ─────────────────────────── Main ───────────────────────────────


def _fmt_time(secs):
    us = secs * 1e6
    if us < 1:
        return f"{us * 1000:.0f} ns"
    if us < 1000:
        return f"{us:.1f} \u03bcs"
    return f"{us / 1000:.2f} ms"


def run_benchmarks_for_size(input_len, log_scripts=False, lang_filter=None):
    """Run solutions for a given input length. If lang_filter is set, only run matching."""
    global INPUT_LEN
    INPUT_LEN = input_len
    _cpp_bin_cache.clear()
    _d_bin_cache.clear()
    _rust_bin_cache.clear()
    _rust_nightly_bin_cache.clear()
    _nim_bin_cache.clear()

    results = []
    for sol in SOLUTIONS:
        if lang_filter and not any(
            f.lower() in sol["name"].lower() or f.lower() in sol["code"].lower()
            for f in lang_filter
        ):
            continue
        tag = f"{sol['name']}  {sol['code']}"
        print(f"  \u25b8 {tag}")
        if log_scripts and sol.get("script"):
            print(f"    \u250c\u2500 script (N={N_ITERS}, L={INPUT_LEN}):")
            for line in sol["script"].splitlines():
                print(f"    \u2502 {line}")
            print(f"    \u2514\u2500")

        times = []
        for r_idx in range(N_RUNS):
            t = sol["bench"]()
            if t is not None and t >= 0:
                times.append(t)
                print(f"      run {r_idx + 1}/{N_RUNS}: {t * 1e6:>10.2f} \u03bcs")
            else:
                print(f"      run {r_idx + 1}/{N_RUNS}: FAILED")

        if times:
            med = statistics.median(times)
            results.append(
                dict(
                    name=sol["name"],
                    code=sol["code"],
                    bytes=sol.get("bytes"),
                    color=sol["color"],
                    logo=sol.get("logo", ""),
                    time=med,
                    all_times=times,
                    script=sol.get("script", ""),
                    source_code=sol.get("source_code", ""),
                )
            )
            print(f"    \u2192 median: {_fmt_time(med)}")
        else:
            print(f"    \u2717 skipped")

    results.sort(key=lambda x: x["time"])
    return results


def _result_key(r):
    return r["name"] + "\x00" + r["code"]


def _load_cache():
    """Load previous results from JSON cache."""
    import json

    if CACHE_FILE.exists():
        try:
            raw = json.loads(CACHE_FILE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:
            pass
    return {}


def _save_cache(all_results):
    """Persist results to JSON cache."""
    import json

    serializable = {}
    for size, results in all_results.items():
        serializable[str(size)] = [
            {k: v for k, v in r.items() if k != "bench"} for r in results
        ]
    CACHE_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))


def _merge_results(cached, fresh):
    """Merge fresh results into cached, replacing entries with matching name+code."""
    by_key = {_result_key(r): r for r in cached}
    for r in fresh:
        by_key[_result_key(r)] = r
    merged = list(by_key.values())
    merged.sort(key=lambda x: x["time"])
    return merged


def _refresh_metadata(all_results):
    """Overlay non-timing fields from SOLUTIONS onto cached results."""
    meta = {}
    for sol in SOLUTIONS:
        key = sol["name"] + "\x00" + sol["code"]
        meta[key] = {
            k: sol[k]
            for k in ("source_code", "script", "color", "logo", "bytes")
            if k in sol
        }
    for size, results in all_results.items():
        for r in results:
            key = _result_key(r)
            if key in meta:
                r.update(meta[key])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Maximum Odd Binary Number")
    parser.add_argument(
        "--lang",
        nargs="+",
        metavar="FILTER",
        help="Only run solutions matching these names (e.g. --lang Python Rust). "
        "Results merge into cached data from previous runs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available solution names and exit.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Rebuild HTML from cached results without running benchmarks.",
    )
    args = parser.parse_args()

    if args.html_only:
        cached = _load_cache()
        if not cached:
            print("No cached results found. Run benchmarks first.")
            sys.exit(1)
        _refresh_metadata(cached)
        print("Rebuilding outputs from cache...")
        default_size = 1_000
        results = cached.get(default_size) or next(iter(cached.values()))
        generate_chart(results, ROOT / "img" / "benchmark_all.png")
        results_filtered = [r for r in results if r["name"] not in ("TinyAPL", "Kap")]
        generate_chart(results_filtered, ROOT / "img" / "benchmark_no_tinyapl.png")
        generate_html(cached, ROOT / "benchmark.html")
        return

    if args.list:
        seen = set()
        for sol in SOLUTIONS:
            tag = f"{sol['name']:>10}  {sol['code']}"
            if tag not in seen:
                seen.add(tag)
                print(tag)
        return

    lang_filter = args.lang

    print("=" * 60)
    print("  Maximum Odd Binary Number \u2014 Benchmark")
    print(f"  Sizes: {INPUT_SIZES}")
    print(f"  Iterations: {N_ITERS:,}  |  Runs: {N_RUNS} (median)")
    if lang_filter:
        print(f"  Filter: {', '.join(lang_filter)}")
    print("=" * 60)

    cached = _load_cache()
    if lang_filter and not cached:
        print("\n  WARNING: No cached results found. Run once without --lang first")
        print(
            "  to benchmark all solutions, then use --lang to update specific ones.\n"
        )

    all_results = {}
    for size in INPUT_SIZES:
        print(f"\n{'─' * 60}")
        print(f"  Input size: {size:,} chars")
        print(f"{'─' * 60}")
        fresh = run_benchmarks_for_size(
            size,
            log_scripts=(size == INPUT_SIZES[0]),
            lang_filter=lang_filter,
        )
        base = cached.get(size, [])
        all_results[size] = _merge_results(base, fresh)

    _save_cache(all_results)

    default_size = 1_000
    results = all_results.get(default_size) or next(iter(all_results.values()))

    print(f"\n{'=' * 60}")
    print(f"  Results for {default_size:,}-char input (fastest \u2192 slowest)")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(
            f"  {i:>2}. {r['name']:>10}  {r['code']:<24} {_fmt_time(r['time']):>10}"
        )

    print("\nGenerating outputs...")
    generate_chart(results, ROOT / "img" / "benchmark_all.png")
    results_filtered = [r for r in results if r["name"] not in ("TinyAPL", "Kap")]
    generate_chart(results_filtered, ROOT / "img" / "benchmark_no_tinyapl.png")
    generate_html(all_results, ROOT / "benchmark.html")


if __name__ == "__main__":
    main()
