from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "metadata.json"
DEFAULT_GPU_BENCHMARK_ARGS = [
    "--metrics",
    "speedup,accuracy,tflops,gbps",
    "--input-sample-mode",
    "equally-spaced-k",
    "--num-inputs",
    "20",
]


@dataclass
class RepoTarget:
    label: str
    repo_root: Path
    git_ref: str
    git_sha: str
    dirty: bool
    temporary: bool = False


@dataclass
class CommandArtifact:
    status: str
    command: list[str]
    cwd: Path
    stdout_path: Path
    stderr_path: Path
    json_path: Path | None = None
    extra_json_path: Path | None = None
    reason: str | None = None


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )


def git_stdout(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def is_dirty(repo_root: Path) -> bool:
    return bool(git_stdout(repo_root, "status", "--short"))


def current_target(repo_root: Path) -> RepoTarget:
    sha = git_stdout(repo_root, "rev-parse", "HEAD")
    return RepoTarget(
        label="candidate",
        repo_root=repo_root,
        git_ref="WORKTREE",
        git_sha=sha,
        dirty=is_dirty(repo_root),
    )


def make_worktree(repo_root: Path, ref: str, workspace: Path) -> RepoTarget:
    worktree_root = workspace / f"worktree-{ref.replace('/', '-')}"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_root), ref],
        cwd=repo_root,
        check=True,
    )
    sha = git_stdout(worktree_root, "rev-parse", "HEAD")
    return RepoTarget(
        label="baseline",
        repo_root=worktree_root,
        git_ref=ref,
        git_sha=sha,
        dirty=False,
        temporary=True,
    )


def cleanup_worktree(repo_root: Path, target: RepoTarget) -> None:
    if not target.temporary:
        return
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(target.repo_root)],
        cwd=repo_root,
        check=False,
    )


def run_microbench(
    harness_root: Path,
    target: RepoTarget,
    output_dir: Path,
    repeat: int,
) -> CommandArtifact:
    stdout_path = output_dir / f"{target.label}_micro.stdout"
    stderr_path = output_dir / f"{target.label}_micro.stderr"
    json_path = output_dir / f"{target.label}_micro.json"
    cmd = [
        sys.executable,
        str(harness_root / "benchmarks" / "autotuner_hotpaths.py"),
        "--repo-root",
        str(target.repo_root),
        "--repeat",
        str(repeat),
        "--json",
        str(json_path),
    ]
    result = run_command(
        cmd,
        cwd=harness_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    return CommandArtifact(
        status="ok" if result.returncode == 0 else "failed",
        command=cmd,
        cwd=harness_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        json_path=json_path if json_path.exists() else None,
        reason=None if result.returncode == 0 else f"exit code {result.returncode}",
    )


def can_run_gpu_suite() -> tuple[bool, str | None]:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, torch; "
            "have_tb = importlib.util.find_spec('tritonbench') is not None; "
            "print(int(torch.cuda.is_available() and have_tb))",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return False, probe.stderr.strip() or "environment probe failed"
    if probe.stdout.strip() == "1":
        return True, None
    return False, "CUDA or tritonbench is unavailable in this environment"


def run_gpu_suite(
    target: RepoTarget,
    output_dir: Path,
    benchmark_args: list[str],
) -> CommandArtifact:
    supported, reason = can_run_gpu_suite()
    stdout_path = output_dir / f"{target.label}_gpu.stdout"
    stderr_path = output_dir / f"{target.label}_gpu.stderr"
    json_path = output_dir / f"{target.label}_gpu.json"
    autotune_path = output_dir / f"{target.label}_autotune.json"
    if not supported:
        stdout_path.write_text("")
        stderr_path.write_text(reason or "")
        return CommandArtifact(
            status="skipped",
            command=[],
            cwd=target.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            json_path=None,
            extra_json_path=None,
            reason=reason,
        )

    cmd = [
        sys.executable,
        str(target.repo_root / "benchmarks" / "run.py"),
        "--output",
        str(json_path),
        "--autotune-metrics-json",
        str(autotune_path),
        *benchmark_args,
    ]
    env = dict(os.environ)
    env["HELION_BENCHMARK_DISABLE_LOGGING"] = "1"
    result = run_command(
        cmd,
        cwd=target.repo_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env=env,
    )
    return CommandArtifact(
        status="ok" if result.returncode == 0 else "failed",
        command=cmd,
        cwd=target.repo_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        json_path=json_path if json_path.exists() else None,
        extra_json_path=autotune_path if autotune_path.exists() else None,
        reason=None if result.returncode == 0 else f"exit code {result.returncode}",
    )


def load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_microbench(
    records: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    return {
        record["name"]: {
            "parity_ok": bool(record["parity_ok"]),
            "current_ms": float(record["current_ms"]),
            "legacy_ms": float(record["legacy_ms"]),
            "speedup": float(record["speedup"]),
        }
        for record in records
    }


def summarize_gpu_records(
    records: list[dict[str, Any]] | None,
) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}
    if not records:
        return summary
    for record in records:
        kernel = str(record["model"]["name"])
        metric = str(record["metric"]["name"])
        if not metric.startswith("helion_"):
            continue
        values = [float(value) for value in record["metric"]["benchmark_values"]]
        if not values:
            continue
        summary[(kernel, metric)] = {
            "count": float(len(values)),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_autotune(records: dict[str, Any] | None) -> dict[str, float]:
    if not records:
        return {}
    runs = list(records.get("runs", []))
    if not runs:
        return {}
    total_time = sum(float(run["autotune_time"]) for run in runs)
    total_configs = sum(int(run["num_configs_tested"]) for run in runs)
    total_compile_failures = sum(int(run["num_compile_failures"]) for run in runs)
    total_accuracy_failures = sum(int(run["num_accuracy_failures"]) for run in runs)
    avg_best_perf = sum(float(run["best_perf_ms"]) for run in runs) / len(runs)
    avg_generations = sum(int(run["num_generations"]) for run in runs) / len(runs)
    return {
        "num_runs": float(len(runs)),
        "total_autotune_time_s": total_time,
        "avg_autotune_time_s": total_time / len(runs),
        "total_configs_tested": float(total_configs),
        "avg_configs_tested": total_configs / len(runs),
        "total_compile_failures": float(total_compile_failures),
        "total_accuracy_failures": float(total_accuracy_failures),
        "avg_best_perf_ms": avg_best_perf,
        "avg_generations": avg_generations,
    }


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_NAME


def serialize_target(target: RepoTarget) -> dict[str, Any]:
    return {
        "label": target.label,
        "git_ref": target.git_ref,
        "git_sha": target.git_sha,
        "dirty": target.dirty,
    }


def deserialize_target(data: dict[str, Any]) -> RepoTarget:
    return RepoTarget(
        label=str(data["label"]),
        repo_root=REPO_ROOT,
        git_ref=str(data["git_ref"]),
        git_sha=str(data["git_sha"]),
        dirty=bool(data["dirty"]),
    )


def serialize_path(path: Path | None, *, relative_to: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(relative_to))
    except ValueError:
        return str(path)


def deserialize_path(raw_path: str | None, *, relative_to: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return relative_to / path


def serialize_artifact(
    artifact: CommandArtifact, *, relative_to: Path
) -> dict[str, Any]:
    return {
        "status": artifact.status,
        "command": artifact.command,
        "cwd": serialize_path(artifact.cwd, relative_to=relative_to),
        "stdout_path": serialize_path(artifact.stdout_path, relative_to=relative_to),
        "stderr_path": serialize_path(artifact.stderr_path, relative_to=relative_to),
        "json_path": serialize_path(artifact.json_path, relative_to=relative_to),
        "extra_json_path": serialize_path(
            artifact.extra_json_path, relative_to=relative_to
        ),
        "reason": artifact.reason,
    }


def deserialize_artifact(data: dict[str, Any], *, relative_to: Path) -> CommandArtifact:
    return CommandArtifact(
        status=str(data["status"]),
        command=[str(part) for part in data.get("command", [])],
        cwd=deserialize_path(data.get("cwd"), relative_to=relative_to) or REPO_ROOT,
        stdout_path=deserialize_path(data.get("stdout_path"), relative_to=relative_to)
        or relative_to / "missing.stdout",
        stderr_path=deserialize_path(data.get("stderr_path"), relative_to=relative_to)
        or relative_to / "missing.stderr",
        json_path=deserialize_path(data.get("json_path"), relative_to=relative_to),
        extra_json_path=deserialize_path(
            data.get("extra_json_path"), relative_to=relative_to
        ),
        reason=None if data.get("reason") is None else str(data["reason"]),
    )


def write_manifest(
    output_dir: Path,
    *,
    baseline_target: RepoTarget,
    candidate_target: RepoTarget | None,
    micro_baseline: CommandArtifact,
    micro_candidate: CommandArtifact | None,
    gpu_baseline: CommandArtifact | None,
    gpu_candidate: CommandArtifact | None,
    benchmark_args: list[str],
    micro_repeat: int,
) -> None:
    payload = {
        "version": 1,
        "baseline_target": serialize_target(baseline_target),
        "candidate_target": None
        if candidate_target is None
        else serialize_target(candidate_target),
        "benchmark_args": list(benchmark_args),
        "micro_repeat": micro_repeat,
        "artifacts": {
            "micro_baseline": serialize_artifact(
                micro_baseline, relative_to=output_dir
            ),
            "micro_candidate": None
            if micro_candidate is None
            else serialize_artifact(micro_candidate, relative_to=output_dir),
            "gpu_baseline": None
            if gpu_baseline is None
            else serialize_artifact(gpu_baseline, relative_to=output_dir),
            "gpu_candidate": None
            if gpu_candidate is None
            else serialize_artifact(gpu_candidate, relative_to=output_dir),
        },
    }
    manifest_path(output_dir).write_text(json.dumps(payload, indent=2))


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected cached baseline manifest at {path}, but it does not exist."
        )
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {path} is not a JSON object.")
    return data


def load_cached_baseline(
    output_dir: Path,
) -> tuple[RepoTarget, CommandArtifact, CommandArtifact | None, dict[str, Any]]:
    data = load_manifest(output_dir)
    baseline_target = deserialize_target(data["baseline_target"])
    artifacts = data["artifacts"]
    micro_baseline = deserialize_artifact(
        artifacts["micro_baseline"], relative_to=output_dir
    )
    gpu_baseline = None
    if artifacts.get("gpu_baseline") is not None:
        gpu_baseline = deserialize_artifact(
            artifacts["gpu_baseline"], relative_to=output_dir
        )
    return baseline_target, micro_baseline, gpu_baseline, data


def validate_cached_baseline(
    metadata: dict[str, Any],
    *,
    benchmark_args: list[str],
    micro_repeat: int,
    require_gpu: bool,
) -> None:
    cached_micro_repeat = int(metadata["micro_repeat"])
    if cached_micro_repeat != micro_repeat:
        raise ValueError(
            "Cached baseline micro-repeat does not match this run: "
            f"{cached_micro_repeat} != {micro_repeat}."
        )
    cached_args = [str(arg) for arg in metadata.get("benchmark_args", [])]
    if require_gpu and cached_args != benchmark_args:
        raise ValueError(
            "Cached baseline GPU args do not match this run. "
            f"cached={cached_args}, requested={benchmark_args}"
        )
    if require_gpu and metadata["artifacts"].get("gpu_baseline") is None:
        raise ValueError("Cached baseline does not include GPU artifacts.")


def normalize_benchmark_args(args: list[str]) -> list[str]:
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        return list(DEFAULT_GPU_BENCHMARK_ARGS)
    present_flags = {arg for arg in args if arg.startswith("--")}
    merged = []
    default_pairs = zip(
        DEFAULT_GPU_BENCHMARK_ARGS[::2],
        DEFAULT_GPU_BENCHMARK_ARGS[1::2],
        strict=True,
    )
    for flag, value in default_pairs:
        if flag not in present_flags:
            merged.extend([flag, value])
    merged.extend(args)
    return merged


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compare_micro_sections(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> str:
    case_names = sorted(set(baseline) | set(candidate))
    rows = []
    for case in case_names:
        base = baseline.get(case)
        cand = candidate.get(case)
        base_ms = None if base is None else float(base["current_ms"])
        cand_ms = None if cand is None else float(cand["current_ms"])
        branch_speedup = None
        if base_ms is not None and cand_ms is not None and cand_ms != 0:
            branch_speedup = base_ms / cand_ms
        rows.append(
            [
                case,
                "-" if base is None else ("ok" if base["parity_ok"] else "FAIL"),
                "-" if cand is None else ("ok" if cand["parity_ok"] else "FAIL"),
                format_float(base_ms),
                format_float(cand_ms),
                format_float(branch_speedup, digits=2),
            ]
        )
    return markdown_table(
        [
            "Case",
            "Baseline Parity",
            "Candidate Parity",
            "Baseline ms",
            "Candidate ms",
            "Main/Candidate",
        ],
        rows,
    )


def compare_gpu_sections(
    baseline: dict[tuple[str, str], dict[str, float]],
    candidate: dict[tuple[str, str], dict[str, float]],
) -> str:
    keys = sorted(set(baseline) | set(candidate))
    rows = []
    for kernel, metric in keys:
        base = baseline.get((kernel, metric))
        cand = candidate.get((kernel, metric))
        base_mean = None if base is None else base["mean"]
        cand_mean = None if cand is None else cand["mean"]
        ratio = None
        if base_mean is not None and cand_mean is not None and base_mean != 0:
            ratio = cand_mean / base_mean
        rows.append(
            [
                kernel,
                metric,
                format_float(base_mean),
                format_float(cand_mean),
                format_float(ratio, digits=2),
            ]
        )
    return markdown_table(
        ["Kernel", "Metric", "Baseline Mean", "Candidate Mean", "Candidate/Baseline"],
        rows,
    )


def compare_autotune_sections(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> str:
    keys = sorted(set(baseline) | set(candidate))
    rows = []
    for key in keys:
        base = baseline.get(key)
        cand = candidate.get(key)
        ratio = None
        if base is not None and cand is not None and base != 0:
            ratio = cand / base
        rows.append(
            [
                key,
                format_float(base),
                format_float(cand),
                format_float(ratio, digits=2),
            ]
        )
    return markdown_table(
        ["Metric", "Baseline", "Candidate", "Candidate/Baseline"],
        rows,
    )


def artifact_summary(name: str, artifact: CommandArtifact) -> str:
    lines = [
        f"- {name} status: `{artifact.status}`",
        f"- stdout: `{artifact.stdout_path}`",
        f"- stderr: `{artifact.stderr_path}`",
    ]
    if artifact.json_path is not None:
        lines.append(f"- json: `{artifact.json_path}`")
    if artifact.extra_json_path is not None:
        lines.append(f"- extra json: `{artifact.extra_json_path}`")
    if artifact.reason is not None:
        lines.append(f"- reason: {artifact.reason}")
    if artifact.command:
        lines.append(f"- command: `{' '.join(artifact.command)}`")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    *,
    baseline_target: RepoTarget,
    candidate_target: RepoTarget,
    micro_baseline: CommandArtifact,
    micro_candidate: CommandArtifact,
    gpu_baseline: CommandArtifact | None,
    gpu_candidate: CommandArtifact | None,
) -> None:
    baseline_micro = summarize_microbench(load_json(micro_baseline.json_path))
    candidate_micro = summarize_microbench(load_json(micro_candidate.json_path))

    gpu_baseline_records = (
        None if gpu_baseline is None else load_json(gpu_baseline.json_path)
    )
    gpu_candidate_records = (
        None if gpu_candidate is None else load_json(gpu_candidate.json_path)
    )
    gpu_baseline_summary = summarize_gpu_records(gpu_baseline_records)
    gpu_candidate_summary = summarize_gpu_records(gpu_candidate_records)

    autotune_baseline = (
        {}
        if gpu_baseline is None
        else summarize_autotune(load_json(gpu_baseline.extra_json_path))
    )
    autotune_candidate = (
        {}
        if gpu_candidate is None
        else summarize_autotune(load_json(gpu_candidate.extra_json_path))
    )

    lines = [
        "# Helion Benchmark Comparison",
        "",
        f"- Timestamp: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Baseline: `{baseline_target.git_ref}` @ `{baseline_target.git_sha[:12]}`",
        f"- Candidate: `{candidate_target.git_ref}` @ `{candidate_target.git_sha[:12]}`",
        f"- Candidate dirty: `{candidate_target.dirty}`",
        "",
        "## Control-Plane Microbenchmarks",
        "",
        compare_micro_sections(baseline_micro, candidate_micro),
        "",
        "### Artifacts",
        artifact_summary("baseline microbench", micro_baseline),
        "",
        artifact_summary("candidate microbench", micro_candidate),
        "",
    ]

    if gpu_baseline is not None and gpu_candidate is not None:
        lines.extend(
            [
                "## GPU Kernel Benchmarks",
                "",
                compare_gpu_sections(gpu_baseline_summary, gpu_candidate_summary),
                "",
                "## Autotune Metrics",
                "",
                compare_autotune_sections(autotune_baseline, autotune_candidate),
                "",
                "### Artifacts",
                artifact_summary("baseline gpu suite", gpu_baseline),
                "",
                artifact_summary("candidate gpu suite", gpu_candidate),
                "",
            ]
        )

    report_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Helion benchmark results between main and the current checkout.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--baseline-ref",
        default="main",
        help="Git ref to use as the baseline checkout (default: main).",
    )
    parser.add_argument(
        "--candidate-ref",
        default=None,
        help="Optional git ref to use as the candidate instead of the current worktree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="Directory for JSON artifacts and the Markdown summary.",
    )
    parser.add_argument(
        "--micro-repeat",
        type=int,
        default=10,
        help="Repeat count for the control-plane microbenchmarks (default: 10).",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Only run the CPU control-plane harness.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Capture baseline artifacts only and skip the candidate run.",
    )
    parser.add_argument(
        "--reuse-baseline-dir",
        type=Path,
        default=None,
        help="Reuse baseline artifacts from an existing results directory instead of rerunning the baseline.",
    )

    args, benchmark_args = parser.parse_known_args()
    benchmark_args = normalize_benchmark_args(benchmark_args)
    if args.baseline_only and args.reuse_baseline_dir is not None:
        parser.error("--baseline-only cannot be combined with --reuse-baseline-dir")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = Path(tempfile.mkdtemp(prefix="helion-bench-"))
    baseline_target: RepoTarget | None = None
    candidate_target: RepoTarget | None = None
    micro_baseline: CommandArtifact | None = None
    micro_candidate: CommandArtifact | None = None
    gpu_baseline: CommandArtifact | None = None
    gpu_candidate: CommandArtifact | None = None

    try:
        if args.reuse_baseline_dir is None:
            baseline_target = make_worktree(REPO_ROOT, args.baseline_ref, workspace)
            micro_baseline = run_microbench(
                REPO_ROOT, baseline_target, output_dir, args.micro_repeat
            )
            if not args.skip_gpu:
                gpu_baseline = run_gpu_suite(
                    baseline_target, output_dir, benchmark_args
                )
        else:
            (
                baseline_target,
                micro_baseline,
                gpu_baseline,
                baseline_metadata,
            ) = load_cached_baseline(args.reuse_baseline_dir)
            validate_cached_baseline(
                baseline_metadata,
                benchmark_args=benchmark_args,
                micro_repeat=args.micro_repeat,
                require_gpu=not args.skip_gpu,
            )

        assert baseline_target is not None
        assert micro_baseline is not None

        if args.baseline_only:
            write_manifest(
                output_dir,
                baseline_target=baseline_target,
                candidate_target=None,
                micro_baseline=micro_baseline,
                micro_candidate=None,
                gpu_baseline=gpu_baseline,
                gpu_candidate=None,
                benchmark_args=benchmark_args,
                micro_repeat=args.micro_repeat,
            )
            print(f"Baseline artifacts written to {output_dir}")
            return 0

        if args.candidate_ref is None:
            candidate_target = current_target(REPO_ROOT)
        else:
            candidate_target = make_worktree(REPO_ROOT, args.candidate_ref, workspace)
            candidate_target.label = "candidate"

        micro_candidate = run_microbench(
            REPO_ROOT, candidate_target, output_dir, args.micro_repeat
        )
        if not args.skip_gpu:
            gpu_candidate = run_gpu_suite(candidate_target, output_dir, benchmark_args)

        report_path = output_dir / "summary.md"
        write_report(
            report_path,
            baseline_target=baseline_target,
            candidate_target=candidate_target,
            micro_baseline=micro_baseline,
            micro_candidate=micro_candidate,
            gpu_baseline=gpu_baseline,
            gpu_candidate=gpu_candidate,
        )
        write_manifest(
            output_dir,
            baseline_target=baseline_target,
            candidate_target=candidate_target,
            micro_baseline=micro_baseline,
            micro_candidate=micro_candidate,
            gpu_baseline=gpu_baseline,
            gpu_candidate=gpu_candidate,
            benchmark_args=benchmark_args,
            micro_repeat=args.micro_repeat,
        )

        print(f"Report written to {report_path}")
        return 0
    finally:
        if baseline_target is not None:
            cleanup_worktree(REPO_ROOT, baseline_target)
        if candidate_target is not None and candidate_target.temporary:
            cleanup_worktree(REPO_ROOT, candidate_target)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
