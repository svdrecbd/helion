import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BenchmarkResult:
    name: str
    parity_ok: bool
    legacy_ms: float
    current_ms: float
    speedup: float
    metadata: dict[str, Any]


@dataclass
class RepoApi:
    CompileEnvironment: Any
    CompactedShape: Any
    Config: Any
    DeviceFunction: Any
    LFBOPatternSearch: Any
    PerformanceTarget: Any
    ShapeConfigData: Any
    TileStrategyDispatch: Any
    select_config_subset_single: Any


def load_repo_api(repo_root: Path) -> RepoApi:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.tile_dispatch import TileStrategyDispatch
    from helion._compiler.tile_strategy import CompactedShape
    from helion.autotuner.heuristic_generator import PerformanceTarget
    from helion.autotuner.heuristic_generator import ShapeConfigData
    from helion.autotuner.heuristic_generator import _select_config_subset_single
    from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
    from helion.runtime.config import Config

    return RepoApi(
        CompileEnvironment=CompileEnvironment,
        CompactedShape=CompactedShape,
        Config=Config,
        DeviceFunction=DeviceFunction,
        LFBOPatternSearch=LFBOPatternSearch,
        PerformanceTarget=PerformanceTarget,
        ShapeConfigData=ShapeConfigData,
        TileStrategyDispatch=TileStrategyDispatch,
        select_config_subset_single=_select_config_subset_single,
    )


class MockSurrogate:
    def __init__(self, proba: np.ndarray, leaf_indices: np.ndarray) -> None:
        self.proba = proba
        self.leaf_indices = leaf_indices

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ids = np.asarray(X)[:, 0].astype(int)
        probs = self.proba[ids]
        return np.column_stack((1.0 - probs, probs))

    def apply(self, X: np.ndarray) -> np.ndarray:
        ids = np.asarray(X)[:, 0].astype(int)
        return self.leaf_indices[ids]


class MockTileStrategy:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def block_size_var(self, block_idx: int) -> str:
        return f"_BLOCK_{block_idx}"

    def compact_shape(self, shapes: list[Any]) -> list[Any]:
        return shapes


def _time_ms(fn: Any, repeat: int) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(repeat, 1)


def legacy_surrogate_select(
    search: Any,
    candidates: list[SimpleNamespace],
    n_sorted: int,
) -> list[SimpleNamespace]:
    candidate_X = np.array(
        [search.config_gen.encode_config(member.flat_values) for member in candidates]
    )
    n_samples = len(candidate_X)
    surrogate = search.surrogate
    assert surrogate is not None
    proba = np.asarray(surrogate.predict_proba(candidate_X))[:, 1]
    similarity_matrix = search.compute_leaf_similarity(surrogate, candidate_X)
    selected_indices: list[int] = []
    remaining_indices = list(range(n_samples))
    scores = np.zeros(n_samples)

    for rank in range(n_samples):
        if selected_indices:
            mean_similarities = np.zeros(len(remaining_indices))
            for i, idx in enumerate(remaining_indices):
                similarities_to_selected = similarity_matrix[idx, selected_indices]
                mean_similarities[i] = np.mean(similarities_to_selected)
            ranked_scores = (
                proba[remaining_indices] - search.similarity_penalty * mean_similarities
            )
        else:
            ranked_scores = proba[remaining_indices]

        best_local_idx = int(np.argmax(ranked_scores))
        best_global_idx = remaining_indices[best_local_idx]
        scores[best_global_idx] = rank
        selected_indices.append(best_global_idx)
        remaining_indices.remove(best_global_idx)

    ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: item[1])[
        :n_sorted
    ]
    return [member for member, _ in ranked]


def legacy_select_config_subset_single(
    data: Any,
    target: Any,
) -> tuple[list[int], dict[str, float]]:
    n_shapes, n_configs = data.timings.shape
    best_per_shape = np.min(data.timings, axis=1)
    selected_indices: list[int] = []
    satisfied = np.zeros(n_shapes, dtype=bool)
    current_best = np.full(n_shapes, np.inf)

    while len(selected_indices) < target.max_configs:
        if satisfied.all():
            break

        best_score = -1
        best_config_idx = -1

        for config_idx in range(n_configs):
            if config_idx in selected_indices:
                continue

            new_best = np.minimum(current_best, data.timings[:, config_idx])
            slowdowns = new_best / best_per_shape

            if target.goal_type == "max_slowdown":
                score = np.sum(slowdowns <= target.threshold)
            elif target.goal_type == "geomean_slowdown":
                score = np.sum(
                    np.exp(np.mean(np.log(slowdowns + 1e-10))) <= target.threshold
                )
            else:
                score = np.sum(np.mean(slowdowns) <= target.threshold)

            if score > best_score:
                best_score = score
                best_config_idx = config_idx

        if best_config_idx == -1:
            break

        selected_indices.append(best_config_idx)
        current_best = np.minimum(current_best, data.timings[:, best_config_idx])

        slowdowns = current_best / best_per_shape
        if target.goal_type == "max_slowdown":
            satisfied = slowdowns <= target.threshold
        elif target.goal_type == "geomean_slowdown":
            geomean = np.exp(np.mean(np.log(slowdowns + 1e-10)))
            satisfied[:] = geomean <= target.threshold
        else:
            avg = np.mean(slowdowns)
            satisfied[:] = avg <= target.threshold

    slowdowns = current_best / best_per_shape
    stats = {
        "max_slowdown": float(np.max(slowdowns)),
        "geomean_slowdown": float(np.exp(np.mean(np.log(slowdowns + 1e-10)))),
        "avg_slowdown": float(np.mean(slowdowns)),
        "satisfied_ratio": float(np.mean(satisfied)),
        "num_configs": len(selected_indices),
    }
    return selected_indices, stats


def legacy_compact_shape(
    api: RepoApi,
    dispatch: Any,
    shapes: list[int],
) -> list[Any]:
    compacted_shapes = []
    for idx, shape in enumerate(shapes):
        block_idx = api.CompileEnvironment.current().resolve_block_id(shape)
        if block_idx is None:
            shape_str = dispatch._get_shape_string(shape)
            compacted_shapes.append(api.CompactedShape(shape_str, [idx], []))
        else:
            strategy = dispatch.block_id_to_strategy.get((block_idx,))
            if strategy is None:
                strategy = next(
                    (
                        candidate
                        for candidate in dispatch.strategies
                        if block_idx in candidate.block_ids
                    ),
                    None,
                )
            if strategy is not None:
                block_size = strategy.block_size_var(block_idx)
            else:
                block_size = api.DeviceFunction.current().block_size_var(block_idx)
            if block_size is None:
                block_size = "1"
            compacted_shapes.append(
                api.CompactedShape(block_size, [idx], [block_idx])
            )
    for strategy in dispatch.strategies:
        compacted_shapes = strategy.compact_shape(compacted_shapes)
    return compacted_shapes


def _stats_match(lhs: dict[str, float], rhs: dict[str, float]) -> bool:
    return all(
        math.isclose(lhs[key], rhs[key], rel_tol=1e-12, abs_tol=1e-12) for key in lhs
    )


def make_lfbo_case(
    api: RepoApi,
    *,
    seed: int,
    n_candidates: int,
    n_trees: int,
    n_sorted: int,
    similarity_penalty: float,
) -> tuple[Any, list[SimpleNamespace], int, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    proba = rng.random(n_candidates)
    leaf_indices = rng.integers(
        0,
        max(n_candidates // 4, 2),
        size=(n_candidates, n_trees),
        dtype=np.int64,
    )

    search = api.LFBOPatternSearch.__new__(api.LFBOPatternSearch)
    search.config_gen = SimpleNamespace(encode_config=lambda flat: [flat[0]])
    search.similarity_penalty = similarity_penalty
    search.log = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    search.surrogate = MockSurrogate(proba, leaf_indices)
    candidates = [
        SimpleNamespace(index=i, flat_values=[i]) for i in range(n_candidates)
    ]
    meta = {
        "candidates": n_candidates,
        "trees": n_trees,
        "selected": n_sorted,
        "similarity_penalty": similarity_penalty,
    }
    return search, candidates, n_sorted, meta


def make_subset_case(
    api: RepoApi,
    *,
    seed: int,
    n_shapes: int,
    n_configs: int,
    invalid_rate: float,
    goal_type: str,
    threshold: float,
    max_configs: int,
) -> tuple[Any, Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    timings = rng.uniform(1.0, 25.0, size=(n_shapes, n_configs))
    invalid_mask = rng.random((n_shapes, n_configs)) < invalid_rate
    timings[invalid_mask] = np.inf

    for shape_idx in range(n_shapes):
        if not np.isfinite(timings[shape_idx]).any():
            config_idx = int(rng.integers(0, n_configs))
            timings[shape_idx, config_idx] = float(rng.uniform(1.0, 25.0))

    data = api.ShapeConfigData(
        kernel_name=f"synthetic_{goal_type}",
        shape_features=[{"shape_id": i} for i in range(n_shapes)],
        timings=timings,
        configs=[api.Config(block_sizes=[i + 1]) for i in range(n_configs)],
        shape_hashes=[f"s{i}" for i in range(n_shapes)],
        config_hashes=[f"c{i}" for i in range(n_configs)],
    )
    target = api.PerformanceTarget(
        goal_type=goal_type,
        threshold=threshold,
        max_configs=max_configs,
        verbose=False,
    )
    meta = {
        "shapes": n_shapes,
        "configs": n_configs,
        "invalid_rate": invalid_rate,
        "goal_type": goal_type,
        "threshold": threshold,
        "max_configs": max_configs,
    }
    return data, target, meta


def make_tile_dispatch_case(
    api: RepoApi,
    *,
    n_strategies: int,
    repeats_per_block: int,
) -> tuple[Any, list[int], dict[str, Any]]:
    dispatch = api.TileStrategyDispatch.__new__(api.TileStrategyDispatch)
    dispatch.strategies = [
        MockTileStrategy([2 * idx, 2 * idx + 1]) for idx in range(n_strategies)
    ]
    dispatch.block_id_to_strategy = {
        tuple(strategy.block_ids): strategy for strategy in dispatch.strategies
    }
    dispatch._block_id_to_any_strategy = {
        block_id: strategy
        for strategy in dispatch.strategies
        for block_id in strategy.block_ids
    }
    shapes = [
        block_id
        for block_id in range(2 * n_strategies)
        for _ in range(repeats_per_block)
    ]
    meta = {
        "strategies": n_strategies,
        "probes": len(shapes),
        "repeats_per_block": repeats_per_block,
    }
    return dispatch, shapes, meta


def benchmark_lfbo_case(
    api: RepoApi,
    name: str,
    *,
    repeat: int,
    seed: int,
    n_candidates: int,
    n_trees: int,
    n_sorted: int,
    similarity_penalty: float,
) -> BenchmarkResult:
    search, candidates, n_sorted, meta = make_lfbo_case(
        api,
        seed=seed,
        n_candidates=n_candidates,
        n_trees=n_trees,
        n_sorted=n_sorted,
        similarity_penalty=similarity_penalty,
    )
    expected = legacy_surrogate_select(search, candidates, n_sorted)
    actual = search._surrogate_select(candidates, n_sorted)
    legacy_ms = _time_ms(
        lambda: legacy_surrogate_select(search, candidates, n_sorted), repeat
    )
    current_ms = _time_ms(lambda: search._surrogate_select(candidates, n_sorted), repeat)
    parity_ok = [c.index for c in expected] == [c.index for c in actual]
    return BenchmarkResult(
        name=name,
        parity_ok=parity_ok,
        legacy_ms=legacy_ms,
        current_ms=current_ms,
        speedup=legacy_ms / current_ms if current_ms else float("inf"),
        metadata=meta,
    )


def benchmark_subset_case(
    api: RepoApi,
    name: str,
    *,
    repeat: int,
    seed: int,
    n_shapes: int,
    n_configs: int,
    invalid_rate: float,
    goal_type: str,
    threshold: float,
    max_configs: int,
) -> BenchmarkResult:
    data, target, meta = make_subset_case(
        api,
        seed=seed,
        n_shapes=n_shapes,
        n_configs=n_configs,
        invalid_rate=invalid_rate,
        goal_type=goal_type,
        threshold=threshold,
        max_configs=max_configs,
    )
    expected_selected, expected_stats = legacy_select_config_subset_single(data, target)
    actual_selected, actual_stats = api.select_config_subset_single(data, target)
    legacy_ms = _time_ms(
        lambda: legacy_select_config_subset_single(data, target), repeat
    )
    current_ms = _time_ms(lambda: api.select_config_subset_single(data, target), repeat)
    parity_ok = expected_selected == actual_selected and _stats_match(
        expected_stats, actual_stats
    )
    return BenchmarkResult(
        name=name,
        parity_ok=parity_ok,
        legacy_ms=legacy_ms,
        current_ms=current_ms,
        speedup=legacy_ms / current_ms if current_ms else float("inf"),
        metadata=meta,
    )


def benchmark_tile_dispatch_case(
    api: RepoApi,
    name: str,
    *,
    repeat: int,
    n_strategies: int,
    repeats_per_block: int,
) -> BenchmarkResult:
    dispatch, shapes, meta = make_tile_dispatch_case(
        api,
        n_strategies=n_strategies,
        repeats_per_block=repeats_per_block,
    )
    env = SimpleNamespace(
        get_block_id=lambda shape: shape,
        resolve_block_id=lambda shape: shape,
    )
    with patch(
        "helion._compiler.tile_dispatch.CompileEnvironment.current",
        return_value=env,
    ):
        expected = legacy_compact_shape(api, dispatch, shapes)
        actual = dispatch._compact_shape(shapes)
        legacy_ms = _time_ms(lambda: legacy_compact_shape(api, dispatch, shapes), repeat)
        current_ms = _time_ms(lambda: dispatch._compact_shape(shapes), repeat)

    parity_ok = expected == actual
    return BenchmarkResult(
        name=name,
        parity_ok=parity_ok,
        legacy_ms=legacy_ms,
        current_ms=current_ms,
        speedup=legacy_ms / current_ms if current_ms else float("inf"),
        metadata=meta,
    )


def run_all_cases(
    repeat: int = 10,
    repo_root: Path | None = None,
) -> list[BenchmarkResult]:
    api = load_repo_api(repo_root or DEFAULT_REPO_ROOT)
    return [
        benchmark_lfbo_case(
            api,
            "lfbo_medium",
            repeat=repeat,
            seed=0,
            n_candidates=192,
            n_trees=64,
            n_sorted=16,
            similarity_penalty=0.35,
        ),
        benchmark_lfbo_case(
            api,
            "lfbo_large",
            repeat=repeat,
            seed=1,
            n_candidates=384,
            n_trees=96,
            n_sorted=24,
            similarity_penalty=0.35,
        ),
        benchmark_subset_case(
            api,
            "subset_max",
            repeat=repeat,
            seed=2,
            n_shapes=96,
            n_configs=192,
            invalid_rate=0.35,
            goal_type="max_slowdown",
            threshold=1.10,
            max_configs=8,
        ),
        benchmark_subset_case(
            api,
            "subset_geomean",
            repeat=repeat,
            seed=3,
            n_shapes=96,
            n_configs=192,
            invalid_rate=0.35,
            goal_type="geomean_slowdown",
            threshold=1.10,
            max_configs=8,
        ),
        benchmark_subset_case(
            api,
            "subset_avg",
            repeat=repeat,
            seed=4,
            n_shapes=96,
            n_configs=192,
            invalid_rate=0.35,
            goal_type="avg_slowdown",
            threshold=1.10,
            max_configs=8,
        ),
        benchmark_tile_dispatch_case(
            api,
            "tile_dispatch_fallback",
            repeat=repeat,
            n_strategies=256,
            repeats_per_block=8,
        ),
    ]


def _print_results(results: list[BenchmarkResult]) -> None:
    header = f"{'case':<22} {'parity':<8} {'legacy_ms':>12} {'current_ms':>12} {'speedup':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        parity = "ok" if result.parity_ok else "FAIL"
        print(
            f"{result.name:<22} {parity:<8} {result.legacy_ms:>12.4f} {result.current_ms:>12.4f} {result.speedup:>10.2f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parity and microbenchmark harness for autotuner hot paths."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of timing repetitions per case (default: 10).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repo root to benchmark (default: current checkout).",
    )
    args = parser.parse_args()

    results = run_all_cases(repeat=args.repeat, repo_root=args.repo_root)
    _print_results(results)

    if args.json is not None:
        args.json.write_text(json.dumps([asdict(result) for result in results], indent=2))

    return 0 if all(result.parity_ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
