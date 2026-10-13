#!/usr/bin/env python3
"""离线评测 AIOps 工具轨迹。

金标记标准动作：必须出现的工具组、参数子串、禁止工具、报告里必须出现或不能出现的文字。
同一题可重复跑，统计轨迹通过率和 Executor token。

示例：
  python scripts/eval_aiops.py --gold eval/gold/aiops_trajectory.jsonl
  python scripts/eval_aiops.py --gold eval/gold/aiops_trajectory.jsonl --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class ArgSpec:
    """单条参数断言。"""

    tool: str
    arg: str
    contains: str
    mode: str


@dataclass
class TrajectorySample:
    """一条轨迹金标。"""

    id: str
    task: str
    type: str = "procedure"
    required_any: list[list[str]] = field(default_factory=list)
    args: list[ArgSpec] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    report_any: list[str] = field(default_factory=list)
    report_forbid: list[str] = field(default_factory=list)
    max_calls: int = 12
    max_repeat: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any], line_no: int) -> TrajectorySample:
        task = str(raw.get("task") or raw.get("question") or "").strip()
        if not task:
            raise ValueError(f"第 {line_no} 行缺少 task")
        groups = []
        for group in raw.get("required_any") or []:
            if isinstance(group, str):
                groups.append([group])
            elif isinstance(group, list):
                groups.append([str(name) for name in group if str(name).strip()])
        specs: list[ArgSpec] = []
        for item in raw.get("args") or []:
            if not isinstance(item, dict):
                continue
            specs.append(
                ArgSpec(
                    tool=str(item.get("tool") or ""),
                    arg=str(item.get("arg") or ""),
                    contains=str(item.get("contains") or ""),
                    mode=str(item.get("mode") or "required"),
                )
            )
        return cls(
            id=str(raw.get("id") or f"line-{line_no}"),
            task=task,
            type=str(raw.get("type") or "procedure"),
            required_any=groups,
            args=specs,
            forbidden_tools=[str(name) for name in (raw.get("forbidden_tools") or [])],
            report_any=[str(text) for text in (raw.get("report_any") or [])],
            report_forbid=[str(text) for text in (raw.get("report_forbid") or [])],
            max_calls=int(raw.get("max_calls") or 12),
            max_repeat=int(raw.get("max_repeat") or 2),
        )


def load_gold(path: Path, limit: int | None = None) -> list[TrajectorySample]:
    """读取轨迹金标。"""
    if not path.is_file():
        raise FileNotFoundError(f"金标文件不存在: {path}")
    samples: list[TrajectorySample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            samples.append(TrajectorySample.from_dict(raw, line_no))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"金标文件为空: {path}")
    return samples


def flatten_calls(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按执行顺序展开每步里的工具调用。"""
    calls: list[dict[str, Any]] = []
    for step in trace:
        for call in step.get("calls") or []:
            if isinstance(call, dict) and call.get("name"):
                calls.append(call)
    return calls


def _arg_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _max_repeat(calls: list[dict[str, Any]]) -> int:
    counts: dict[str, int] = defaultdict(int)
    for call in calls:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        key = f"{call.get('name')}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        counts[key] += 1
    return max(counts.values(), default=0)


def _arg_ok(spec: ArgSpec, calls: list[dict[str, Any]], task: str) -> bool:
    matched = [call for call in calls if call.get("name") == spec.tool]
    needle = spec.contains.lower()

    def hit(call: dict[str, Any]) -> bool:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        return needle in _arg_text(args.get(spec.arg)).lower()

    if spec.mode == "required":
        return any(hit(call) for call in matched)
    if spec.mode == "if_called":
        return all(hit(call) for call in matched) if matched else True
    if spec.mode == "forbidden":
        return all(not hit(call) for call in matched)
    if spec.mode == "grounded":
        task_text = task.lower()
        for call in matched:
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            value = _arg_text(args.get(spec.arg)).strip()
            if value and value.lower() not in task_text:
                return False
        return True
    return False


def score_trace(
    sample: TrajectorySample,
    trace: list[dict[str, Any]],
    report: str,
) -> dict[str, Any]:
    """用规则断言轨迹。passed 为 1 表示工具、参数、步数和报告约束都满足。"""
    calls = flatten_calls(trace)
    names = [str(call.get("name") or "") for call in calls]
    group_hits = [
        any(name in names for name in group) if group else True for group in sample.required_any
    ]
    arg_hits = [_arg_ok(spec, calls, sample.task) for spec in sample.args]
    forbidden = [name for name in sample.forbidden_tools if name in names]
    repeat = _max_repeat(calls)
    report_hit = (
        any(piece in (report or "") for piece in sample.report_any) if sample.report_any else True
    )
    report_clean = all(piece not in (report or "") for piece in sample.report_forbid)
    within_max = len(calls) <= sample.max_calls
    no_spin = repeat <= sample.max_repeat
    passed = (
        all(group_hits)
        and all(arg_hits)
        and not forbidden
        and within_max
        and no_spin
        and report_hit
        and report_clean
    )
    prompt = sum(int(step.get("prompt_tokens") or 0) for step in trace)
    completion = sum(int(step.get("completion_tokens") or 0) for step in trace)
    return {
        "passed": 1.0 if passed else 0.0,
        "tool_groups": group_hits,
        "args": arg_hits,
        "forbidden": forbidden,
        "call_count": len(calls),
        "max_repeat": repeat,
        "within_max": within_max,
        "no_spin": no_spin,
        "report_hit": report_hit,
        "report_clean": report_clean,
        "calls": [{"name": call.get("name"), "args": call.get("args")} for call in calls],
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _callback_tokens(handler: Any) -> tuple[int, int]:
    usage = getattr(handler, "usage_metadata", None) or {}
    prompt = 0
    completion = 0
    if isinstance(usage, dict):
        for item in usage.values():
            if isinstance(item, dict):
                prompt += int(item.get("input_tokens") or 0)
                completion += int(item.get("output_tokens") or 0)
            else:
                prompt += int(getattr(item, "input_tokens", 0) or 0)
                completion += int(getattr(item, "output_tokens", 0) or 0)
    return prompt, completion


async def run_once(task: str, session_id: str) -> tuple[str, list[dict[str, Any]], int, int]:
    """跑一轮 Plan-Execute-Replan，返回报告、轨迹和 token。"""
    from app.services.aiops_service import aiops_service

    handler = None
    try:
        from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

        handler = UsageMetadataCallbackHandler()
    except Exception:
        handler = None

    # MemorySaver 按 thread_id 隔离；评测每次换新会话，避免上一条轨迹串进来。
    config = {"configurable": {"thread_id": session_id}}
    if handler is not None:
        config["callbacks"] = [handler]

    report = ""
    trace: list[dict[str, Any]] = []
    async for event in aiops_service.graph.astream(
        input={
            "input": task,
            "plan": [],
            "past_steps": [],
            "tool_trace": [],
            "response": "",
        },
        config=config,
        stream_mode="updates",
    ):
        for node_output in event.values():
            if isinstance(node_output, dict) and node_output.get("response"):
                report = str(node_output.get("response") or "")

    final_state = aiops_service.graph.get_state(config)
    if final_state and final_state.values:
        report = str(final_state.values.get("response") or report)
        trace = list(final_state.values.get("tool_trace") or [])

    prompt, completion = _callback_tokens(handler) if handler is not None else (0, 0)
    return report, trace, prompt, completion


def write_bad_cases(path: Path, rows: list[dict[str, Any]]) -> int:
    """追加没稳定通过的轨迹题。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    stamped = datetime.now().isoformat(timespec="seconds")
    bad: list[dict[str, Any]] = []
    for sample_id, items in grouped.items():
        rate = _mean([float(item["passed"]) for item in items])
        if rate >= 1:
            continue
        last = items[-1]
        bad.append(
            {
                "ts": stamped,
                "source": "aiops",
                "id": sample_id,
                "task": last.get("task"),
                "type": last.get("type"),
                "reasons": ["轨迹未稳定通过"],
                "pass_rate": round(rate, 4),
                "calls": last.get("calls"),
            }
        )
    if not bad:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in bad:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(bad)


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_gold(Path(args.gold), limit=args.limit)
    runs = int(args.runs)
    if runs < 1:
        raise ValueError("--runs 至少为 1")
    type_counts: dict[str, int] = {}
    for sample in samples:
        type_counts[sample.type] = type_counts.get(sample.type, 0) + 1
    print(f"轨迹金标 {len(samples)} 条 | 题型={type_counts} | runs={runs}")

    reports: list[dict[str, Any]] = []
    schedule = [(run, sample) for run in range(1, runs + 1) for sample in samples]
    for index, (run_index, sample) in enumerate(schedule, start=1):
        prefix = f"[{index}/{len(schedule)}]"
        if runs > 1:
            prefix += f" run {run_index}/{runs}"
        print(f"{prefix} {sample.id}: {sample.task}")
        started = time.perf_counter()
        session_id = f"eval-{sample.id}-r{run_index}-{uuid.uuid4().hex[:8]}"
        report, trace, graph_prompt, graph_completion = await run_once(sample.task, session_id)
        scored = score_trace(sample, trace, report)
        if graph_prompt or graph_completion:
            scored["prompt_tokens"] = graph_prompt
            scored["completion_tokens"] = graph_completion
            scored["token_scope"] = "graph"
        else:
            scored["token_scope"] = "executor"
        latency_ms = (time.perf_counter() - started) * 1000.0
        print(
            f"  passed={int(scored['passed'])} calls={scored['call_count']} "
            f"repeat={scored['max_repeat']} "
            f"tokens={scored['prompt_tokens'] + scored['completion_tokens']} "
            f"{latency_ms:.0f}ms"
        )
        reports.append(
            {
                "id": sample.id,
                "run": run_index,
                "task": sample.task,
                "type": sample.type,
                "latency_ms": round(latency_ms, 1),
                "report": report,
                **scored,
            }
        )

    by_id: dict[str, list[float]] = defaultdict(list)
    for row in reports:
        by_id[row["id"]].append(float(row["passed"]))
    rates = [_mean(values) for values in by_id.values()]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in reports)
    completion_tokens = sum(int(row["completion_tokens"]) for row in reports)
    total_tokens = prompt_tokens + completion_tokens
    observations = len(reports)
    summary = {
        "samples": len(samples),
        "runs": runs,
        "pass_rate": round(_mean(rates), 4) if rates else None,
        "always_pass": sum(1 for rate in rates if rate == 1),
        "unstable": sum(1 for rate in rates if 0 < rate < 1),
        "always_fail": sum(1 for rate in rates if rate == 0),
        "avg_calls": round(_mean([float(row["call_count"]) for row in reports]), 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_run": round(total_tokens / observations, 1) if observations else 0,
        "token_scope": reports[-1]["token_scope"] if reports else "",
    }
    result = {
        "config": {
            "gold": str(Path(args.gold)),
            "runs": runs,
            "types": type_counts,
            "sample_count": len(samples),
        },
        "summary": summary,
        "samples": reports,
    }

    bad_path = (
        Path(args.bad_cases) if args.bad_cases else _ROOT / "eval" / "reports" / "bad_cases.jsonl"
    )
    bad_count = write_bad_cases(bad_path, reports)
    result["bad_cases"] = {"path": str(bad_path), "count": bad_count}

    print("\n=== 轨迹 ===")
    print(
        f"通过率={summary['pass_rate']}  全通过={summary['always_pass']}  "
        f"会飘={summary['unstable']}  全失败={summary['always_fail']}  "
        f"平均调用={summary['avg_calls']}  "
        f"token/轮≈{summary['avg_tokens_per_run']} ({summary['token_scope']})"
    )
    if bad_count:
        print(f"错题 {bad_count} 条已追加到 {bad_path}")

    output = Path(args.output) if args.output else _ROOT / "eval" / "reports" / "aiops_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线评测 AIOps 工具轨迹")
    parser.add_argument(
        "--gold",
        default=str(_ROOT / "eval" / "gold" / "aiops_trajectory.jsonl"),
        help="轨迹金标 JSONL",
    )
    parser.add_argument("--output", default="", help="报告路径，默认 eval/reports/aiops_eval.json")
    parser.add_argument("--runs", type=int, default=1, help="每题重复次数，稳定性用 5")
    parser.add_argument("--limit", type=int, default=None, help="只评前 N 条")
    parser.add_argument("--bad-cases", default="", help="错题追加路径")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(run_eval(args))
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
