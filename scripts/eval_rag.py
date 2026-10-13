#!/usr/bin/env python3
"""离线 RAG 评测脚本。

检索：直接调用混合检索 / 向量检索 / 重排，计算 Hit@K、Recall@K、Precision@K、MRR、nDCG@K。
问答：用重排后的上下文生成答案，再用 LLM 评忠实度与正确性。

金标 JSONL 字段：
  id, question
  gold_chunk_ids  可选，Milvus 主键（重索引后会变）
  gold_contains   可选，命中文本须包含的片段（推荐）
  gold_files      可选，来源文件名；无 contains/ids 时才作为召回金标
  gold_answer     可选，用于正确性
  answerable      默认 true
  should_retrieve 默认 true
  type            可选分类

示例：
  python scripts/eval_rag.py --gold eval/gold/sample.jsonl
  python scripts/eval_rag.py --gold eval/gold/sample.jsonl --stage retrieve
  python scripts/eval_rag.py --gold eval/gold/sample.jsonl --qa-mode agent
  python scripts/eval_rag.py --gold eval/gold/sample.jsonl --stage rerank --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# 金标
# ---------------------------------------------------------------------------


@dataclass
class GoldSample:
    """单条评测样本。"""

    id: str
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_contains: list[str] = field(default_factory=list)
    gold_files: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    answerable: bool = True
    should_retrieve: bool = True
    type: str = "fact"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], line_no: int) -> GoldSample:
        question = str(raw.get("question") or raw.get("query") or "").strip()
        if not question:
            raise ValueError(f"第 {line_no} 行缺少 question")
        sample_id = str(raw.get("id") or f"line-{line_no}")
        return cls(
            id=sample_id,
            question=question,
            gold_chunk_ids=_as_str_list(raw.get("gold_chunk_ids")),
            gold_contains=_as_str_list(raw.get("gold_contains")),
            gold_files=_as_str_list(raw.get("gold_files")),
            gold_answer=(str(raw["gold_answer"]) if raw.get("gold_answer") else None),
            answerable=bool(raw.get("answerable", True)),
            should_retrieve=bool(raw.get("should_retrieve", True)),
            type=str(raw.get("type") or "fact"),
        )


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def load_gold(path: Path, limit: int | None = None) -> list[GoldSample]:
    """读取 JSONL 金标。"""
    if not path.is_file():
        raise FileNotFoundError(f"金标文件不存在: {path}")
    samples: list[GoldSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            samples.append(GoldSample.from_dict(raw, line_no))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"金标文件为空: {path}")
    return samples


# ---------------------------------------------------------------------------
# 文档匹配
# ---------------------------------------------------------------------------


def _doc_ids(doc: Any) -> set[str]:
    meta = getattr(doc, "metadata", None) or {}
    ids: set[str] = set()
    for key in ("id", "pk"):
        if meta.get(key) not in (None, ""):
            ids.add(str(meta[key]))
    return ids


def _doc_files(doc: Any) -> list[str]:
    meta = getattr(doc, "metadata", None) or {}
    values = [meta.get("_file_name"), meta.get("_source"), meta.get("source")]
    return [str(v) for v in values if v]


def gold_keys(sample: GoldSample) -> list[tuple[str, str]]:
    """金标条目：(kind, value)。优先 id，其次 contains，最后 files。"""
    keys = [("id", v) for v in sample.gold_chunk_ids]
    if keys:
        return keys
    keys = [("contains", v) for v in sample.gold_contains]
    if keys:
        return keys
    return [("file", v) for v in sample.gold_files]


def _file_hit(doc: Any, needle: str) -> bool:
    blob = " ".join(_doc_files(doc)).replace("\\", "/").lower()
    name = Path(needle).name.lower()
    return needle.lower() in blob or name in blob


def doc_matches_key(doc: Any, kind: str, value: str) -> bool:
    content = getattr(doc, "page_content", None) or ""
    if kind == "id":
        return value in _doc_ids(doc)
    if kind == "contains":
        return value in content
    if kind == "file":
        return _file_hit(doc, value)
    return False


def doc_is_relevant(doc: Any, sample: GoldSample) -> bool:
    return any(doc_matches_key(doc, kind, value) for kind, value in gold_keys(sample))


def serialize_doc(doc: Any, max_chars: int = 180) -> dict[str, str]:
    meta = getattr(doc, "metadata", None) or {}
    content = getattr(doc, "page_content", None) or ""
    snippet = content[:max_chars] + ("..." if len(content) > max_chars else "")
    return {
        "id": str(meta.get("id") or meta.get("pk") or ""),
        "file": str(meta.get("_file_name") or meta.get("_source") or ""),
        "snippet": snippet,
    }


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def metrics_at_k(docs: Sequence[Any], sample: GoldSample, k: int) -> dict[str, float]:
    """对单条样本计算 @k 指标；无金标时返回空 dict。"""
    keys = gold_keys(sample)
    if not keys:
        return {}
    ranked = list(docs[:k])
    binary = [1 if doc_is_relevant(doc, sample) else 0 for doc in ranked]
    hit_keys = 0
    for kind, value in keys:
        if any(doc_matches_key(doc, kind, value) for doc in ranked):
            hit_keys += 1
    first_rank = next((i + 1 for i, flag in enumerate(binary) if flag), None)
    idcg = _dcg([1] * min(k, len(keys)))
    return {
        "hit": 1.0 if any(binary) else 0.0,
        "recall": hit_keys / len(keys),
        "precision": (sum(binary) / k) if k else 0.0,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg": (_dcg(binary) / idcg) if idcg else 0.0,
        "empty": 1.0 if not ranked else 0.0,
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((p / 100.0) * (len(xs) - 1))))
    return xs[idx]


# ---------------------------------------------------------------------------
# LLM 生成与裁判
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_REFUSAL_HINTS = (
    "不知道",
    "无法确定",
    "无法回答",
    "没有找到",
    "未找到",
    "资料不足",
    "知识库无法",
    "不足以回答",
    "无法从",
)

GEN_SYSTEM = (
    "你是运维知识助手。只根据给定参考资料回答用户问题。"
    "资料不足以回答时，明确说「根据现有知识库无法确定」，不要编造。"
    "回答简洁、专业，使用中文。"
)

FAITH_SYSTEM = (
    "你是 RAG 忠实度评委。根据【参考资料】判断【回答】中的事实是否都能被资料支撑。"
    "忽略礼貌用语。若回答是拒答且资料确实不够，忠实度给 1。"
    '只输出 JSON：{"faithfulness": 0到1的小数, "unsupported": ["不被支持的句子"], "reason": "一句话原因"}'
)

CORR_SYSTEM = (
    "你是问答正确性评委。允许表述不同，但关键事实必须与【标准答案】一致。"
    "若 answerable=false：承认知识库没有该信息、拒绝编造，得 1；"
    "顺带引用手册里真实存在的相邻事实、但没有补造缺失内容，仍得 1；"
    "编造了标准答案里明确没有的号码、配置、命令或步骤，得 0。"
    '只输出 JSON：{"correctness": 0到1的小数, "reason": "一句话原因"}'
)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _JSON_FENCE.sub("", (text or "").strip()).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def looks_like_refusal(answer: str) -> bool:
    return any(hint in (answer or "") for hint in _REFUSAL_HINTS)


def _message_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(response).strip()


def _usage_from_response(response: Any) -> tuple[int, int]:
    """读取聊天模型的 prompt / completion token。缺失时为 0。"""
    meta = getattr(response, "usage_metadata", None) or {}
    if not isinstance(meta, dict):
        prompt = int(getattr(meta, "input_tokens", 0) or 0)
        completion = int(getattr(meta, "output_tokens", 0) or 0)
        return prompt, completion
    if meta.get("input_tokens") or meta.get("output_tokens"):
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    raw = getattr(response, "response_metadata", None) or {}
    usage = raw.get("token_usage") or raw.get("usage") or {} if isinstance(raw, dict) else {}
    if not isinstance(usage, dict):
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt, completion
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


class LlmHelper:
    """封装生成与裁判，失败时降级为规则分。"""

    def __init__(self, llm: Any):
        self.llm = llm
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def invoke(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        prompt, completion = _usage_from_response(response)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        return _message_text(response)

    def generate(self, question: str, context: str) -> str:
        user = f"【参考资料】\n{context or '（无）'}\n\n【问题】\n{question}"
        return self.invoke(GEN_SYSTEM, user)

    def judge_faithfulness(self, answer: str, context: str) -> dict[str, Any]:
        user = f"【参考资料】\n{context[:8000] or '（无）'}\n\n【回答】\n{answer}"
        try:
            data = parse_json_object(self.invoke(FAITH_SYSTEM, user))
            score = float(data.get("faithfulness"))
            score = min(1.0, max(0.0, score))
            return {
                "faithfulness": score,
                "unsupported": data.get("unsupported") or [],
                "reason": str(data.get("reason") or ""),
            }
        except (TypeError, ValueError):
            fallback = 1.0 if looks_like_refusal(answer) and not context.strip() else 0.0
            return {
                "faithfulness": fallback,
                "unsupported": [],
                "reason": "裁判解析失败，已降级",
            }

    def judge_correctness(self, sample: GoldSample, answer: str) -> dict[str, Any]:
        if sample.gold_answer is None and sample.answerable:
            return {"correctness": None, "reason": "无标准答案，跳过正确性"}
        gold = sample.gold_answer or "根据现有知识库无法确定。"
        user = (
            f"answerable={str(sample.answerable).lower()}\n"
            f"【标准答案】\n{gold}\n\n【模型回答】\n{answer}"
        )
        try:
            data = parse_json_object(self.invoke(CORR_SYSTEM, user))
            score = float(data.get("correctness"))
            score = min(1.0, max(0.0, score))
            return {"correctness": score, "reason": str(data.get("reason") or "")}
        except (TypeError, ValueError):
            if not sample.answerable:
                score = 1.0 if looks_like_refusal(answer) else 0.0
            else:
                score = 1.0 if (gold and gold in answer) else 0.0
            return {"correctness": score, "reason": "裁判解析失败，已降级"}


# ---------------------------------------------------------------------------
# 检索通道
# ---------------------------------------------------------------------------


def retrieve_channel(
    channel: str,
    query: str,
    recall_k: int,
    rerank_n: int,
    hybrid: Any,
    rerank: Any,
    store: Any,
) -> tuple[list[Any], float]:
    """返回 (文档列表, 耗时毫秒)。"""
    started = time.perf_counter()
    if channel == "vector":
        docs = store.similarity_search(query, k=recall_k)
    elif channel == "hybrid":
        docs = hybrid.retrieve(query, recall_k=recall_k)
    elif channel == "hybrid_rerank":
        candidates = hybrid.retrieve(query, recall_k=recall_k)
        docs = rerank.rerank(query, candidates, top_n=rerank_n)
    else:
        raise ValueError(f"未知检索通道: {channel}")
    latency_ms = (time.perf_counter() - started) * 1000.0
    return list(docs or []), latency_ms


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------


async def eval_agent_answer(question: str, sample_id: str) -> tuple[str, dict[str, Any] | None]:
    from app.services.rag_agent_service import rag_agent_service
    from app.tools.knowledge_tool import clear_last_rag_result, get_last_rag_result

    clear_last_rag_result()
    session_id = f"eval-{sample_id}-{uuid.uuid4().hex[:8]}"
    result = await rag_agent_service.query(question, session_id=session_id)
    answer = ""
    if isinstance(result, dict):
        answer = str(result.get("answer") or "")
        rag_meta = result.get("rag")
    else:
        rag_meta = get_last_rag_result()
    return answer, rag_meta if isinstance(rag_meta, dict) else get_last_rag_result()


def aggregate_channel(per_sample: list[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    latencies = [row["latency_ms"] for row in per_sample]
    out["latency_ms_p50"] = round(percentile(latencies, 50), 1)
    out["latency_ms_p95"] = round(percentile(latencies, 95), 1)
    out["n"] = len(per_sample)
    for k in ks:
        keyed = f"k={k}"
        rows = [row["metrics"].get(keyed) for row in per_sample]
        rows = [m for m in rows if m]
        if not rows:
            out[keyed] = {"n": 0}
            continue
        out[keyed] = {
            "n": len(rows),
            "hit": round(mean([m["hit"] for m in rows]), 4),
            "recall": round(mean([m["recall"] for m in rows]), 4),
            "precision": round(mean([m["precision"] for m in rows]), 4),
            "mrr": round(mean([m["mrr"] for m in rows]), 4),
            "ndcg": round(mean([m["ndcg"] for m in rows]), 4),
            "empty": round(mean([m["empty"] for m in rows]), 4),
        }
    return out


def print_retrieval_table(retrieval: dict[str, Any], ks: Sequence[int]) -> None:
    headers = (
        f"{'通道':<16}{'K':>4}{'Hit@K':>10}{'Recall@K':>12}{'P@K':>10}"
        f"{'MRR':>10}{'nDCG@K':>10}{'空召回':>10}{'n':>6}{'P50ms':>10}"
    )
    print("\n=== 检索质量 ===")
    print(headers)
    print("-" * len(headers))
    for channel, summary in retrieval.items():
        for k in ks:
            row = summary.get(f"k={k}") or {}
            if not row.get("n"):
                continue
            print(
                f"{channel:<16}{k:>4}"
                f"{row['hit']:>10.3f}{row['recall']:>12.3f}{row['precision']:>10.3f}"
                f"{row['mrr']:>10.3f}{row['ndcg']:>10.3f}{row['empty']:>10.3f}"
                f"{row['n']:>6}{summary.get('latency_ms_p50', 0):>10.1f}"
            )


def stability_summary(per_sample: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """同题多次运行的 Hit@K 通过率。1 表示每轮都命中，0 到 1 之间表示结果会飘。"""
    by_id: dict[str, list[float]] = defaultdict(list)
    for row in per_sample:
        metrics = (row.get("metrics") or {}).get(f"k={k}") or {}
        if "hit" not in metrics:
            continue
        by_id[str(row.get("id") or "")].append(float(metrics["hit"]))
    rates = [mean(values) for values in by_id.values()]
    run_count = max((len(values) for values in by_id.values()), default=0)
    return {
        "k": k,
        "samples": len(rates),
        "runs": run_count,
        "pass_rate": round(mean(rates), 4) if rates else None,
        "always_pass": sum(1 for rate in rates if rate == 1),
        "unstable": sum(1 for rate in rates if 0 < rate < 1),
        "always_fail": sum(1 for rate in rates if rate == 0),
    }


def answer_stability(qa_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """问答稳定性：可答题看正确性是否达到 0.6，拒答题看是否拒答。"""
    by_id: dict[str, list[float]] = defaultdict(list)
    for row in qa_rows:
        if row.get("refusal_ok") is not None:
            flag = float(row["refusal_ok"])
        elif row.get("correctness") is not None:
            flag = 1.0 if float(row["correctness"]) >= 0.6 else 0.0
        else:
            continue
        by_id[str(row.get("id") or "")].append(flag)
    rates = [mean(values) for values in by_id.values()]
    run_count = max((len(values) for values in by_id.values()), default=0)
    return {
        "samples": len(rates),
        "runs": run_count,
        "pass_rate": round(mean(rates), 4) if rates else None,
        "unstable": sum(1 for rate in rates if 0 < rate < 1),
    }


def print_stability(summary: dict[str, Any]) -> None:
    if not summary or not summary.get("samples") or summary.get("runs", 0) <= 1:
        return
    print(
        f"\n=== 稳定性（Hit@{summary['k']}，每题 {summary['runs']} 次）===\n"
        f"通过率={_fmt_rate(summary['pass_rate'] or 0)}  "
        f"全通过={summary['always_pass']}  "
        f"会飘={summary['unstable']}  "
        f"全失败={summary['always_fail']}  "
        f"题数={summary['samples']}"
    )


def print_cost(cost: dict[str, Any]) -> None:
    print("\n=== 成本 ===")
    if not cost:
        print("（没有 token 统计）")
        return
    print(
        f"embedding={cost.get('embedding_tokens', 0)}  "
        f"chat_prompt={cost.get('chat_prompt_tokens', 0)}  "
        f"chat_completion={cost.get('chat_completion_tokens', 0)}  "
        f"合计={cost.get('total_tokens', 0)}  "
        f"每题每轮≈{cost.get('avg_tokens_per_run', 0)}  "
        f"观测={cost.get('observations', 0)}"
    )
    note = cost.get("note")
    if note:
        print(note)


def _primary_retrieval(report: dict[str, Any]) -> dict[str, Any]:
    retrieval = report.get("retrieval") or {}
    for name in ("hybrid_rerank", "hybrid", "vector"):
        if name in retrieval:
            return retrieval[name]
    if retrieval:
        return next(iter(retrieval.values()))
    return {}


def write_bad_cases(
    path: Path,
    reports: list[dict[str, Any]],
    k: int,
) -> int:
    """把未稳定命中、拒答失败或正确性不足的题追加到 bad case 库。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report.get("id") or "")].append(report)

    stamped = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for sample_id, items in grouped.items():
        reasons: list[str] = []
        hits: list[float] = []
        correctness: list[float] = []
        refusals: list[float] = []
        for item in items:
            metrics = (_primary_retrieval(item).get("metrics") or {}).get(f"k={k}") or {}
            if "hit" in metrics:
                hits.append(float(metrics["hit"]))
            qa = item.get("qa") or {}
            if qa.get("refusal_ok") is not None:
                refusals.append(float(qa["refusal_ok"]))
            score = (qa.get("correctness") or {}).get("correctness")
            if score is not None and item.get("answerable", True):
                correctness.append(float(score))
        if hits and mean(hits) < 1:
            reasons.append(f"Hit@{k} 未稳定通过")
        if refusals and mean(refusals) < 1:
            reasons.append("拒答失败")
        if correctness and mean(correctness) < 0.6:
            reasons.append("正确性低于 0.6")
        if not reasons:
            continue
        last = items[-1]
        rows.append(
            {
                "ts": stamped,
                "source": "rag",
                "id": sample_id,
                "question": last.get("question"),
                "type": last.get("type"),
                "reasons": reasons,
                "hit_rate": round(mean(hits), 4) if hits else None,
                "correctness": round(mean(correctness), 4) if correctness else None,
                "refusal_accuracy": round(mean(refusals), 4) if refusals else None,
            }
        )
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def print_qa_table(qa: dict[str, Any]) -> None:
    print("\n=== 问答质量 ===")
    if not qa:
        print("（本轮未评问答）")
        return
    print(
        f"忠实度={qa.get('faithfulness', '-')}  "
        f"正确性={qa.get('correctness', '-')}  "
        f"拒答准确率={qa.get('refusal_accuracy', '-')}  "
        f"工具调用覆盖={qa.get('tool_coverage', '-')}  "
        f"n={qa.get('n', 0)}  "
        f"生成P50={qa.get('latency_ms_p50', '-')}ms"
    )


def _lift_pct(new: float, baseline: float) -> float | None:
    """相对基线提升比例；(new-base)/base。基线为 0 时无法计算。"""
    if baseline == 0:
        return None
    return (new - baseline) / baseline


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def _fmt_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_kpi_lift(
    retrieval: dict[str, Any],
    k: int,
    baseline: str = "vector",
) -> dict[str, Any]:
    """相对纯向量，计算召回率、准确率、效率提升。"""
    keyed = f"k={k}"
    base = retrieval.get(baseline) or {}
    base_row = base.get(keyed) or {}
    if not base_row.get("n"):
        return {}

    base_recall = float(base_row["recall"])
    base_precision = float(base_row["precision"])
    base_latency = float(base.get("latency_ms_p50") or 0.0)
    base_eff = (base_precision / (base_latency / 1000.0)) if base_latency else 0.0

    channels: dict[str, Any] = {}
    for name, summary in retrieval.items():
        row = (summary or {}).get(keyed) or {}
        if not row.get("n"):
            continue
        recall = float(row["recall"])
        precision = float(row["precision"])
        latency = float(summary.get("latency_ms_p50") or 0.0)
        efficiency = (precision / (latency / 1000.0)) if latency else 0.0
        # 正值表示比基线更快：(base - new) / base
        latency_delta = _lift_pct(latency, base_latency)
        latency_speedup = None if latency_delta is None else -latency_delta
        channels[name] = {
            "n": row["n"],
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "latency_ms_p50": round(latency, 1),
            "efficiency": round(efficiency, 4),
            "recall_lift": (
                None if name == baseline else _round_lift(_lift_pct(recall, base_recall))
            ),
            "precision_lift": (
                None
                if name == baseline
                else _round_lift(_lift_pct(precision, base_precision))
            ),
            "latency_speedup": (
                None if name == baseline else _round_lift(latency_speedup)
            ),
            "efficiency_lift": (
                None if name == baseline else _round_lift(_lift_pct(efficiency, base_eff))
            ),
        }
    return {
        "baseline": baseline,
        "k": k,
        "channels": channels,
    }


def _round_lift(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def print_kpi_lift(kpis: dict[str, Any]) -> None:
    if not kpis or not kpis.get("channels"):
        print("\n=== 三项指标（相对纯向量）===")
        print("缺少向量基线，请不要加 --no-ablation。")
        return

    k = kpis["k"]
    baseline = kpis["baseline"]
    channels = kpis["channels"]
    order = [baseline] + [name for name in channels if name != baseline]

    print(f"\n=== 三项指标（相对{baseline}，K={k}）===")
    print(
        f"{'通道':<16}{'召回率':>10}{'召回提升':>10}"
        f"{'准确率':>10}{'准确提升':>10}"
        f"{'P50ms':>10}{'时延变化':>10}"
        f"{'效率':>10}{'效率提升':>10}"
    )
    print("-" * 96)
    for name in order:
        row = channels.get(name)
        if not row:
            continue
        print(
            f"{name:<16}"
            f"{_fmt_rate(row['recall']):>10}{_fmt_pct(row['recall_lift']):>10}"
            f"{_fmt_rate(row['precision']):>10}{_fmt_pct(row['precision_lift']):>10}"
            f"{row['latency_ms_p50']:>10.1f}{_fmt_pct(row['latency_speedup']):>10}"
            f"{row['efficiency']:>10.2f}{_fmt_pct(row['efficiency_lift']):>10}"
        )
    print(
        "准确率=Precision@K；效率=准确率/秒（越高越好）；"
        "时延变化正值表示比基线更快。"
    )


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    from app.config import config
    from app.services.hybrid_retriever_service import hybrid_retriever_service
    from app.services.rerank_service import rerank_service
    from app.services.vector_embedding_service import vector_embedding_service
    from app.services.vector_store_manager import vector_store_manager
    from app.tools.knowledge_tool import format_docs

    samples = load_gold(Path(args.gold), limit=args.limit)
    runs = int(args.runs)
    if runs < 1:
        raise ValueError("--runs 至少为 1")
    vector_embedding_service.total_tokens = 0
    type_counts: dict[str, int] = {}
    for sample in samples:
        type_counts[sample.type] = type_counts.get(sample.type, 0) + 1
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    if not ks:
        raise ValueError("--ks 不能为空")
    recall_k = args.recall_k or config.rag_recall_k
    top_k = args.top_k or config.rag_top_k
    rerank_n = max(top_k, max(ks))
    do_retrieve = args.stage in {"retrieve", "rerank", "all"}
    do_qa = args.stage in {"qa", "all"} and args.qa_mode != "off"

    channels = ["hybrid_rerank"]
    if do_retrieve and not args.no_ablation:
        channels = ["vector", "hybrid", "hybrid_rerank"]
    if args.stage == "retrieve":
        channels = [c for c in channels if c != "hybrid_rerank"] or ["hybrid"]
        if "hybrid" not in channels:
            channels.append("hybrid")

    helper: LlmHelper | None = None
    need_llm = do_qa and (args.qa_mode == "grounded" or not args.skip_judge)
    if need_llm:
        from app.core.llm_factory import llm_factory

        helper = LlmHelper(llm_factory.create_chat_model(temperature=0, streaming=False))

    channel_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in channels}
    sample_reports: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    print(
        f"金标 {len(samples)} 条 | 题型={type_counts} | runs={runs} | stage={args.stage} | "
        f"channels={','.join(channels)} | recall_k={recall_k} top_k={top_k} ks={ks} | "
        f"qa_mode={args.qa_mode}"
    )

    schedule = [(run, sample) for run in range(1, runs + 1) for sample in samples]
    for index, (run_index, sample) in enumerate(schedule, start=1):
        prefix = f"[{index}/{len(schedule)}]"
        if runs > 1:
            prefix += f" run {run_index}/{runs}"
        print(f"{prefix} {sample.id}: {sample.question}")
        retrieved: dict[str, list[Any]] = {}
        report: dict[str, Any] = {
            "id": sample.id,
            "run": run_index,
            "question": sample.question,
            "type": sample.type,
            "answerable": sample.answerable,
        }

        if do_retrieve:
            retrieval_report: dict[str, Any] = {}
            for channel in channels:
                docs, latency_ms = retrieve_channel(
                    channel,
                    sample.question,
                    recall_k,
                    rerank_n,
                    hybrid_retriever_service,
                    rerank_service,
                    vector_store_manager,
                )
                retrieved[channel] = docs
                per_k = {f"k={k}": metrics_at_k(docs, sample, k) for k in ks}
                row = {
                    "id": sample.id,
                    "run": run_index,
                    "latency_ms": latency_ms,
                    "metrics": per_k,
                    "empty": 1.0 if not docs else 0.0,
                }
                channel_rows[channel].append(row)
                retrieval_report[channel] = {
                    "latency_ms": round(latency_ms, 1),
                    "count": len(docs),
                    "metrics": {k: v for k, v in per_k.items() if v},
                    "docs": [serialize_doc(d) for d in docs[: max(ks)]],
                }
            report["retrieval"] = retrieval_report

        if do_qa:
            qa_started = time.perf_counter()
            rag_meta: dict[str, Any] | None = None
            context_docs: list[Any] = []
            if args.qa_mode == "agent":
                answer, rag_meta = await eval_agent_answer(sample.question, sample.id)
                tool_called = rag_meta is not None
            else:
                if helper is None:
                    raise RuntimeError("grounded 模式需要 LLM")
                if "hybrid_rerank" in retrieved:
                    context_docs = retrieved["hybrid_rerank"][:top_k]
                else:
                    context_docs, _ = retrieve_channel(
                        "hybrid_rerank",
                        sample.question,
                        recall_k,
                        top_k,
                        hybrid_retriever_service,
                        rerank_service,
                        vector_store_manager,
                    )
                    context_docs = context_docs[:top_k]
                context = format_docs(context_docs) if context_docs else ""
                answer = helper.generate(sample.question, context)
                tool_called = True
            qa_latency = (time.perf_counter() - qa_started) * 1000.0

            context_for_judge = format_docs(context_docs) if context_docs else ""
            if args.qa_mode == "agent" and rag_meta:
                snippets = [
                    str(src.get("snippet") or "")
                    for src in (rag_meta.get("sources") or [])
                    if isinstance(src, dict)
                ]
                context_for_judge = "\n".join(snippets)

            skip_judge = args.skip_judge or helper is None
            faith = (
                {"faithfulness": None, "unsupported": [], "reason": "跳过裁判"}
                if skip_judge
                else helper.judge_faithfulness(answer, context_for_judge)
            )
            corr = (
                {"correctness": None, "reason": "跳过裁判"}
                if skip_judge
                else helper.judge_correctness(sample, answer)
            )
            refusal = looks_like_refusal(answer)
            refusal_ok: float | None
            if sample.answerable:
                refusal_ok = None
            else:
                refusal_ok = 1.0 if refusal else 0.0
            tool_ok: float | None
            if sample.should_retrieve:
                tool_ok = 1.0 if tool_called else 0.0
            else:
                tool_ok = 1.0 if not tool_called else 0.0

            qa_row = {
                "id": sample.id,
                "run": run_index,
                "latency_ms": qa_latency,
                "faithfulness": faith.get("faithfulness"),
                "correctness": corr.get("correctness"),
                "refusal_ok": refusal_ok,
                "tool_ok": tool_ok,
            }
            qa_rows.append(qa_row)
            report["qa"] = {
                "mode": args.qa_mode,
                "answer": answer,
                "tool_called": tool_called,
                "rag": rag_meta,
                "latency_ms": round(qa_latency, 1),
                "faithfulness": faith,
                "correctness": corr,
                "refusal": refusal,
                "refusal_ok": refusal_ok,
                "tool_ok": tool_ok,
            }

        sample_reports.append(report)

    retrieval_summary = {
        channel: aggregate_channel(rows, ks) for channel, rows in channel_rows.items() if rows
    }
    qa_summary: dict[str, Any] = {}
    if qa_rows:
        faiths = [r["faithfulness"] for r in qa_rows if r["faithfulness"] is not None]
        corrs = [r["correctness"] for r in qa_rows if r["correctness"] is not None]
        refusals = [r["refusal_ok"] for r in qa_rows if r["refusal_ok"] is not None]
        tools = [r["tool_ok"] for r in qa_rows if r["tool_ok"] is not None]
        qa_summary = {
            "n": len(qa_rows),
            "faithfulness": round(mean(faiths), 4) if faiths else None,
            "correctness": round(mean(corrs), 4) if corrs else None,
            "refusal_accuracy": round(mean(refusals), 4) if refusals else None,
            "tool_coverage": round(mean(tools), 4) if tools else None,
            "latency_ms_p50": round(percentile([r["latency_ms"] for r in qa_rows], 50), 1),
            "latency_ms_p95": round(percentile([r["latency_ms"] for r in qa_rows], 95), 1),
        }

    result = {
        "config": {
            "gold": str(Path(args.gold)),
            "stage": args.stage,
            "channels": channels,
            "recall_k": recall_k,
            "top_k": top_k,
            "ks": ks,
            "runs": runs,
            "types": type_counts,
            "qa_mode": args.qa_mode,
            "skip_judge": args.skip_judge,
            "sample_count": len(samples),
        },
        "retrieval": retrieval_summary,
        "qa": qa_summary,
        "samples": sample_reports,
    }

    kpi_k = args.top_k or config.rag_top_k
    if kpi_k not in ks:
        kpi_k = ks[0]
    kpis = build_kpi_lift(retrieval_summary, k=kpi_k, baseline="vector")
    result["kpis"] = kpis

    stability_channel = next(
        (
            name
            for name in ("hybrid_rerank", "hybrid", "vector")
            if channel_rows.get(name)
        ),
        "",
    )
    stability = stability_summary(channel_rows.get(stability_channel) or [], kpi_k)
    stability["channel"] = stability_channel
    result["stability"] = stability
    answer_stats = answer_stability(qa_rows) if qa_rows else {}
    if answer_stats:
        result["answer_stability"] = answer_stats

    embed_tokens = int(getattr(vector_embedding_service, "total_tokens", 0) or 0)
    chat_prompt = helper.prompt_tokens if helper else 0
    chat_completion = helper.completion_tokens if helper else 0
    observations = len(samples) * runs
    total_tokens = embed_tokens + chat_prompt + chat_completion
    cost_note = "embedding 覆盖本轮全部检索通道。"
    if args.qa_mode == "agent":
        cost_note += " chat 只含裁判，不含 Agent 规划与工具调用。"
    elif do_qa:
        cost_note += " chat 为生成与裁判合计。"
    cost = {
        "embedding_tokens": embed_tokens,
        "chat_prompt_tokens": chat_prompt,
        "chat_completion_tokens": chat_completion,
        "total_tokens": total_tokens,
        "observations": observations,
        "avg_tokens_per_run": round(total_tokens / observations, 1) if observations else 0,
        "note": cost_note,
    }
    result["cost"] = cost

    bad_path = (
        Path(args.bad_cases)
        if args.bad_cases
        else _ROOT / "eval" / "reports" / "bad_cases.jsonl"
    )
    bad_count = write_bad_cases(bad_path, sample_reports, kpi_k)
    result["bad_cases"] = {"path": str(bad_path), "count": bad_count}

    print_retrieval_table(retrieval_summary, ks)
    print_kpi_lift(kpis)
    print_stability(stability)
    if answer_stats.get("runs", 0) > 1:
        print(
            f"问答稳定性={_fmt_rate(answer_stats['pass_rate'] or 0)}  "
            f"会飘={answer_stats['unstable']}"
        )
    if do_qa:
        print_qa_table(qa_summary)
    print_cost(cost)
    if bad_count:
        print(f"错题 {bad_count} 条已追加到 {bad_path}")

    output = Path(args.output) if args.output else _ROOT / "eval" / "reports" / "rag_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线评测 RAG 检索质量与问答质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "金标最少给 question + gold_contains 或 gold_files。\n"
            "Milvus 需已导入知识库，.env 中需配置 DASHSCOPE_API_KEY。"
        ),
    )
    parser.add_argument(
        "--gold",
        default=str(_ROOT / "eval" / "gold" / "sample.jsonl"),
        help="JSONL 金标路径",
    )
    parser.add_argument(
        "--output",
        default="",
        help="报告输出路径，默认 eval/reports/rag_eval.json",
    )
    parser.add_argument(
        "--stage",
        choices=["retrieve", "rerank", "qa", "all"],
        default="all",
        help="retrieve=只评召回; rerank/all=含重排; qa=只评问答",
    )
    parser.add_argument(
        "--qa-mode",
        choices=["grounded", "agent", "off"],
        default="grounded",
        help="grounded=检索上下文+LLM; agent=走完整 Agent; off=不评问答",
    )
    parser.add_argument("--recall-k", type=int, default=0, help="召回数量，默认配置 rag_recall_k")
    parser.add_argument("--top-k", type=int, default=0, help="生成用上下文条数，默认 rag_top_k")
    parser.add_argument("--ks", default="3,10", help="检索指标 K 列表，逗号分隔")
    parser.add_argument("--limit", type=int, default=None, help="只评前 N 条")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每题重复次数。稳定性看 --runs 5",
    )
    parser.add_argument(
        "--bad-cases",
        default="",
        help="错题追加路径，默认 eval/reports/bad_cases.jsonl",
    )
    parser.add_argument(
        "--no-ablation", action="store_true", help="只评 hybrid_rerank，不做向量/混合对比"
    )
    parser.add_argument("--skip-judge", action="store_true", help="生成答案但不做 LLM 裁判")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stage in {"retrieve", "rerank"} and args.qa_mode != "off":
        args.qa_mode = "off"
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
