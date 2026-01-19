#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估 FunASR runtime(WFST) 热词效果（支持三路对比，热词由服务端控制）：

指标：
  hotword_recall    = 正确识别的热词数 / 实际出现的热词总数
  hotword_precision = 正确识别的热词数 / 系统识别的总热词数
  hotword_f1        = 2PR/(P+R)

这里“热词数”按“出现次数(可重复)”统计：
  correct = sum_over_utt,sum_over_hotword min(count(ref, hw), count(hyp, hw))

输入（用于“统计指标”，不一定用于“注入热词”）：
  - manifest: JSONL，每行至少包含 {"wav": "...", "text": "..."}  (wav 一句，对应参考文本)
  - hotwords: 热词文件，每行一个热词，可带权重：  阿里巴巴 20

运行方式（最简单：启动 3 个离线 server，对应 3 个端口）：
  1) greedy：          funasr-wss-server，--lm-dir 置空（走 GreedySearch）
  2) wfst_no_hotwords：funasr-wss-server，--lm-dir 指向 TLG.fst 目录，--hotword 置空
  3) wfst_with_hotwords：funasr-wss-server，--lm-dir 指向 TLG.fst 目录，--hotword 指向热词文件

  2) 运行本脚本，对同一份 manifest 分别请求三个 URI，输出三路对比指标。

注意：
  - 本脚本走 websocket 协议，因此“WFST 只在 runtime”并不妨碍评估。
  - 该评估只关心最终识别文本，不依赖时间戳/2pass。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln} of {path}: {e}") from e
    return items


def _parse_hotwords_file(path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    支持两种格式：
      - 热词 权重      e.g. 阿里巴巴 20
      - 纯热词         e.g. 阿里巴巴
    """
    hotwords: List[str] = []
    weights: Dict[str, int] = {}
    if not path:
        return hotwords, weights
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and re.fullmatch(r"\d+", parts[-1]):
                hw = " ".join(parts[:-1]).strip()
                w = int(parts[-1])
            else:
                hw = line
                w = 20
            if hw:
                hotwords.append(hw)
                weights[hw] = w
    # 去重，保持顺序
    seen = set()
    uniq = []
    for hw in hotwords:
        if hw not in seen:
            uniq.append(hw)
            seen.add(hw)
    return uniq, {hw: weights.get(hw, 20) for hw in uniq}


def _normalize_text(s: str, *, ignore_case: bool, normalize_space: bool) -> str:
    if s is None:
        return ""
    if normalize_space:
        s = re.sub(r"\s+", " ", s)
    if ignore_case:
        s = s.lower()
    return s


def _count_non_overlapping(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    cnt = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        cnt += 1
        start = idx + len(needle)
    return cnt


def _count_hotwords(text: str, hotwords: Iterable[str], *, ignore_case: bool, normalize_space: bool) -> Dict[str, int]:
    norm_text = _normalize_text(text, ignore_case=ignore_case, normalize_space=normalize_space)
    out: Dict[str, int] = {}
    for hw in hotwords:
        norm_hw = _normalize_text(hw, ignore_case=ignore_case, normalize_space=normalize_space)
        out[hw] = _count_non_overlapping(norm_text, norm_hw)
    return out


def _read_wav_pcm16(path: str) -> Tuple[bytes, int]:
    """
    读取 wav 并返回 PCM16 bytes + sample_rate。
    要求：
      - sample width = 2 (int16)
      - channels = 1（如果不是 1，这里直接报错，避免指标混乱）
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Unsupported wav sampwidth={sampwidth} (need 16-bit PCM): {path}")
        if channels != 1:
            raise ValueError(f"Unsupported wav channels={channels} (need mono): {path}")
        frames = wf.readframes(wf.getnframes())
        return frames, sr


@dataclass
class AsrResult:
    wav: str
    text: str
    raw_msg: dict


async def _asr_once(
    uri: str,
    wav_path: str,
    *,
    ssl_insecure: bool,
    hotwords_msg: str,
    mode: str = "offline",
    chunk_size: List[int] = None,
    chunk_interval: int = 10,
    use_itn: bool = True,
) -> AsrResult:
    try:
        import websockets  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'websockets'. Please install: pip install websockets"
        ) from e

    if chunk_size is None:
        chunk_size = [5, 10, 5]

    audio_bytes, sample_rate = _read_wav_pcm16(wav_path)
    wav_name = os.path.basename(wav_path)

    ssl_context = None
    if uri.startswith("wss://"):
        ssl_context = ssl.SSLContext()
        if ssl_insecure:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
    ) as ws:
        first = {
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_interval": chunk_interval,
            "audio_fs": sample_rate,
            "wav_name": wav_name,
            "wav_format": "pcm",
            "is_speaking": True,
            "hotwords": hotwords_msg,
            "itn": bool(use_itn),
        }
        await ws.send(json.dumps(first, ensure_ascii=False))

        # 离线服务端会累积所有 binary，再在 is_speaking=false 时统一解码；
        # 因此这里只要保证分块发送即可，不需要 sleep。
        stride = 32000  # bytes, ~1s@16k mono pcm16
        for beg in range(0, len(audio_bytes), stride):
            await ws.send(audio_bytes[beg : beg + stride])

        await ws.send(json.dumps({"is_speaking": False}))

        # 收到 mode=offline 的结果即结束
        while True:
            msg = await ws.recv()
            msgj = json.loads(msg)
            if msgj.get("mode") == "offline":
                return AsrResult(wav=wav_path, text=str(msgj.get("text", "")), raw_msg=msgj)


def _metric_from_counts(correct: int, actual: int, predicted: int) -> Dict[str, float]:
    recall = (correct / actual) if actual > 0 else 0.0
    precision = (correct / predicted) if predicted > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"correct": float(correct), "actual": float(actual), "predicted": float(predicted), "recall": recall, "precision": precision, "f1": f1}


def _accumulate_counts(
    pairs: List[Tuple[str, str, str]],
    hotwords: List[str],
    *,
    ignore_case: bool,
    normalize_space: bool,
) -> Dict[str, object]:
    """
    pairs: [(wav, ref_text, hyp_text), ...]
    """
    per_hw = {hw: {"correct": 0, "actual": 0, "predicted": 0} for hw in hotwords}
    total_correct = 0
    total_actual = 0
    total_pred = 0

    for _, ref, hyp in pairs:
        ref_c = _count_hotwords(ref, hotwords, ignore_case=ignore_case, normalize_space=normalize_space)
        hyp_c = _count_hotwords(hyp, hotwords, ignore_case=ignore_case, normalize_space=normalize_space)
        for hw in hotwords:
            a = int(ref_c.get(hw, 0))
            p = int(hyp_c.get(hw, 0))
            c = min(a, p)
            per_hw[hw]["actual"] += a
            per_hw[hw]["predicted"] += p
            per_hw[hw]["correct"] += c
            total_actual += a
            total_pred += p
            total_correct += c

    summary = _metric_from_counts(total_correct, total_actual, total_pred)
    per_hw_metrics = {
        hw: _metric_from_counts(v["correct"], v["actual"], v["predicted"])
        for hw, v in per_hw.items()
    }
    return {"summary": summary, "per_hotword": per_hw_metrics}


async def _run_condition(
    name: str,
    uri: str,
    items: List[dict],
    hotwords: List[str],
    weights: Dict[str, int],
    *,
    ssl_insecure: bool,
    ignore_case: bool,
    normalize_space: bool,
) -> Dict[str, object]:
    # 这里不向服务端“发送热词”，热词是否启用由服务端启动参数 --hotword 控制；
    # hotwords 文件仅用于本地统计命中次数。
    hotwords_msg = ""

    pairs: List[Tuple[str, str, str]] = []
    details: List[dict] = []
    for it in items:
        wav = it.get("wav") or it.get("audio") or it.get("wav_path")
        ref = it.get("text") or it.get("ref") or it.get("transcript")
        if not wav or ref is None:
            raise ValueError("manifest 每行需要至少包含 wav/audio + text/ref 字段")
        res = await _asr_once(uri, wav, ssl_insecure=ssl_insecure, hotwords_msg=hotwords_msg)
        hyp = res.text
        pairs.append((wav, str(ref), hyp))
        details.append({"wav": wav, "ref": ref, "hyp": hyp, "raw": res.raw_msg})

    metrics = _accumulate_counts(pairs, hotwords, ignore_case=ignore_case, normalize_space=normalize_space)
    return {"name": name, "uri": uri, "metrics": metrics, "details": details}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL: 每行 {'wav':..., 'text':...}")
    ap.add_argument("--hotwords", required=True, help="热词文件：每行 '热词 权重' 或 '热词'")
    ap.add_argument("--greedy_uri", required=True, help="greedy 服务端，如 ws://127.0.0.1:10095（--lm-dir 置空）")
    ap.add_argument("--wfst_no_hotwords_uri", required=True, help="wfst(无热词) 服务端，如 ws://127.0.0.1:10096（--lm-dir 有，--hotword 空）")
    ap.add_argument("--wfst_with_hotwords_uri", required=True, help="wfst(有热词) 服务端，如 ws://127.0.0.1:10097（--lm-dir 有，--hotword 指向热词文件）")
    ap.add_argument("--ssl_insecure", action="store_true", help="wss 自签名证书：跳过校验")
    ap.add_argument("--ignore_case", action="store_true", help="匹配热词时忽略大小写（英文建议开）")
    ap.add_argument("--normalize_space", action="store_true", default=True, help="把连续空白归一成单空格（默认开）")
    ap.add_argument("--output_json", default="hotword_eval.json", help="输出明细 json")
    args = ap.parse_args()

    items = _read_jsonl(args.manifest)
    hotwords, weights = _parse_hotwords_file(args.hotwords)
    if not hotwords:
        raise SystemExit("hotwords 文件为空")

    # 逐条跑，避免并发把服务器打爆；需要加速可自行改成 gather + semaphore。
    greedy = asyncio.get_event_loop().run_until_complete(
        _run_condition(
            "greedy",
            args.greedy_uri,
            items,
            hotwords,
            weights,
            ssl_insecure=args.ssl_insecure,
            ignore_case=bool(args.ignore_case),
            normalize_space=bool(args.normalize_space),
        )
    )

    wfst_no_hotwords = asyncio.get_event_loop().run_until_complete(
        _run_condition(
            "wfst_no_hotwords",
            args.wfst_no_hotwords_uri,
            items,
            hotwords,
            weights,
            ssl_insecure=args.ssl_insecure,
            ignore_case=bool(args.ignore_case),
            normalize_space=bool(args.normalize_space),
        )
    )

    wfst_with_hotwords = asyncio.get_event_loop().run_until_complete(
        _run_condition(
            "wfst_with_hotwords",
            args.wfst_with_hotwords_uri,
            items,
            hotwords,
            weights,
            ssl_insecure=args.ssl_insecure,
            ignore_case=bool(args.ignore_case),
            normalize_space=bool(args.normalize_space),
        )
    )

    merged = {
        "hotwords": hotwords,
        "greedy": greedy,
        "wfst_no_hotwords": wfst_no_hotwords,
        "wfst_with_hotwords": wfst_with_hotwords,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    g = greedy["metrics"]["summary"]
    w0 = wfst_no_hotwords["metrics"]["summary"]
    w1 = wfst_with_hotwords["metrics"]["summary"]
    print("=== greedy (no LM, GreedySearch) ===")
    print(
        f"correct={int(g['correct'])} actual={int(g['actual'])} predicted={int(g['predicted'])} "
        f"R={g['recall']:.4f} P={g['precision']:.4f} F1={g['f1']:.4f}"
    )
    print("=== wfst_no_hotwords (LM enabled, no hotwords sent) ===")
    print(
        f"correct={int(w0['correct'])} actual={int(w0['actual'])} predicted={int(w0['predicted'])} "
        f"R={w0['recall']:.4f} P={w0['precision']:.4f} F1={w0['f1']:.4f}"
    )
    print("=== wfst_with_hotwords (LM enabled, hotwords sent) ===")
    print(
        f"correct={int(w1['correct'])} actual={int(w1['actual'])} predicted={int(w1['predicted'])} "
        f"R={w1['recall']:.4f} P={w1['precision']:.4f} F1={w1['f1']:.4f}"
    )
    print(f"(details saved to {args.output_json})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

