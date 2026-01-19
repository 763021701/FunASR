#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 AutoModel 批量评估热词性能

支持的模型：
  1. seaco_paraformer - 深层融合热词（支持 seaco_weight 和 nfilter 调优）
     iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
  
  2. contextual_paraformer - 上下文热词
     iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-pytorch
  
  3. fun-asr-nano - LLM prompt 热词
     FunAudioLLM/Fun-ASR-Nano-2512

运行示例：
  # Seaco-Paraformer（可调节 seaco_weight）
  python eval_hotwords_automodel.py \\
    --manifest test.jsonl \\
    --hotwords hotwords.txt \\
    --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \\
    --vad_model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \\
    --seaco_weight 0.5 \\
    --output_json seaco_eval.json
  
  # Fun-ASR-Nano（自动启用 trust_remote_code）
  python eval_hotwords_automodel.py \\
    --manifest test.jsonl \\
    --hotwords hotwords.txt \\
    --model FunAudioLLM/Fun-ASR-Nano-2512 \\
    --output_json nano_eval.json
  
  # Contextual-Paraformer
  python eval_hotwords_automodel.py \\
    --manifest test.jsonl \\
    --hotwords hotwords.txt \\
    --model iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-pytorch \\
    --vad_model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \\
    --output_json contextual_eval.json

热词文件格式：每行一个热词（不需要权重）
  阿里巴巴
  夸父
  通义

评估指标：
  hotword_recall    = 正确识别的热词数 / 实际出现的热词总数
  hotword_precision = 正确识别的热词数 / 系统识别的总热词数
  hotword_f1        = 2PR/(P+R)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# 抑制 transformers 的警告信息（主要用于 Fun-ASR-Nano）
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 并行警告


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


def _count_hotwords(
    text: str, hotwords: Iterable[str], *, ignore_case: bool, normalize_space: bool
) -> Dict[str, int]:
    norm_text = _normalize_text(text, ignore_case=ignore_case, normalize_space=normalize_space)
    out: Dict[str, int] = {}
    for hw in hotwords:
        norm_hw = _normalize_text(hw, ignore_case=ignore_case, normalize_space=normalize_space)
        out[hw] = _count_non_overlapping(norm_text, norm_hw)
    return out


def _metric_from_counts(correct: int, actual: int, predicted: int) -> Dict[str, float]:
    recall = (correct / actual) if actual > 0 else 0.0
    precision = (correct / predicted) if predicted > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "correct": float(correct),
        "actual": float(actual),
        "predicted": float(predicted),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


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
        ref_c = _count_hotwords(
            ref, hotwords, ignore_case=ignore_case, normalize_space=normalize_space
        )
        hyp_c = _count_hotwords(
            hyp, hotwords, ignore_case=ignore_case, normalize_space=normalize_space
        )
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


def _run_condition(
    name: str,
    model,
    items: List[dict],
    hotwords: List[str],
    weights: Dict[str, int],
    *,
    hotword_str: Optional[str],
    ignore_case: bool,
    normalize_space: bool,
    batch_size: int,
    model_type: str = "seaco",  # "seaco", "contextual", "fun_asr_nano"
) -> Dict[str, object]:
    """
    使用 AutoModel 推理一批 wav，计算热词指标
    """
    pairs: List[Tuple[str, str, str]] = []
    details: List[dict] = []

    # 准备推理参数
    kwargs = {"batch_size": batch_size}
    if hotword_str is not None:
        # 根据模型类型传递不同格式的热词参数
        if model_type == "fun_asr_nano":
            # fun-asr-nano 需要列表格式
            kwargs["hotwords"] = hotwords
        else:
            # seaco/contextual 需要字符串格式
            kwargs["hotword"] = hotword_str
        
        # 添加 seaco 相关参数（仅对 seaco_paraformer）
        if model_type == "seaco" and hasattr(model.model, "_seaco_decode_with_ASF"):
            kwargs["seaco_weight"] = model.kwargs.get("seaco_weight", 1.0)
            kwargs["nfilter"] = model.kwargs.get("nfilter", 50)

    print(f"\n[{name}] 开始推理，共 {len(items)} 条...")
    
    for idx, it in enumerate(items, 1):
        wav = it.get("wav") or it.get("audio") or it.get("wav_path")
        ref = it.get("text") or it.get("ref") or it.get("transcript")
        if not wav or ref is None:
            raise ValueError("manifest 每行需要至少包含 wav/audio + text/ref 字段")

        # 推理
        try:
            result = model.generate(input=wav, **kwargs)
            hyp = result[0].get("text", "")
        except Exception as e:
            print(f"  [{idx}/{len(items)}] 推理失败: {wav}, 错误: {e}")
            hyp = ""

        pairs.append((wav, str(ref), hyp))
        details.append({"wav": wav, "ref": ref, "hyp": hyp})
        
        if idx % 10 == 0 or idx == len(items):
            print(f"  进度: {idx}/{len(items)}")

    metrics = _accumulate_counts(
        pairs, hotwords, ignore_case=ignore_case, normalize_space=normalize_space
    )
    return {"name": name, "metrics": metrics, "details": details}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="评估 contextual_paraformer / seaco_paraformer 热词效果"
    )
    ap.add_argument("--manifest", required=True, help="JSONL: 每行 {'wav':..., 'text':...}")
    ap.add_argument("--hotwords", required=True, help="热词文件：每行 '热词 权重' 或 '热词'")
    ap.add_argument(
        "--model",
        required=True,
        help="模型名称或路径，支持: seaco_paraformer, contextual_paraformer, Fun-ASR-Nano-2512",
    )
    ap.add_argument("--model_revision", default="v2.0.4", help="模型版本（Fun-ASR-Nano 会自动忽略此参数）")
    ap.add_argument(
        "--punc_model",
        default="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        help="标点模型（留空则不加标点）",
    )
    ap.add_argument("--punc_model_revision", default="v2.0.4", help="标点模型版本")
    ap.add_argument(
        "--vad_model",
        default="",
        help="VAD 模型（可选，用于长音频切分），如 iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    )
    ap.add_argument("--vad_model_revision", default="v2.0.4", help="VAD 模型版本")
    ap.add_argument("--device", default="cuda", help="推理设备：cuda 或 cpu")
    ap.add_argument("--ngpu", type=int, default=1, help="GPU 数量（0 表示 CPU）")
    ap.add_argument("--ncpu", type=int, default=16, help="CPU 线程数")
    ap.add_argument("--batch_size", type=int, default=1, help="批大小")
    ap.add_argument(
        "--seaco_weight",
        type=float,
        default=1.0,
        help="[仅 seaco_paraformer] 热词融合权重（默认 1.0，建议 0.3-0.8，越小越保守）",
    )
    ap.add_argument(
        "--nfilter",
        type=int,
        default=50,
        help="[仅 seaco_paraformer] ASF 过滤器数量（默认 50，减小会更谨慎地选择热词）",
    )
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="[仅 Fun-ASR-Nano] 信任远程代码（Fun-ASR-Nano 必需）",
    )
    ap.add_argument("--ignore_case", action="store_true", help="匹配热词时忽略大小写")
    ap.add_argument(
        "--normalize_space",
        action="store_true",
        default=True,
        help="把连续空白归一成单空格（默认开）",
    )
    ap.add_argument("--output_json", default="automodel_hotword_eval.json", help="输出明细 json")
    args = ap.parse_args()

    items = _read_jsonl(args.manifest)
    hotwords, weights = _parse_hotwords_file(args.hotwords)
    if not hotwords:
        raise SystemExit("hotwords 文件为空")

    print(f"加载模型: {args.model}")
    if args.punc_model:
        print(f"标点模型: {args.punc_model}")
    if args.vad_model:
        print(f"VAD 模型: {args.vad_model}")
    
    try:
        from funasr import AutoModel
    except ImportError as e:
        raise SystemExit(
            "请先安装 funasr: pip install -U funasr\n"
            "国内用户: pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"
        ) from e

    # 检测是否是 Fun-ASR-Nano 模型
    is_fun_asr_nano = "Fun-ASR-Nano" in args.model or "FunAudioLLM" in args.model
    
    # 构建 AutoModel 参数
    model_kwargs = {
        "model": args.model,
        "device": args.device,
        "ngpu": args.ngpu,
        "ncpu": args.ncpu,
        "disable_pbar": True,
        "disable_log": True,  # 禁用组件注册表的打印
    }
    
    # Fun-ASR-Nano 不使用 model_revision 参数
    if not is_fun_asr_nano:
        model_kwargs["model_revision"] = args.model_revision
    
    # 添加 punc_model
    if args.punc_model:
        model_kwargs["punc_model"] = args.punc_model
        model_kwargs["punc_model_revision"] = args.punc_model_revision
    
    # 添加 vad_model（可选，用于长音频）
    if args.vad_model:
        model_kwargs["vad_model"] = args.vad_model
        model_kwargs["vad_model_revision"] = args.vad_model_revision
    
    # 添加 seaco 相关参数（供后续使用）
    model_kwargs["seaco_weight"] = args.seaco_weight
    model_kwargs["nfilter"] = args.nfilter
    
    # Fun-ASR-Nano 的特殊处理
    if is_fun_asr_nano:
        print("检测到 Fun-ASR-Nano 模型")
        import logging
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
        model_kwargs["language"] = "zh"
    elif args.trust_remote_code:
        model_kwargs["trust_remote_code"] = args.trust_remote_code

    model = AutoModel(**model_kwargs)
    print("模型加载完成！")
    
    # 检测模型类型
    model_class_name = model.model.__class__.__name__
    if "FunASRNano" in model_class_name:
        model_type = "fun_asr_nano"
        print(f"检测到模型类型: Fun-ASR-Nano (使用 LLM prompt 热词)")
    elif "SeacoParaformer" in model_class_name:
        model_type = "seaco"
        print(f"检测到模型类型: Seaco-Paraformer (深层融合热词)")
        print(f"  seaco_weight: {args.seaco_weight}, nfilter: {args.nfilter}")
    elif "ContextualParaformer" in model_class_name:
        model_type = "contextual"
        print(f"检测到模型类型: Contextual-Paraformer (上下文热词)")
    else:
        model_type = "unknown"
        print(f"警告: 未知模型类型 {model_class_name}，将使用默认热词传递方式")

    # 准备热词字符串（所有热词用空格连接）
    # 注意：seaco/contextual 不支持权重，只需要热词本身
    hotword_str = " ".join(hotwords)
    print(f"\n热词列表（共 {len(hotwords)} 个）: {hotword_str}")

    # 1. baseline: 不使用热词
    baseline = _run_condition(
        "baseline_no_hotword",
        model,
        items,
        hotwords,
        weights,
        hotword_str=None,
        ignore_case=args.ignore_case,
        normalize_space=args.normalize_space,
        batch_size=args.batch_size,
        model_type=model_type,
    )

    # 2. with_hotword: 使用热词
    with_hotword = _run_condition(
        "with_hotword",
        model,
        items,
        hotwords,
        weights,
        hotword_str=hotword_str,
        ignore_case=args.ignore_case,
        normalize_space=args.normalize_space,
        batch_size=args.batch_size,
        model_type=model_type,
    )

    # 输出结果
    merged = {
        "model": args.model,
        "model_type": model_type,
        "hotwords": hotwords,
        "baseline_no_hotword": baseline,
        "with_hotword": with_hotword,
    }
    if model_type == "seaco":
        merged["seaco_weight"] = args.seaco_weight
        merged["nfilter"] = args.nfilter
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    b = baseline["metrics"]["summary"]
    w = with_hotword["metrics"]["summary"]

    print("\n" + "=" * 60)
    print("=== baseline (no hotword) ===")
    print(
        f"correct={int(b['correct'])} actual={int(b['actual'])} predicted={int(b['predicted'])} "
        f"R={b['recall']:.4f} P={b['precision']:.4f} F1={b['f1']:.4f}"
    )
    print("\n=== with_hotword ===")
    print(
        f"correct={int(w['correct'])} actual={int(w['actual'])} predicted={int(w['predicted'])} "
        f"R={w['recall']:.4f} P={w['precision']:.4f} F1={w['f1']:.4f}"
    )
    print("=" * 60)
    print(f"\n详细结果已保存到: {args.output_json}")
    print(
        f"热词召回率提升: {(w['recall'] - b['recall']):.4f} "
        f"({b['recall']:.4f} -> {w['recall']:.4f})"
    )
    print(
        f"热词F1提升: {(w['f1'] - b['f1']):.4f} " f"({b['f1']:.4f} -> {w['f1']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
