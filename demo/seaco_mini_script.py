#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Optional


def _normalize_hotwords_arg(hotwords: Optional[str]) -> Optional[str]:
    """
    FunASR SeacoParaformer expects hotwords via cfg key `hotword`,
    which can be:
      - a local .txt path (each line a hotword)
      - a url
      - a plain string with hotwords separated by spaces
    """
    if hotwords is None:
        return None
    hotwords = hotwords.strip()
    if not hotwords:
        return None
    # If it exists locally, pass path directly (FunASR will parse txt).
    if os.path.exists(hotwords):
        return hotwords
    return hotwords


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Seaco-Paraformer ASR demo (FunASR AutoModel)."
    )
    parser.add_argument("--audio", required=True, help="Path to an audio file, e.g. a.wav")
    parser.add_argument(
        "--hotwords",
        default=None,
        help='Hotwords txt path or string, e.g. "热词1 热词2" or hotwords.txt',
    )
    parser.add_argument(
        "--model",
        default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        help="ModelScope model id for Seaco-Paraformer.",
    )
    parser.add_argument(
        "--model_revision",
        default="v2.0.4",
        help="Model revision on the hub (default: v2.0.4).",
    )

    # Optional: enable VAD + PUNC for a "full pipeline" (segment + sentence timestamps)
    parser.add_argument(
        "--vad_model",
        default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="Optional VAD model id. Set to empty string to disable.",
    )
    parser.add_argument(
        "--vad_model_revision",
        default="v2.0.4",
        help="VAD model revision (default: v2.0.4).",
    )
    parser.add_argument(
        "--punc_model",
        default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        help="Optional punctuation model id. Set to empty string to disable.",
    )
    parser.add_argument(
        "--punc_model_revision",
        default="v2.0.4",
        help="PUNC model revision (default: v2.0.4).",
    )

    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU threads")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"[ERROR] audio not found: {audio_path}", file=sys.stderr)
        return 2

    hotword = _normalize_hotwords_arg(args.hotwords)

    # Lazy import so the script can print argument errors quickly.
    from funasr import AutoModel

    vad_model = args.vad_model.strip() if isinstance(args.vad_model, str) else args.vad_model
    punc_model = args.punc_model.strip() if isinstance(args.punc_model, str) else args.punc_model
    vad_model = vad_model if vad_model else None
    punc_model = punc_model if punc_model else None

    model = AutoModel(
        model=args.model,
        model_revision=args.model_revision,
        vad_model=vad_model,
        vad_model_revision=args.vad_model_revision,
        punc_model=punc_model,
        punc_model_revision=args.punc_model_revision,
        device=args.device,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        disable_pbar=True,
        disable_log=True,
    )

    cfg = {
        # If VAD is enabled, this controls per-batch total audio duration (seconds).
        "batch_size_s": 300,
    }
    if hotword is not None:
        cfg["hotword"] = hotword
    # Sentence timestamps rely on punctuation post-processing in AutoModel.inference_with_vad().
    if vad_model is not None and punc_model is not None:
        cfg["sentence_timestamp"] = True

    results = model.generate(input=audio_path, **cfg)
    # AutoModel returns a list of results (one per input).
    out_obj = results[0] if isinstance(results, list) and results else {"text": ""}

    text = out_obj.get("text", "")
    print(text)

    json_str = json.dumps(out_obj, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(json_str)
            f.write("\n")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


