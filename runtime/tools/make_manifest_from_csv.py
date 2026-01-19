#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 “wav目录 + metadata_filter_fix.csv” 转成 eval_hotwords_wfst.py 需要的 manifest.jsonl。

输入 CSV（你描述的格式）：
  列：ID, Filename, Original Text, duration
    - Filename: 对应 wav 文件名（例如 0001.wav 或者不含扩展名也可）
    - Original Text: 参考文本

输出 JSONL：
  每行一个样本，例如：
    {"id":"0001","wav":"/abs/path/0001.wav","text":"...","duration":1.23}

用法：
  python3 runtime/tools/make_manifest_from_csv.py \
    --wav_dir /path/to/wavs \
    --csv metadata_filter_fix.csv \
    --out manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Optional


def _pick(row: dict, *candidates: str) -> Optional[str]:
    for k in candidates:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v != "":
                return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True, help="包含 wav 文件的目录")
    ap.add_argument("--csv", required=True, help="metadata_filter_fix.csv 路径")
    ap.add_argument("--out", required=True, help="输出 manifest.jsonl 路径")
    ap.add_argument("--require_exists", action="store_true", help="若 wav 文件不存在则报错（默认跳过）")
    args = ap.parse_args()

    wav_dir = os.path.abspath(args.wav_dir)
    if not os.path.isdir(wav_dir):
        raise SystemExit(f"--wav_dir not a directory: {wav_dir}")
    csv_path = os.path.abspath(args.csv)
    out_path = os.path.abspath(args.out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    kept = 0
    skipped_missing = 0

    # utf-8-sig: 兼容带 BOM 的 CSV
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f, open(out_path, "w", encoding="utf-8") as fo:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")

        for row in reader:
            total += 1
            wav_name = _pick(row, "Filename", "filename", "wav", "wav_name")
            text = _pick(row, "Original Text", "OriginalText", "original_text", "text", "transcript")
            sid = _pick(row, "ID", "id", "utt_id", "uttid")
            dur = _pick(row, "duration", "Duration", "dur")

            if not wav_name:
                continue
            if not text:
                text = ""

            # 允许 Filename 不带扩展名
            if not wav_name.lower().endswith(".wav"):
                wav_name = wav_name + ".wav"
            wav_path = os.path.join(wav_dir, wav_name)

            if not os.path.exists(wav_path):
                if args.require_exists:
                    raise SystemExit(f"Missing wav: {wav_path} (row={total})")
                skipped_missing += 1
                continue

            item = {"wav": wav_path, "text": text}
            if sid is not None:
                item["id"] = sid
            if dur is not None:
                try:
                    item["duration"] = float(dur)
                except ValueError:
                    pass

            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote: {out_path}")
    print(f"rows: {total}, kept: {kept}, skipped_missing_wav: {skipped_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
