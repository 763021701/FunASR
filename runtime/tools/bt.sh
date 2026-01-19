# for weight in 0.3 0.5 0.7 1.0; do
#   python eval_hotwords_automodel.py \
#     --manifest manifest.jsonl \
#     --hotwords hotwords/hotwords_for_automodel.txt \
#     --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
#     --vad_model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
#     --seaco_weight $weight \
#     --output_json seaco_weight_${weight}.json
# done

# python eval_hotwords_automodel.py \
#   --manifest manifest.jsonl \
#   --hotwords hotwords/hotwords_for_automodel.txt \
#   --model iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
#   --vad_model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
#   --output_json contextual_eval.json

python eval_hotwords_automodel.py \
  --manifest manifest.jsonl \
  --hotwords hotwords/hotwords_for_automodel.txt \
  --model FunAudioLLM/Fun-ASR-Nano-2512 \
  --punc_model "" \
  --vad_model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --output_json nano_eval.json