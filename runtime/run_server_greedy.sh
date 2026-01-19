
download_model_dir="/root/autodl-tmp/cache/onnx"
model_dir="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
vad_dir="iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
punc_dir="iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx"
itn_dir="thuduj12/fst_itn_zh"
lm_dir=""
port=10095
certfile=0
keyfile=0
hotword=""
# set decoder_thread_num
decoder_thread_num=$(cat /proc/cpuinfo | grep "processor"|wc -l) || { echo "Get cpuinfo failed. Set decoder_thread_num = 32"; decoder_thread_num=32; }
multiple_io=16
io_thread_num=$(( (decoder_thread_num + multiple_io - 1) / multiple_io ))
model_thread_num=1
cmd_path=/root/autodl-tmp/project/FunASR/runtime/websocket/build/bin
cmd=funasr-wss-server

. ./tools/utils/parse_options.sh || exit 1;

if [ -z "$certfile" ] || [ "$certfile" = "0" ]; then
  certfile=""
  keyfile=""
fi

cd $cmd_path
$cmd_path/${cmd}  \
  --download-model-dir "${download_model_dir}" \
  --model-dir "${model_dir}" \
  --vad-dir "${vad_dir}" \
  --punc-dir "${punc_dir}" \
  --itn-dir "${itn_dir}" \
  --lm-dir "${lm_dir}" \
  --decoder-thread-num ${decoder_thread_num} \
  --model-thread-num ${model_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
  --certfile  "${certfile}" \
  --keyfile "${keyfile}" \
  --hotword "${hotword}" \
  --gpu \
  --bladedisc false

