mkdir /app/log
GLOG_log_dir=/app/log GLOG_v=2 /home/wenet/runtime/libtorch/bin/grpc_server_main --chunk-size -1 --model_path asr/final.zip --unit-path asr/units.txt &
uvicorn main:app --host 0.0.0.0 --reload --port 4343
