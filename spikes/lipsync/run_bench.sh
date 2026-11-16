#!/bin/bash
# run_bench.sh <batch> <outname> : MuseTalk bench on Arc card0 with co-resident service probes
cd ~/skcall-lipsync-bench/MuseTalk && . ../venv/bin/activate
bash ../probe_services.sh 900 10 during > ../probe_during_$2.csv 2>&1 &
PROBE=$!
export ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
/usr/bin/time -v timeout 600 python bench_musetalk.py --portrait ../portrait.png --audio ../clip.wav --device xpu --batch $1 --out ../$2.mp4 > ../bench_$2.log 2>&1
kill $PROBE; wait $PROBE 2>/dev/null
grep -E "RESULT|Error|error|Maximum resident" ../bench_$2.log | tail -8
cat ../probe_during_$2.csv
