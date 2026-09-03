#!/bin/bash
cd /home/kana5123/ETRI/guardrail_eval
PY=/home/kana5123/ETRI/.venv/bin/python
# 128코어를 5개 작업이 나눠 쓰도록 BLAS 스레드 제한 (안 하면 load 263 으로 서로를 느리게 함)
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
$PY -u src/failure_structure/probes.py --which main   > logs/fs_probes.log   2>&1 &
$PY -u src/failure_structure/geometry.py              > logs/fs_geom.log     2>&1 &
$PY -u src/failure_structure/controls.py              > logs/fs_controls.log 2>&1 &
$PY -u src/failure_structure/structure.py             > logs/fs_struct.log   2>&1 &
$PY -u src/failure_structure/crossdataset.py          > logs/fs_cross.log    2>&1 &
wait
echo FS_ALL_DONE
