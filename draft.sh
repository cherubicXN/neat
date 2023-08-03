for ckpt in {0..2000..100}; do
    result_str=../exps/dtu-ins-v4/24/2023_01_13_17_51_02/wireframes/2000-d=1-s=0.05-v5-group-v2-ckpt-$ckpt.npz
    if [ -f $result_str ]; then
        echo $result_str exists
        continue
    fi
    python postprocess/grouping-v2.py   --conf confs/neat-simple/dtu-ins-v4.conf    --checkpoint $ckpt --scan_id 24   --timestamp 2023_01_13_17_51_02 --data ../exps/dtu-ins-v4/24/2023_01_13_17_51_02/wireframes/2000-d=1-s=0.05-v5.npz --comment "ckpt-$ckpt"
    python evaluation/eval-lsr-dtu.py  --data  $result_str  --scan 24 --cam ../data/DTU/scan24/cameras.npz
done