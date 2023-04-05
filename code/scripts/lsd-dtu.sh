# scans=(65 16 17 18 19 20 21 22 23 24 37 40 55 60 105)
scans=(65)

for n in ${scans[@]}; do
  scan=scan$n
  echo processing $scan

  python -m sslib.predict-lsd --img ../data/DTU/$scan/image/*.png --save-txt ../data/DTU/$scan/lsd

done

# for loop to iterate over scans
# for scan in $scans; do
#     echo $scan
#     # create a directory for each scan
#     # mkdir -p $scan
#     # # copy the files to the directory
#     # cp -r $scan/* $scan/
#     # # go to the directory
#     # cd $scan
#     # # run LSD-SLAM
#     # ./lsd-slam $scan
#     # # go back to the parent directory
#     # cd ..
# done
