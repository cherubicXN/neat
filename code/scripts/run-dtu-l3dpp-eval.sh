for scanid in 16 17 18 19 21 22 23 24 37 40 65 105
do
    echo $scanid
    sh scripts/eval-dtu-l3dpp-dft.sh $scanid
done