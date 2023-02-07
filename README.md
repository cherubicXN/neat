## Log of Experiments

|date   | scene | timestamp | checkpoint | inference code lines | train log|
|---|---|---|---|---|---|
|2022-10-10  | DTU-24 | 2022_10_10_15_35_26 | [2000.pth](exps/dtu-offset_24/2022_10_10_15_35_26/checkpoints/ModelParameters/2000.pth) | ``python evaluation/wireframe.py --conf confs/dtu-offset.conf --timestamp 2022_10_10_15_35_26  --scan_id 24`` | [train.log](exps/dtu-offset_24/2022_10_10_15_35_26/train.log)


## Results

### DTU 24 

```bash
# Computing the wireframes
python evaluation/wireframe.py --conf confs/dtu-hat.conf  --timestamp final --scan_id 24 --sdf-threshold 5e-3

# Display the results
python evaluation/show.py --data ../evals/dtuhat_24/wireframes/latest-5e-03.npz \
 --save --load-views ../evals/dtuhat_24/wireframes/latest-5e-03_record
```
#### a): NEAT Fields Learning
<p float="left">
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0001.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0002.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0003.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0004.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0005.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/latest-5e-03_record/image_0006.png" width="16%" />
</p>

#### b): NEAT Fields Learning (updated)
<p float="left">
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0001.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0002.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0003.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0004.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0005.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-ref_record/image_0006.png" width="16%" />
</p>

#### c): NEAT Fields Learning (updated2)
<p float="left">
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0001.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0002.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0003.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0004.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0005.png" width="16%" />
  <img src="evals/dtu-offset_24/wireframes/latest-1e-02_record/image_0006.png" width="16%" />
</p>

#### d): Classical Pipeline
<p float="left">
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0001.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0002.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0003.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0004.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0005.png" width="16%" />
  <img src="evals/dtuhat_24/wireframes/matching_record/image_0006.png" width="16%" />
</p>

### DTU 105

```bash
# Computing the wireframes
python evaluation/wireframe.py --conf confs/dtu-hat.conf \
--scan_id 105 --timestamp 2022_10_05_15_57_45 \
--sdf-threshold 0.01 --checkpoint 1000 --chunksize 2048 

# Display the results
python evaluation/show.py --data  ../evals/dtuhat_105/wireframes/10000-1e-02.npz  \
 --save  --load-views ../evals/dtuhat_105/wireframes/1000-1e-02_record
```
#### a): NAT Fields Learning
<p float="left">
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0001.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0002.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0003.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0004.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0005.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/1000-1e-02_record/image_0006.png" width="16%" />
</p>

#### b): Classical Pipeline
<p float="left">
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0001.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0002.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0003.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0004.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0005.png" width="16%" />
  <img src="evals/dtuhat_105/wireframes/matching_record/image_0006.png" width="16%" />
</p>


### LEGO

```bash
# Computing the wireframes
 python evaluation/wireframe.py --conf confs/blender-hat.conf --timestamp 2022_10_06_11_05_25 --sdf-threshold 0.01 --checkpoint 100 --chunksize 2048
# Display the results
```


