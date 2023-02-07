## NEAT: Neural Attraction Fields for 3D Wireframe Reconstruction

### Authors: Nan Xue, Bin Tan

## Changelogs
- v0.0 ([4cef0e6d9545bdca00b22b47892aa952ec1b23b3](https://github.com/cherubicXN/neat/tree/4cef0e6d9545bdca00b22b47892aa952ec1b23b3))
  - Initially build the wireframe reconstruction pipeline 
    - It is amazing because we can learn the global junctions (and latents) via backpropogation
    - Huganrian Matching is extensively used to make connections between line segments and global junctions (latents)
    - Works well on DTU dataset with great generalization ability across scenes
    - dbscan is used for online clustering
  

- Plan
  - [ ] Checking if the online clustering is necessary
  - [ ] Using junctions for SDF and NEAT field learning
  - [ ] 
