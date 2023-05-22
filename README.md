# GH-NeRF
---

“GH-NeRF”, is a NeRF-like model which uses 3D multivariate Gaussian random variables along with Spherical Harmonics to accelerate the training process by 25% and also, tackle the problem of excessively blurring or alias of the original NeRF implementation, lowering the error rate by 6.4% relative to NeRF on LLFF dataset.

TL;DR: Mip-NeRF + NeRF-SH (promoted by Spherical Harmonics) implementation in Pytorch
### Architecture

![](/static/layout.png)


### Installation/Train:

Preliminaries:

```sh
cd GH-NeRF
pip install -r requirements.txt
```

To download dataset:

- `bash scripts/download_llff.sh` to download LLFF

To train the model:

```sh
python run_nerf.py --config configs/trex.txt
```

Project Layout:

```
├─configs
├─data
│  ├─nerf_llff_data
│     ├─fern
│     │  ├─images
│     │  ├─images_4
│     │  ├─images_8
│     │  ├─mpis4
│     │  └─sparse
│     │      └─0
│     └─trex
│         ├─images
│         ├─images_4
│         ├─images_8
│         ├─outputs
│         └─sparse
│             └─0
├─logs
│  ├─fern_test
│  │  └─train
│  └─trex_test
│      ├─testset_200000
|      ├─...
│      └─train
├─scripts
├─static
```

### Result

trex: 
<div>
  <img src="/static/trex.png" width="49%"> </img> 
  <img src="/static/dep.png" width="49%"> </img> 
</div>


<div>
   <img src="https://github.com/Bariona/GH-NeRF/blob/main/static/trex_test.gif" alt="pic2" width="49%">
   <img src="https://github.com/Bariona/GH-NeRF/blob/main/static/desp_test.gif" alt="pic3" width="49%">
</div>
### Mini paper

see `./static/paper.pdf` in the project.

### Reference
1. NeRF
2. Mip-NeRF
3. MINE: Continuous-Depth MPI with Neural Radiance Fields
4. PlenOctrees
5. Plenoxels
6. [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
7. [mipnerf-pytorch](https://github.com/bebeal/mipnerf-pytorch)