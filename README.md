# CAR_adaptive_grid_sampler_numba
Content Adaptive Resampler (CAR) Adaptive Grid Sampler re-written with Numba

# Adapted from Sun and Chen
https://github.com/sunwj/CAR

```
@article{sun2020learned,
title={Learned image downscaling for upscaling using content adaptive resampler},
author={Sun, Wanjie and Chen, Zhenzhong},
journal={IEEE Transactions on Image Processing},
volume={29},
pages={4027--4040},
year={2020},
publisher={IEEE}
}
```

Unfortunately, the authors didn't seem to be responsive to requests for trainable code and have not updated their code to be compatible with newer versions of pytorch.

# Tested on Windows
* pytorch=1.11.0
* numba=1.23.0

# Advantages over original code
* No compiling shared libraries required that used old CUDA and pytorch version 1.3.1
* Trainable, just drop gridsampler.py in place of the other adaptive_gridsampler/gridsampler.py file

Good luck with your super-resolution projects!

