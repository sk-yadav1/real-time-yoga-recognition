## Yoga Pose Recognition
#### Code repository for the "Real-time Yoga recognition using deep learning", Neural Computing and Applications (NCAA), Springer, 2019.

### Paper:
   https://link.springer.com/article/10.1007/s00521-019-04232-7

### Weights and converted training files:
   https://drive.google.com/drive/folders/1rxPzNUl585lmWYeKMpEhIbz6Gx8liHx5?usp=sharing

### Video dataset used to train the model:
   https://archive.org/details/YogaVidCollected

### To run it on your system:

1. Install and setup OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose
2. Run the OpenPose on webcam mode, and direct the results to \output folder.
3. Run prediction and write output to file python predictSeq.py > export.txt
   or Run prediction and show output on terminal python predictSeq.py

Feel free to adjust predictSeq.py input folder and the weights. You can also use the video files in your project. 
Contact us for any issues not resolved by adjusting file paths.

### Real-time demo:

[![Demo](https://img.youtube.com/vi/C_629XEmd1Q/maxresdefault.jpg)](https://youtu.be/C_629XEmd1Q)

### Please cite as:

Yadav, S. K., Singh, A., Gupta, A., & Raheja, J. L. (2019). Real-time Yoga recognition using deep learning. Neural Computing and Applications, 31(12), 9349-9361.

### Bibtex:

@article{yadav2019real,
  title={Real-time Yoga recognition using deep learning},
  author={Yadav, Santosh Kumar and Singh, Amitojdeep and Gupta, Abhishek and Raheja, Jagdish Lal},
  journal={Neural Computing and Applications},
  volume={31},
  number={12},
  pages={9349--9361},
  year={2019},
  publisher={Springer}
}
