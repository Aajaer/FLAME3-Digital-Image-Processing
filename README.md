Build OPENCV
Required Cuda 11.5, cudNN 9.3.0, OpenCV

For OpenCV:
step 1:

cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && git checkout 4.11.0
cd ../opencv_contrib && git checkout 4.11.0

cd ~/opencv
mkdir build && cd build


cmake -D CMAKE_C_COMPILER=gcc-10 \
      -D CMAKE_CXX_COMPILER=g++-10 \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D BUILD_opencv_python3=ON \
      -D BUILD_EXAMPLES=OFF ..

In directory build from OpenCV in terminal build it with:
make -j$(nproc)

