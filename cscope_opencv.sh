echo "cscope"
cscope -Rbq -I /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/include \
            -I /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/examples/common \
            -I /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/classes/basic \
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/core/include \
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/imgproc/include \
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/video/include \
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/highgui/include\
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/nonfree/include\
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/legacy/include\
            -I /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/features2d/include \
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/core/src \
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/imgproc/src \
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/video/src \
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/highgui/src\
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/nonfree/src\
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/legacy/src\
            -s /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/features2d/src

echo "ctags"
ctags -R /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/include \
         /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/examples/common \
         /home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera/classes/basic \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/core \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/imgproc \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/video \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/highgui \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/nonfree \
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/legacy\
         /home/skt/code/opencv-learning-and-transplantation/opencv-2.4.8/modules/features2d
