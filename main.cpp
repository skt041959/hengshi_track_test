#include <stdio.h>
#include <pthread.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
Mat gray;
int vmin = 10, vmax = 256, smin = 30;

uint32_t paused;

//! updates the object tracking window using CAMSHIFT algorithm
extern int cvCamShift_d( const void* imgProb, CvRect windowIn,
        CvTermCriteria criteria,
        CvConnectedComp* _comp,
        CvBox2D* box );
extern void Canny_d( InputArray _src, OutputArray _dst,
        double low_thresh, double high_thresh,
        int aperture_size, bool L2gradient );

void bsize(Mat & src, CvRect & r)
{
    Mat t;
    int i,j;
    int b[4];
    t = Mat::zeros(1, src.cols, CV_8UC1);
    for(i=0; i<src.rows; i++)
        t += src.row(i);
    for(j=0; j<src.cols; j++)
        if(t.at<uchar>(0, j) == 0u && t.at<uchar>(0, j+1) == 255u)
        { b[0] = j; break; }
    for(j=src.cols-1; j>0; j--)
        if(t.at<uchar>(0, j) == 0u && t.at<uchar>(0, j-1) == 255u)
        { b[1] = j; break; }
    t = Mat::zeros(src.rows, 1, CV_8UC1);
    for(i=0; i<src.cols; i++)
        t += src.col(i);
    for(j=0; j<src.rows; j++)
        if(t.at<uchar>(j, 0) == 0u && t.at<uchar>(j+1, 0) == 255u)
        { b[2] = j; break; }
    for(j=src.rows-1; j>0; j--)
        if(t.at<uchar>(j, 0) == 0u && t.at<uchar>(j-1, 0) == 255u)
        { b[3] = j; break; }
    r.x = b[0];
    r.y = b[2];
    r.width = b[1]-b[0];
    r.height = b[3]-b[2];
}

int main(int argc, char * argv[])
{
    int hsize = 16;
    float hranges[] = {0,256};
    const float* phranges = hranges;
    //char uniqueName[128];
    char filename[128];

    namedWindow("back", WINDOW_NORMAL );
    namedWindow("test", WINDOW_NORMAL );
    namedWindow("CamShift", WINDOW_NORMAL );

    int findtarget = 2;
    int track = 0;
    char code;

    Mat frame, mask, hist, backproj;
    Mat last, diff, bin1, bin2, eage, gray_out;
    CvRect bord;
    CvRect trackWindow;
    RotatedRect trackBox;
    int index;
    index = atoi(argv[1]);

    KalmanFilter KF(4, 2, 0);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    Mat prediction;
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 1, 0,\
            0, 1, 0, 1,\
            0, 0, 1, 0,\
            0, 0, 0, 1);

    setIdentity(KF.measurementMatrix, Scalar::all(1));
    setIdentity(KF.processNoiseCov, Scalar::all(1));
    setIdentity(KF.measurementNoiseCov, Scalar::all(2));
    setIdentity(KF.errorCovPost, Scalar::all(5));

    while( index < 500 )
    {
        //printf("%d\n", index);
        //sprintf(filename, "CorpQtz0k_img/CorpQtz0k_img%04d.bmp", index++);
        //sprintf(filename, "CorUR1C8s_img/CorUR1C8s_img%04d.bmp", index++);
        sprintf(filename, "CorSTvjfE_img/CorSTvjfE_img%04d.bmp", index++);
        gray = imread(filename, IMREAD_GRAYSCALE);
        cvtColor(gray, gray_out, CV_GRAY2BGR);

        if( findtarget )
        {
            if( findtarget>1 )
            {
                gray.copyTo(last);
                findtarget--;
            }
            else
            {
                track = 0;
                absdiff(gray, last, diff);
                GaussianBlur(diff, diff, Size(3, 3), 0);
                threshold(diff, bin1, 30, 255, CV_THRESH_BINARY);
                printf("bin1\n");
                //imshow("test", bin1);
                //waitKey(-1);
                Mat element = getStructuringElement(MORPH_RECT, Point(10,10));
                dilate(bin1, bin2, element);
                erode(bin2, bin1, element);
                dilate(bin1, bin2, element);
                //printf("bin2\n");
                imshow("test", bin2);
                //waitKey(-1);
                bsize(bin2, bord);
                printf("bord %d, %d, %d, %d\n", bord.x, bord.y, bord.width, bord.height);
                if( bord.width <= 0 || bord.height <= 0 )
                {
                    //imwrite("error.bmp", bin2);
                    gray.copyTo(last);
                    findtarget = 1;
                    goto staticimg;
                }

                Mat roi_gray(gray, bord);
                eage = Mat::zeros( gray.rows, gray.cols, CV_8UC1);
                Mat roi_eage(eage, bord);
                Canny_d(roi_gray, roi_eage, 1000, 4000, 5, 3);

                vector< vector<Point> > contours;
                findContours(roi_eage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                drawContours(roi_eage, contours, -1, Scalar(255), CV_FILLED);
                calcHist(&roi_gray, 1, 0, roi_eage, hist, 1, &hsize, &phranges);
                normalize(hist, hist, 0, 255, CV_MINMAX);

                findtarget--;
                trackWindow = bord;
                rectangle(gray_out, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x+trackWindow.width, trackWindow.y+trackWindow.height), Scalar(255, 0, 255), 2);
                printf("set KF start state\n");
                KF.statePost.at<float>(0) = bord.x+bord.width/2.f;
                KF.statePost.at<float>(1) = bord.y+bord.height/2.f;
                KF.statePost.at<float>(2) = 0;
                KF.statePost.at<float>(3) = 0;
                goto cam;
            }
        }
        else
        {
cam:
            calcBackProject(&gray, 1, 0, hist, backproj, &phranges);
            CvConnectedComp comp;
            CvBox2D box;

            CvTermCriteria term;
            term.max_iter = 100;
            term.epsilon = 1;
            term.type = 3;

            CvRect rect = trackWindow;
            unsigned int direction = 0;
            //if(track != 0)
            {
                printf("-------predict\n");
                prediction = KF.predict();
                printf("predict %f, %f\n", prediction.at<float>(0), prediction.at<float>(1));

                int dx, dy;
                dx = prediction.at<float>(0) - trackBox.center.x;
                dy = prediction.at<float>(1) - trackBox.center.y;
                if(dx < 0)
                    rect.x = trackWindow.x + dx; 
                else
                    rect.width = trackWindow.width + dx; 

                if(dy < 0)
                    rect.y = trackWindow.y + dy; 
                else
                    rect.height = trackWindow.height + dy;
                if(dx>0)
                    direction |= 0x1;
                if(dy>0)
                    direction |= 0x2;
                circle(gray_out, Point2f(prediction.at<float>(0), prediction.at<float>(1)), 3, Scalar(255, 0, 0), 2, CV_AA);
            }

            CvMat c_probImage = backproj;
            rectangle(gray_out, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0, 255, 0), 2);
            printf("search x1:%d, y1:%d, x2:%d, y2:%d\n", rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);
            int ret = cvCamShift_d(&c_probImage, rect, term, &comp, &box);
            int iter = 4;
            while(ret == -1 && iter)
            {
                CvRect rect_alt = rect;
                switch( (direction+iter)%4 )
                {
                    case 0: rect_alt.x -= rect.width;
                            rect_alt.y -= rect.height;
                            break;
                    case 1: rect_alt.x -= rect.width;
                            rect_alt.y += rect.height;
                            break;
                    case 2: rect_alt.x += rect.width;
                            rect_alt.y -= rect.height;
                            break;
                    case 3: rect_alt.x += rect.width;
                            rect_alt.y += rect.height;
                            break;
                }
                printf("lost...search%d  x1:%d, y1:%d, x2:%d, y2:%d\n", iter, rect_alt.x, rect_alt.y, rect_alt.x+rect_alt.width, rect_alt.y+rect_alt.height);
                rectangle(gray_out, Point(rect_alt.x, rect_alt.y), Point(rect_alt.x+rect_alt.width, rect_alt.y+rect_alt.height), Scalar(0, 255, 0), 2);
                ret = cvCamShift_d(&c_probImage, rect_alt, term, &comp, &box);
                iter--;
            }
            if(ret == -1)
            {
                printf("retry......\n");
                gray.copyTo(last);
                findtarget = 1;
                goto lost;
            }
            track = 1;
            trackBox = RotatedRect(Point2f(box.center), Size2f(box.size), box.angle);
            trackWindow = comp.rect;
            printf("measure x:%d, y:%d\n", trackWindow.x+trackWindow.width/2, trackWindow.y+trackWindow.height/2);

            measurement.at<float>(0) = trackBox.center.x;
            measurement.at<float>(1) = trackBox.center.y;
            printf("++++++++++correct\n");
            KF.correct(measurement);

            ellipse(gray_out, trackBox, Scalar(0, 0, 255), 2, CV_AA);

lost:
            imshow("back", backproj);
            imshow("CamShift", gray_out);
        }
staticimg:
        code = (char)waitKey(-1);
        if(code == 'q')
            break;
    }
    return 0;
}


