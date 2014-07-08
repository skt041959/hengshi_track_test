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

void bsize(Mat & src, Rect & r)
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

void fill(Mat & src)
{
    int i,j;
    bool in;
    for(i=0; i<src.rows; i++)
    {
        in = false;
        for(j=0; j<src.cols; j++)
        {
            if(src.at<uchar>(i, j+1)==255u && src.at<uchar>(i, j)==0u)
            {
                if(in == false)
                {
                    in = true;
                    continue;
                }
                else
                {
                    in = false;
                }
            }
            if(in == true)
            {
                if(src.at<uchar>(i, j)==255u && src.at<uchar>(i, j+1)==0u)
                    in = false;
                else
                    src.at<uchar>(i, j) = 255u;
            }
        }
    }
}

int main(int argc, char * argv[])
{
    char code;

    Rect trackWindow;
    int hsize = 16;
    int index;
    float hranges[] = {0,256};
    const float* phranges = hranges;
    char uniqueName[128];
    char filename[128];

    namedWindow("CamShift", WINDOW_AUTOSIZE );
    namedWindow("back", WINDOW_AUTOSIZE );
    namedWindow("test", WINDOW_AUTOSIZE );
    int findtarget = 2;

    Mat frame, mask, hist, backproj;
    Mat last, diff, bin1, bin2, eage, gray_out;
    Rect bord;
    index = atoi(argv[1]);

    KalmanFilter KF(4, 2, 0);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    Mat prediction;
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 1, 0,\
            0, 1, 0, 1,\
            0, 0, 1, 0,\
            0, 0, 0, 1);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    while( index < 200 )
    {
        //sprintf(filename, "CorpQtz0k_img/CorpQtz0k_img%04d.bmp", index++);
        sprintf(filename, "CorUR1C8s_img/CorUR1C8s_img%04d.bmp", index++);
        printf("%d\n", index);
        gray = imread(filename, IMREAD_GRAYSCALE);

        if( findtarget )
        {
            if( findtarget>1 )
            {
                gray.copyTo(last);
                findtarget--;
            }
            else
            {
                absdiff(gray, last, diff);
                GaussianBlur(diff, diff, Size(3, 3), 0);
                threshold(diff, bin1, 30, 255, CV_THRESH_BINARY);
                printf("bin1\n");
                imshow("test", bin1);
                waitKey(-1);
                Mat element = getStructuringElement(MORPH_RECT, Point(10,10));
                dilate(bin1, bin2, element);
                erode(bin2, bin1, element);
                dilate(bin1, bin2, element);
                printf("bin2\n");
                imshow("test", bin2);
                waitKey(-1);
                bsize(bin2, bord);
                printf("%d, %d, %d, %d\n", bord.x, bord.y, bord.width, bord.height);

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
                KF.statePost.at<float>(0) = bord.x+bord.width/2;
                KF.statePost.at<float>(1) = bord.y+bord.height/2;
                KF.statePost.at<float>(2) = 0;
                KF.statePost.at<float>(3) = 0;
                goto cam;
            }
        }
        else
        {
cam:
            calcBackProject(&gray, 1, 0, hist, backproj, &phranges);
            //RotatedRect trackBox = CamShift_d(backproj, trackWindow, term);
            CvConnectedComp comp;
            CvBox2D box;

            CvTermCriteria term;
            term.max_iter = 100;
            term.epsilon = 1;
            term.type = 3;

            CvRect rect;
            prediction = KF.predict();
            printf("measure x:%d, y:%d, width:%d, height:%d\n", trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height);
            printf("prediction %f, %f\n", prediction.at<float>(0), prediction.at<float>(1));

            if(prediction.at<float>(0) < trackWindow.x)
            {
                rect.x = prediction.at<float>(0);
                rect.width = trackWindow.width + trackWindow.x - prediction.at<float>(0);
            }
            else if(prediction.at<float>(0) > (trackWindow.x+trackWindow.width))
            {
                rect.x = trackWindow.x;
                rect.width = prediction.at<float>(0) - trackWindow.x;
            }
            else
            {
                rect.x = trackWindow.x;
                rect.width = trackWindow.width;
            }

            if(prediction.at<float>(1) < trackWindow.y)
            {
                rect.y = prediction.at<float>(1);
                rect.height = trackWindow.height + trackWindow.y - prediction.at<float>(1);
            }
            else if(prediction.at<float>(1) > (trackWindow.y+trackWindow.height))
            {
                rect.y = trackWindow.y;
                rect.height = prediction.at<float>(1) - trackWindow.y;
            }
            else
            {
                rect.y = trackWindow.y;
                rect.height = trackWindow.height;
            }

            //Mat roi_gray(gray, bord);

            CvMat c_probImage = backproj;
            int ret = cvCamShift_d(&c_probImage, rect, term, &comp, &box);
            if(ret == -1)
            {
                printf("lost......\n");
                code = getchar();
            }
            trackWindow = comp.rect;
            RotatedRect trackBox = RotatedRect(Point2f(box.center), Size2f(box.size), box.angle);
            printf("x:%d, y:%d, width:%d, height:%d\n", trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height);

            measurement.at<float>(0) = trackBox.center.x;
            measurement.at<float>(1) = trackBox.center.y;
            KF.correct(measurement);

            cvtColor(gray, gray_out, CV_GRAY2BGR);
            ellipse(gray_out, trackBox, Scalar(0, 0, 255), 1, CV_AA);
            imshow("back", backproj);
            imshow("CamShift", gray_out);
        }
        code = (char)waitKey(-1);
        if(code == 'q')
            break;
    }
    return 0;
}


