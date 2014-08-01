#include <stdio.h>
#include <pthread.h>
#include <iostream>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
Mat gray;

uint32_t paused;

//! updates the object tracking window using CAMSHIFT algorithm
extern int cvCamShift_d( const void* imgProb, CvRect windowIn,
    CvTermCriteria criteria,
    CvConnectedComp* _comp,
    CvBox2D* box );
extern void Canny_d( InputArray _src, OutputArray _dst,
    double low_thresh, double high_thresh,
    int aperture_size, bool L2gradient );

void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
    const Mat& descriptors1, const Mat& descriptors2,
    vector<DMatch>& filteredMatches12, int knn=1 )
{
  filteredMatches12.clear();
  vector<vector<DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
  descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );

  for( size_t m = 0; m < matches12.size(); m++ )
  {
    bool findCrossCheck = false;
    for( size_t fk = 0; fk < matches12[m].size(); fk++ )
    {
      DMatch forward = matches12[m][fk];

      for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
      {
        DMatch backward = matches21[forward.trainIdx][bk];
        if( backward.trainIdx == forward.queryIdx )
        {
          filteredMatches12.push_back(forward);
          findCrossCheck = true;
          break;
        }
      }
      if( findCrossCheck ) break;
    }
  }
}

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

  //namedWindow("back", WINDOW_AUTOSIZE );
  namedWindow("test", WINDOW_AUTOSIZE );
  namedWindow("track", WINDOW_AUTOSIZE );
  namedWindow("w1", WINDOW_NORMAL);
  //namedWindow("w2", WINDOW_AUTOSIZE);

  int findtarget = 2;
  char code;

  Mat frame, mask, hist, backproj;
  Mat last1, last2, diff, diff1, diff2, bin1, bin2, eage, gray_out;
  CvRect bord;
  CvRect trackWindow;
  RotatedRect trackBox;
  int index = 1;
  //index = atoi(argv[1]);
  Mat element = getStructuringElement(MORPH_RECT, Point(10,10));

  KalmanFilter KF(4, 2, 0);
  Mat measurement = Mat::zeros(2, 1, CV_32F);
  Mat prediction;
  KF.transitionMatrix = *(Mat_<float>(4, 4) <<\
      1, 0, 1, 0,\
      0, 1, 0, 1,\
      0, 0, 1, 0,\
      0, 0, 0, 1);

  setIdentity(KF.measurementMatrix, Scalar::all(1));
  setIdentity(KF.processNoiseCov, Scalar::all(1));
  setIdentity(KF.measurementNoiseCov, Scalar::all(2));
  setIdentity(KF.errorCovPost, Scalar::all(5));

  Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
  Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
  Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );

  vector<KeyPoint> keypoints1, keypoints2;
  vector<DMatch> filteredMatches;
  vector<int> queryIdxs, trainIdxs;
  vector<char> matchesMask;
  vector<Point2f> points1;
  vector<Point2f> points2;
  Mat descriptors1, descriptors2;

  //Mat mouse = imread("./mouse/mouse_tran.png", IMREAD_GRAYSCALE);
  //detector->detect( mouse, keypoints1 );
  //cout << "object" << keypoints1.size() << " points" << endl << ">" << endl;
  //descriptorExtractor->compute( mouse, keypoints1, descriptors1 );
  //findtarget = 0;
#define FIRST 58
  index = FIRST;

  while( index<500 )
  {
    sprintf(filename, "./mouse/mouse_Steel_1%03d.png", index++);
    printf("%s\n", filename);
    gray = imread(filename, IMREAD_GRAYSCALE);
    cvtColor(gray, gray_out, CV_GRAY2BGR);

    switch( findtarget )
    {
      case 2:
        gray.copyTo(last1);
        findtarget--;
        break;
      case 1:
        absdiff(gray, last1, diff);
        threshold(diff, bin1, 5, 255, CV_THRESH_BINARY);
        printf("bin1\n");
        imshow("test", bin1);
        waitKey(-1);
        dilate(bin1, bin2, element);
        erode(bin2, bin1, element);
        dilate(bin1, mask, element);
        printf("bin2\n");
        imshow("test", mask);
        waitKey(-1);
        bsize(mask, bord);
        printf("bord x:%d, y:%d, width:%d, height:%d\n", bord.x, bord.y, bord.width, bord.height);
        if( bord.width <= 0 || bord.height <= 0 )
        {
          //imwrite("error.bmp", bin2);
          gray.copyTo(last1);
          findtarget = 1;
          break;
        }
        findtarget --;
        trackWindow = bord;
        {
          Mat obj(gray, trackWindow);
          detector->detect( obj, keypoints1 );
          cout << "object" << keypoints1.size() << " points" << endl << ">" << endl;
          descriptorExtractor->compute( obj, keypoints1, descriptors1 );
          for( size_t i = 0; i < keypoints1.size(); i++ )
          {
            printf("KeyPoint %ld, x:%f, y:%f, size:%f, angle:%f\n", i,
                keypoints1[i].pt.x,
                keypoints1[i].pt.y,
                keypoints1[i].size,
                keypoints1[i].angle);
          }
        }
        printf("set KF start state\n");
        KF.statePost.at<float>(0) = bord.x+bord.width/2.f;
        KF.statePost.at<float>(1) = bord.y+bord.height/2.f;
        KF.statePost.at<float>(2) = 0;
        KF.statePost.at<float>(3) = 0;
      case 0:
        rectangle(gray_out, trackWindow, Scalar(0, 255, 0), 1);
        line(gray_out, Point2d(trackWindow.x, trackWindow.y),
            Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y + trackWindow.height - 1),
            Scalar(0, 255, 0), 1);
        line(gray_out, Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y),
            Point2d(trackWindow.x, trackWindow.y + trackWindow.height - 1),
            Scalar(0, 255, 0), 1);
        prediction = KF.predict();
        circle(gray_out, Point2f(prediction.at<float>(0), prediction.at<float>(1)), 3, Scalar(255, 0, 255), 1, CV_AA);
        printf("++++++++++predict; predict x:%f, y:%f\n", prediction.at<float>(0), prediction.at<float>(1));
        printf("trackWindow x:%d, y:%d, width:%d, height:%d\n",
            trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height);

        Mat roi_src(gray, trackWindow);
        detector->detect( roi_src, keypoints2 );
#define DELTA 60
        while( keypoints2.size() < 6 )
        {
          trackWindow.x -= DELTA;
          trackWindow.y -= DELTA;
          trackWindow.width += DELTA;
          trackWindow.height += DELTA;
          roi_src = gray(trackWindow);
          detector->detect( roi_src, keypoints2 );
        }
        Mat roi_out(gray_out, trackWindow);
        cout << keypoints2.size() << " points" << endl;
        descriptorExtractor->compute( roi_src, keypoints2, descriptors2 );

        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        queryIdxs.clear();
        trainIdxs.clear();
        printf("match %ld points\n", filteredMatches.size() );
        for( size_t i = 0; i < filteredMatches.size(); i++ )
        {
          printf("queryIdx %d, trainIdx %d, x:%f, y:%f, size:%f, angle:%f\n",
              filteredMatches[i].queryIdx,
              filteredMatches[i].trainIdx,
              keypoints2[filteredMatches[i].trainIdx].pt.x,
              keypoints2[filteredMatches[i].trainIdx].pt.y,
              keypoints2[filteredMatches[i].trainIdx].size,
              keypoints2[filteredMatches[i].trainIdx].angle);

          float angle = keypoints2[filteredMatches[i].trainIdx].angle;
          Point2f p = keypoints2[filteredMatches[i].trainIdx].pt;

          queryIdxs.push_back( filteredMatches[i].queryIdx );
          trainIdxs.push_back( filteredMatches[i].trainIdx );
          circle(roi_out, p, 3, Scalar(0, 255, 255), 1, CV_AA);
          line(roi_out, p, p + Point2f((float)cos(angle), (float)-sin(angle))*(float)30, Scalar(0, 0, 255), 1);
        }
        points1.clear();
        points2.clear();
        KeyPoint::convert(keypoints1, points1, queryIdxs);
        KeyPoint::convert(keypoints2, points2, trainIdxs);
        Mat status;
        printf("findHomography\n");
        Mat H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, 1 , mask);
        printf("leave findHomography\n");
        for(int i=0; i<mask.rows; i++)
        {
          printf("%d:%d ", i, mask.at<uchar>(i));
        }
        printf("\n");
        Point2f aver;
        for( size_t j = 0; j < points2.size(); j++)
        {
          //aver.x += points2[j].x;
          //aver.y += points2[j].y;
          aver+=points2[j];
        }
        imshow("w1", roi_out);
        aver.x /= points2.size();
        aver.y /= points2.size();
        printf("aver x:%f, y:%f\n", aver.x, aver.y);

        trackWindow.x = aver.x + trackWindow.x - DELTA;
        trackWindow.y = aver.y + trackWindow.y - DELTA;
        trackWindow.width = 2*DELTA;
        trackWindow.height = 2*DELTA;
        rectangle(gray_out, trackWindow, Scalar(0, 0, 255), 1);
        line(gray_out, Point2d(trackWindow.x, trackWindow.y),
            Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y + trackWindow.height - 1),
            Scalar(0, 0, 255), 1);
        line(gray_out, Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y),
            Point2d(trackWindow.x, trackWindow.y + trackWindow.height - 1),
            Scalar(0, 0, 255), 1);

        measurement.at<float>(0) = aver.x + trackWindow.x;
        measurement.at<float>(1) = aver.y + trackWindow.y;
        KF.correct(measurement);
        printf("++++++++++correct; measure x:%f, y:%f\n", measurement.at<float>(0), measurement.at<float>(1));
    }

    imshow("track", gray_out);
    code = waitKey(-1);
    if( code == '\x1b' ) // esc
      break;
  }

  /*
   *  while( index < 500 )
   *  {
   *    sprintf(filename, "mouse/mouse%04d.bmp", index++);
   *    printf("%s\n", filename);
   *    gray = imread(filename, IMREAD_GRAYSCALE);
   *    cvtColor(gray, gray_out, CV_GRAY2BGR);
   *
   *    switch( findtarget )
   *    {
   *      case 2: gray.copyTo(last1);
   *              findtarget--;
   *              break;
   *      case 1: {
   *                absdiff(gray, last1, diff);
   *                //absdiff(last1, last2, diff2);
   *                //imshow("w1", diff1);
   *                //waitKey(-1);
   *                //imshow("w2", diff2);
   *                //waitKey(-1);
   *                //diff = diff1 & diff2;
   *                //GaussianBlur(diff, diff, Size(3, 3), 0);
   *                threshold(diff, bin1, 5, 255, CV_THRESH_BINARY);
   *                printf("bin1\n");
   *                imshow("test", bin1);
   *                waitKey(-1);
   *                dilate(bin1, bin2, element);
   *                erode(bin2, bin1, element);
   *                dilate(bin1, mask, element);
   *                printf("bin2\n");
   *                imshow("test", mask);
   *                waitKey(-1);
   *                bsize(mask, bord);
   *                printf("bord %d, %d, %d, %d\n", bord.x, bord.y, bord.width, bord.height);
   *                if( bord.width <= 0 || bord.height <= 0 )
   *                {
   *                  //imwrite("error.bmp", bin2);
   *                  gray.copyTo(last1);
   *                  findtarget = 1;
   *                  break;
   *                }
   *
   *                Mat roi_gray(gray, bord);
   *                eage = Mat::zeros( gray.rows, gray.cols, CV_8UC1);
   *                Mat roi_eage(eage, bord);
   *                Canny_d(roi_gray, roi_eage, 1000, 4000, 5, 3);
   *
   *                vector< vector<Point> > contours;
   *                findContours(roi_eage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
   *                drawContours(roi_eage, contours, -1, Scalar(255), CV_FILLED);
   *                imshow("test", eage);
   *                waitKey(-1);
   *
   *                calcHist(&gray, 1, 0, eage, hist, 1, &hsize, &phranges);
   *                normalize(hist, hist, 0, 255, CV_MINMAX);
   *
   *                findtarget--;
   *                trackWindow = bord;
   *                rectangle(gray_out, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x+trackWindow.width, trackWindow.y+trackWindow.height), Scalar(255, 0, 255), 2);
   *                printf("set KF start state\n");
   *                KF.statePost.at<float>(0) = bord.x+bord.width/2.f;
   *                KF.statePost.at<float>(1) = bord.y+bord.height/2.f;
   *                KF.statePost.at<float>(2) = 0;
   *                KF.statePost.at<float>(3) = 0;
   *              }
   *      case 0: {
   *                calcBackProject( &gray, 1, 0, hist, backproj, &phranges );
   *                CvConnectedComp comp;
   *                CvBox2D box;
   *
   *                CvTermCriteria term;
*                term.max_iter = 100;
*                term.epsilon = 1;
*                term.type = 3;
*
*                CvRect rect = trackWindow;
*                rectangle(gray_out, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x+trackWindow.width, trackWindow.y+trackWindow.height), Scalar(0, 255, 255), 1);
*                unsigned int direction = 0;
*                printf("-------predict\n");
*                prediction = KF.predict();
*                circle(gray_out, Point2f(prediction.at<float>(0), prediction.at<float>(1)), 3, Scalar(255, 0, 0), 1, CV_AA);
*                printf("predict %f, %f\n", prediction.at<float>(0), prediction.at<float>(1));
*
*                int dx, dy;
*                dx = prediction.at<float>(0) - (trackWindow.x+trackWindow.width/2.f);
*                dy = prediction.at<float>(1) - (trackWindow.y+trackWindow.height/2.f);
*                printf("dx:%d, dy:%d\n", dx, dy);
*                if(dx < 0)
  *                {
    *                  rect.x = trackWindow.x + dx; 
    *                  rect.width = trackWindow.width - dx;
    *                }
    *                else
    *                {
      *                  rect.width = trackWindow.width + dx; 
      *                  direction |= 0x1;
      *                }
  *
*                if(dy < 0)
  *                {
    *                  rect.y = trackWindow.y + dy; 
    *                  rect.height = trackWindow.height - dy;
    *                }
    *                else
    *                {
      *                  rect.height = trackWindow.height + dy;
      *                  direction |= 0x2;
      *                }
      *
      *                CvMat c_probImage = backproj;
      *                rectangle(gray_out, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0, 255, 0), 1);
      *                printf("search x1:%d, y1:%d, x2:%d, y2:%d\n", rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);
      *                int ret = cvCamShift_d(&c_probImage, rect, term, &comp, &box);
      *                int iter = 4;
*                while(ret == -1 && iter)
  *                {
    *                  CvRect rect_alt = rect;
    *                  switch( (direction+iter)%4 )
      *                  {
        *                    case 0: rect_alt.x -= rect.width;
        *                            rect_alt.y -= rect.height;
        *                            break;
        *                    case 1: rect_alt.x -= rect.width;
        *                            rect_alt.y += rect.height;
        *                            break;
        *                    case 2: rect_alt.x += rect.width;
        *                            rect_alt.y -= rect.height;
        *                            break;
        *                    case 3: rect_alt.x += rect.width;
        *                            rect_alt.y += rect.height;
        *                            break;
        *                  }
    *                  printf("lost...search%d  x1:%d, y1:%d, x2:%d, y2:%d\n", iter, rect_alt.x, rect_alt.y, rect_alt.x+rect_alt.width, rect_alt.y+rect_alt.height);
    *                  rectangle(gray_out, Point(rect_alt.x, rect_alt.y), Point(rect_alt.x+rect_alt.width, rect_alt.y+rect_alt.height), Scalar(0, 255, 0), 1);
    *                  ret = cvCamShift_d(&c_probImage, rect_alt, term, &comp, &box);
    *                  iter--;
    *                }
*                if(ret == -1)
  *                {
    *                  printf("retry......\n");
    *                  gray.copyTo(last1);
    *                  findtarget = 1;
    *                  imshow("back", backproj);
    *                  break;
    *                }
    *                trackBox = RotatedRect(Point2f(box.center), Size2f(box.size), box.angle);
    *                trackWindow = comp.rect;
    *                printf("measure x:%d, y:%d\n", trackWindow.x+trackWindow.width/2, trackWindow.y+trackWindow.height/2);
    *
    *                measurement.at<float>(0) = trackBox.center.x;
    *                measurement.at<float>(1) = trackBox.center.y;
    *                printf("++++++++++correct\n");
    *                KF.correct(measurement);
    *
    *                ellipse(gray_out, trackBox, Scalar(0, 0, 255), 1, CV_AA);
    *                imshow("back", backproj);
    *              }
    *    }
*    imshow("CamShift", gray_out);
*    code = (char)waitKey(-1);
*    if(code == 'q')
  *      break;
*  }
*/

  return 0;
}


