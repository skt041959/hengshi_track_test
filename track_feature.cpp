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
  //namedWindow("test", WINDOW_AUTOSIZE );
  namedWindow("track", WINDOW_AUTOSIZE );
  namedWindow("w1", WINDOW_NORMAL);
  //namedWindow("w2", WINDOW_AUTOSIZE);

  int findtarget = 2;
  char code;

  Mat frame, mask, hist, backproj;
  Mat last, last2, diff, diff1, diff2, bin1, bin2, eage, gray_out;
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
  vector<Point2f> point_last;
  vector<Point2f> point_now;
  vector<Point2f> point_predict;
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
        gray.copyTo(last);
        findtarget--;
        break;
      case 1:
        absdiff(gray, last, diff);
        threshold(diff, bin1, 5, 255, CV_THRESH_BINARY);
        printf("bin1\n");
        //imshow("test", bin1);
        //waitKey(-1);
        dilate(bin1, bin2, element);
        erode(bin2, bin1, element);
        dilate(bin1, mask, element);
        printf("bin2\n");
        //imshow("test", mask);
        //waitKey(-1);
        bsize(mask, bord);
        printf("bord x:%d, y:%d, width:%d, height:%d\n", bord.x, bord.y, bord.width, bord.height);
        if( bord.width <= 0 || bord.height <= 0 )
        {
          //imwrite("error.bmp", bin2);
          gray.copyTo(last);
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
#define DELTA 60
        if( !point_last.empty() )
        {
          TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
          vector<uchar> sta;
          vector<float> err;

          printf("calcOpticalFlowPyrLK\n");
          int maxlevel = 3;
          int valid_sum = 0;
          do {
            calcOpticalFlowPyrLK(last, gray, point_last, point_now, sta, err,
                Size(30, 30), maxlevel, termcrit, 1, 0.001);
            point_predict.clear();
            for( size_t i=0; i<point_now.size(); i++ )
            {
              if( sta[i] != 0 && err[i] <= 1 )
              {
                valid_sum += 1;
                point_predict.push_back(point_now[i]);
              }
            }
            maxlevel ++;
            //if( index>=94 )
            //{
            //  for(size_t i=0; i<point_now.size(); i++)
            //  {
            //    printf("s:%d, optical x:%.2f,y:%.2f -> x:%.2f,y:%.2f; e:%f\n",
            //        sta[i], point_last[i].x, point_last[i].y, point_now[i].x, point_now[i].y, err[i]);
            //  }
            //  waitKey(-1);
            //}
          } while(valid_sum < 1);//FIXME:lost

          for(size_t i=0; i<point_now.size(); i++)
          {
            printf("s:%d, optical x:%.2f,y:%.2f -> x:%.2f,y:%.2f; e:%f\n",
                sta[i], point_last[i].x, point_last[i].y, point_now[i].x, point_now[i].y, err[i]);
            circle(gray_out, point_now[i], 3, Scalar(0, 0, 255), 1);
            line(gray_out, point_now[i], point_last[i], Scalar(0, 0, 255), 1);
          }
          Point2f _sum;
          for(size_t i=0; i<point_predict.size(); i++)
          {
            _sum += point_predict[i];
          }
          trackWindow.x = _sum.x / point_predict.size() -DELTA/2;
          trackWindow.y = _sum.y / point_predict.size() -DELTA/2;
          trackWindow.width = trackWindow.x + DELTA > gray.cols ? gray.cols - trackWindow.x : DELTA;
          trackWindow.height = trackWindow.y + DELTA > gray.rows ? gray.rows - trackWindow.y : DELTA;

          gray.copyTo(last);
          //cv::swap(last, gray);
          //std::swap(point_last, point_now);
        }
        //trackWindow.
        rectangle(gray_out, trackWindow, Scalar(0, 255, 0), 1);
        //line(gray_out, Point2d(trackWindow.x, trackWindow.y),
        //    Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y + trackWindow.height - 1),
        //    Scalar(0, 255, 0), 1);
        //line(gray_out, Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y),
        //    Point2d(trackWindow.x, trackWindow.y + trackWindow.height - 1),
        //    Scalar(0, 255, 0), 1);
        prediction = KF.predict();
        circle(gray_out, Point2f(prediction.at<float>(0), prediction.at<float>(1)), 3, Scalar(255, 0, 255), 1, CV_AA);
        printf("++++++++++predict; predict x:%f, y:%f\n", prediction.at<float>(0), prediction.at<float>(1));
        printf("trackWindow x:%d, y:%d, width:%d, height:%d\n",
            trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height);

        Mat roi_src(gray, trackWindow);
        detector->detect( roi_src, keypoints2 );
        //TODO:detect lost
        //while( keypoints2.size() < 6 )
        //{
        //  trackWindow.x -= DELTA;
        //  trackWindow.y -= DELTA;
        //  trackWindow.width += DELTA;
        //  trackWindow.height += DELTA;
        //  roi_src = gray(trackWindow);
        //  detector->detect( roi_src, keypoints2 );
        //}
        Mat roi_out(gray_out, trackWindow);
        cout << keypoints2.size() << " points" << endl;
        descriptorExtractor->compute( roi_src, keypoints2, descriptors2 );

        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        queryIdxs.clear();
        trainIdxs.clear();
        printf("match %ld points\n", filteredMatches.size() );
        for( size_t i=0; i<filteredMatches.size(); i++ )
        {
          printf("queryIdx %d, trainIdx %d, x:%f, y:%f, size:%f, angle:%f\n",
              filteredMatches[i].queryIdx,
              filteredMatches[i].trainIdx,
              keypoints2[filteredMatches[i].trainIdx].pt.x,
              keypoints2[filteredMatches[i].trainIdx].pt.y,
              keypoints2[filteredMatches[i].trainIdx].size,
              keypoints2[filteredMatches[i].trainIdx].angle);

          Point2f p = keypoints2[filteredMatches[i].trainIdx].pt;

          queryIdxs.push_back( filteredMatches[i].queryIdx );
          trainIdxs.push_back( filteredMatches[i].trainIdx );
          circle(roi_out, p, 3, Scalar(0, 255, 255), 1);
          //float angle = keypoints2[filteredMatches[i].trainIdx].angle;
          //line(roi_out, p, p + Point2f((float)cos(angle), (float)-sin(angle))*(float)30, Scalar(0, 0, 255), 1);
        }
        //for( size_t i=0; i<point_now.size(); i++)
        //{
        //  circle(roi_out, point_now[i], 3, Scalar(0, 0, 255), 1);
        //}
        points1.clear();
        points2.clear();
        KeyPoint::convert(keypoints1, points1, queryIdxs);
        KeyPoint::convert(keypoints2, points2, trainIdxs);
        //point_last = points2;
        point_last.clear();
        point_last.push_back(points2[0] + Point2f(trackWindow.x, trackWindow.y));
        for( size_t i=1; i<points2.size(); i++ )
        {
          if( points2[i] != points2[i-1] )
            point_last.push_back(points2[i] + Point2f(trackWindow.x, trackWindow.y));
        }
        Mat status;
        printf("findHomography\n");
        Mat H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, 1 , mask);
        printf("leave findHomography\n");
        //for(int i=0; i<mask.rows; i++)
        //{
        //  printf("%d:%d ", i, mask.at<uchar>(i));
        //}
        printf("\n");
        Point2f aver;
        for( size_t j = 0; j < points2.size(); j++)
        {
          aver+=points2[j];
        }
        imshow("w1", roi_out);
        aver.x /= points2.size();
        aver.y /= points2.size();
        //printf("aver x:%f, y:%f\n", aver.x, aver.y);

        //trackWindow.x = aver.x + trackWindow.x - DELTA;
        //trackWindow.y = aver.y + trackWindow.y - DELTA;
        //trackWindow.width = 2*DELTA;
        //trackWindow.height = 2*DELTA;
        //rectangle(gray_out, trackWindow, Scalar(0, 0, 255), 1);
        //line(gray_out, Point2d(trackWindow.x, trackWindow.y),
        //    Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y + trackWindow.height - 1),
        //    Scalar(0, 0, 255), 1);
        //line(gray_out, Point2d(trackWindow.x + trackWindow.width - 1, trackWindow.y),
        //    Point2d(trackWindow.x, trackWindow.y + trackWindow.height - 1),
        //    Scalar(0, 0, 255), 1);

        measurement.at<float>(0) = aver.x + trackWindow.x;
        measurement.at<float>(1) = aver.y + trackWindow.y;
        KF.correct(measurement);
        printf("----------correct; measure x:%f, y:%f\n", measurement.at<float>(0), measurement.at<float>(1));
    }

    imshow("track", gray_out);
    code = waitKey(-1);
    if( code == '\x1b' ) // esc
      break;
  }

  return 0;
}


