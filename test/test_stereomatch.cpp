#include "gslam/Config.h"
#include "gslam/Frame.h"
#include<opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
//#include "opencv2/contrib/contrib.hpp"  
#include <iostream>
using namespace cv;  
std::string imgdir = "/work/download/kitti/dataset/sequences/14/";
void stereoMatchByOpenCV(Mat imLeft, Mat imRight, Mat& imDisparity)
{
    Size img_size = imLeft.size();  
    int numberOfDisparities = ((img_size.width / 8) + 15) & -16;  
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);
    Rect roi1, roi2;  
    Mat Q;  
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setSpeckleWindowSize(100);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
    bm->compute(imLeft, imRight, imDisparity);  
    
    namedWindow("left", 1);  
    imshow("left", imLeft);  
  
    namedWindow("right", 1);  
    imshow("right", imRight);  
  
    namedWindow("disparity", 1);  
    imshow("disparity", imDisparity);  
  
    imwrite("result.png", imDisparity);  
    std::cout << "press any key to continue..." << std::endl;
    waitKey();  
}

int main()
{
    std::string configfile = imgdir+"config.yaml";
    std::cout<< configfile << std::endl;
    gslam::Config::setParameterFile (configfile);
    std::shared_ptr<gslam::Camera> camera(new gslam::Camera);
    std::string imgdirLeft = imgdir + "image_0/000000.png";
    std::string imgdirRight = imgdir + "image_1/000000.png";
    Mat imLeft, imRight;
    Mat color = cv::imread (imgdirLeft);
    cvtColor(color, imLeft, CV_RGB2GRAY);
    color = cv::imread (imgdirRight);
    cvtColor(color, imRight, CV_RGB2GRAY);
    gslam::Frame::Ptr pFrame = gslam::Frame::createFrame();
    pFrame->imLeft_ = imLeft;
    pFrame->imRight_ = imRight;
    pFrame->camera_ = camera;
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbLeft(new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7) );
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbRight(new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7) );
    pFrame->setORBextractor(orbLeft, orbRight);
    pFrame->detectAndComputeFeatures();
    Mat imDisparity;
    stereoMatchByOpenCV(imLeft,imRight, imDisparity);
    imDisparity.convertTo(imDisparity, CV_32F, 1.0/16);
    std::cout << imDisparity.type() << std::endl;
    for(size_t i = 0; i < pFrame->vKeys_.size(); ++i){
        if(pFrame->uRight_[i]>0){
            int ux = pFrame->vKeys_[i].pt.x;
            int uy = pFrame->vKeys_[i].pt.y;
            //std::cout << pFrame->vKeys_[i].pt.x - pFrame->uRight_[i] << " " <<imDisparity.at<float>(uy, ux) << std::endl;
        }
    }
    
}