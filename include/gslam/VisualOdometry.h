/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "gslam/common_include.h"
#include "gslam/Map.h"
#include "ORBextractor.h"

#include <opencv2/features2d/features2d.hpp>
#include <fstream>
namespace gslam 
{
enum VO_TYPE{
    VO_STEREO = 0,
    VO_RGBD,
    VO_UNKNOW
};    
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    
    Frame::Ptr  ref_;       // reference key-frame 
    Frame::Ptr  curr_;      // current frame 
    Frame::Ptr  prev_;      // last frame    
    Sophus::SE3d vel_;
    vector<unsigned long> keyFrameIds_;
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbLeft_;  // orb detector and computer 
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbRight_;  // orb detector and computer 
    
    int numInliers_;        // number of inlier features in icp
    int numLost_;           // number of lost times
    
    vector<Frame::Ptr> vLocalKeyFrames_; //the neighbor keyframes of frame curr_, used for local pose optimization
    vector<MapPoint::Ptr> vLocalMapPts_; //the 
    //
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double map_point_erase_ratio_; // remove map point ratio

    //log file output
    std::ofstream flog_;
    static VO_TYPE voType_;
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 

    bool setLogFile(const std::string& logpath);
    void dumpMapAndKeyFrames();
    
protected:  
    // inner operation 
    void detectAndComputeFeatures();
    void featureMatching();
    int featureMatchingWithPrev();
    void poseEstimationPnP(); 
    int poseEstimationOptimization();
    bool trackLocalMap(); //based on the pose estimation, find more match between map and keypoints;
    void optimizeMap();
    
    void addKeyFrame();
    void triangulateForNewKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );

    void validateProjection();
    void optimizePnP(const vector<cv::Point3f>& pts3d, const vector<cv::Point2f>& pts2d, Mat& inliers,
            const Mat& rvec, const Mat& tvec);
    void reInitializeFrame();
    
    void updateLocalKeyFrames();
    void updateLocalMapPoints();
    void searchLocalMapPoints();
    
    void initialize();
    
};
}

#endif // VISUALODOMETRY_H
