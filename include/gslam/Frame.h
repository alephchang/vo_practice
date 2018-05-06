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

#ifndef FRAME_H
#define FRAME_H

#include "gslam/common_include.h"
#include "gslam/Camera.h"
#include "gslam/ORBVocabulary.h"
#include "gslam/ORBextractor.h"
#include "gslam/MapPoint.h"
#include "3rdparty/DBoW2/DBoW2/BowVector.h"
#include "3rdparty/DBoW2/DBoW2/FeatureVector.h"
#include<map>
using gslam::MapPoint;
namespace gslam 
{
    
// forward declare 
class Frame : public std::enable_shared_from_this< Frame >
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long                  id_;         // id of this frame
    double                         timeStamp_; // when it is recorded
    SE3<double>                    Tcw_;      // transform from world to camera
    SE3<double>                    Twc_;      // transform from camera to world, get camera center;       
    Camera::Ptr                    camera_;     // Pinhole RGBD Camera model 
    Mat                            imLeft_, imDepth_; // color and depth image 
    Mat                            imRight_;     
    std::vector<float>             uRight_;
    std::vector<float>             uDepth_;
    std::vector<gslam::MapPoint::Ptr>     vpMapPoints_;  // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<cv::KeyPoint>      vKeys_;
    std::vector<cv::KeyPoint>      vKeysRight_;
    cv::Mat                        descriptors_;
    cv::Mat                        descriptorsRight_;
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbLeft_;  // orb detector and computer 
    std::shared_ptr<ORB_SLAM2::ORBextractor> orbRight_;  // orb detector and computer 
    // Bag of Words Vector structures.
    DBoW2::BowVector               BowVec_;
    DBoW2::FeatureVector           featVec_;
    static shared_ptr<ORBVocabulary>  pORBvocab_;
    vector<bool>                   vbOutlier_;
    int                            N_;
    vector<float>                  vInvLevelSigma2_;
    //for key frames connection
    std::map<Frame::Ptr,int>    connectedKeyFrameWeights_;
    std::vector<Frame::Ptr>     orderedConnectedKeyFrames_;
    std::vector<int>               orderedWeights_;
    size_t                      fuseTargetFrameId_;
    size_t                      baId_;
    size_t                      baFixedId_;
public: // data members 
    Frame();
    Frame( long id, double time_stamp=0, SE3<double> Tcw=SE3<double>(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
    ~Frame();
    
    static Frame::Ptr createFrame(); 
    
    void setORBextractor(std::shared_ptr<ORB_SLAM2::ORBextractor> orbLeft, std::shared_ptr<ORB_SLAM2::ORBextractor> orbRight=nullptr);
    void detectAndComputeFeatures();
    void computeStereoMatches();
    void collectDetphFromImDetph();
    inline double getDepth(size_t kpIdx){
        return uDepth_[kpIdx];
    }
    
    // Get Camera Center
    Vector3d getCamCenter() const;
    
    void setPose( const SE3<double>& T_c_w );
    
    // check if a point is in this frame 
    bool isInFrame( const Vector3d& pt_world );
    bool isInFrame( const Vector2d& pD2c);
    
    vector<size_t> getFeaturesInAera(float x, float y, float r) const;

    bool isInFrustum(MapPoint::Ptr pMp);
    void addMapPoint(MapPoint::Ptr pMp, size_t i);
    void sortMapPoint2d();
    void computeBoW();
    bool isBad(){return false;}
    vector<MapPoint::Ptr> getMapPointMatches();
    void updateConnections();
    void addConnection(Frame::Ptr frame, const int& weight);
    void updateBestCovisibles();
    vector<Frame::Ptr> getBestCovisibilityKeyFrames(int n);
    void setBadFlag();
    void eraseConnection(Frame::Ptr frame);
};

}

#endif // FRAME_H
