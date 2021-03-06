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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "gslam/common_include.h"

namespace gslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    size_t id_;        // ID
    static unsigned long factory_id_;    // factory id
    bool        good_;      // wheter a good point 
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    
//    list<Frame*>    observed_frames_;   // key-frames that can observe this point 
    std::map<std::shared_ptr<Frame>, size_t> observations_;
    size_t baId_;
    size_t last_frame_seen_;
    size_t first_frame_seen_;
    size_t track_ref_frame_;
    bool track_in_view_;
    float track_proj_x_ ;
    float track_proj_y_ ;
    float track_view_cos_;  
    
    int         found_times_;     // being an inliner in pose estimation
    int         visible_times_;     // being visible in current frame 
    
    size_t      fuseCandidateForKF_;
    MapPoint();
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, 
        const Vector3d& norm, 
        std::shared_ptr<Frame> frame=nullptr, 
        const Mat& descriptor=Mat() 
    );
    
    inline cv::Point3f getPositionCV() const {
        return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
    }
    
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint( 
        const Vector3d& pos_world, 
        const Vector3d& norm_,
        const Mat& descriptor,
        std::shared_ptr<Frame> frame );
    inline bool isBad(){
        return !good_;
    }
    void increaseVisible(int n = 1){
        visible_times_ += n;
    }
    cv::Mat getDescriptor(){return descriptor_;}
    void computeDistinctiveDescriptors();
    void addObservation(std::shared_ptr<Frame>, size_t i);
    void updateNormalAndDepth();
    bool isInFrame(std::shared_ptr<Frame>);
    bool replace(MapPoint::Ptr pMP);
    int getIndexInFrame(std::shared_ptr<Frame> pKF);
    void eraseObservation(std::shared_ptr<Frame> pKF);
};
}

#endif // MAPPOINT_H
