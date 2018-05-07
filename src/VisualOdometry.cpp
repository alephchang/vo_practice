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

//#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include "gslam/Config.h"
#include "gslam/VisualOdometry.h"
#include "gslam/Optimizer.h"
#include "gslam/ORBmatcher.h"
namespace gslam
{
    
VO_TYPE VisualOdometry::voType_ = VO_UNKNOW;

const int TH_HIGH = 100;
const int TH_LOW = 50;
const int HISTO_LENGTH = 30;
VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), prev_(nullptr), map_ ( new Map ), numLost_ ( 0 ), numInliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orbLeft_.reset(new ORB_SLAM2::ORBextractor(num_of_features_,scale_factor_,4,20,7) );
    orbRight_.reset(new ORB_SLAM2::ORBextractor(num_of_features_,scale_factor_,4,20,7) );
}

VisualOdometry::~VisualOdometry()
{
    flog_.close();
}
void VisualOdometry::initialize()
{
    if( curr_->vKeys_.size() < 400) return;
    for ( size_t i=0; i<curr_->vKeys_.size(); i++ )
    {
        double d = curr_->getDepth( i );
        if ( d < 0 ) 
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y ), curr_->Tcw_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr pNewMp= MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_);
        pNewMp->addObservation(curr_, i);
        curr_->addMapPoint(pNewMp, i);
        pNewMp->computeDistinctiveDescriptors();
        pNewMp->updateNormalAndDepth();
        map_->insertMapPoint(pNewMp);
    }
    keyFrameIds_.push_back(curr_->id_);
    map_->insertKeyFrame ( curr_ );
    curr_->computeBoW();
    state_ = OK;
}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    flog_ << "==========Frame ID: " << frame->id_ << std::endl;
    switch ( state_ )
    {
    case INITIALIZING:
    {
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        detectAndComputeFeatures();
        initialize();
        break;
    }
    case OK:
    {
        curr_ = frame;
        
        //0. init guess
        curr_->Tcw_ = vel_*prev_->Tcw_;
        flog_ << "init guess translation: "<<curr_->Tcw_.translation().transpose() << std::endl;
        
        //1. detect features and match
        detectAndComputeFeatures();
        int nmatches = featureMatchingWithPrev();
        
        //2. estimate pose with matches
        int ngoodmatches = poseEstimationOptimization();
        flog_ << "estimated translation: "<<curr_->Tcw_.translation().transpose() << std::endl;
        flog_ << "inliers: " << ngoodmatches << " of matches: " << nmatches <<endl;
        
        //3. project the map to current frame, find more match by projection and optimize the pose. 
        trackLocalMap();
        
        curr_->Twc_ = curr_->Tcw_.inverse();//Twc is used in addKeyFrame()
        
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            validateProjection(); //for validation, dump the features with large error
            optimizeMap();      //erase the old map points.
            numLost_ = 0;
            //4. add key frame if necessary, local BA for the new key frame.
            if ( checkKeyFrame()){ // check a new key-frame
                addKeyFrame();
            }
        }
        else{ // bad estimation due to various reasons
            curr_->Tcw_ = vel_ * prev_->Tcw_; //use the init guess
            reInitializeFrame();
            numLost_++;
            if ( numLost_ > max_num_lost_ ){
                state_ = LOST;
            }
        }
        break;
    }
    case LOST:
    {
        flog_<<"vo has lost."<<endl;
        return false;
    }
    }
    //5. compute the vel, used for next pose guess
    if(prev_ != nullptr){
        vel_ = curr_->Tcw_*prev_->Tcw_.inverse(); 
        flog_ << "velocity: " << vel_.translation().transpose() << std::endl;
    }
    
    curr_->Twc_ = curr_->Tcw_.inverse(); //update Twc
    prev_ = curr_;
    
    flog_ << "There are "<< map_->mapPoints_.size()<<" map points, "
        <<"and "<< map_->keyframes_.size()<< " key frames"<< endl;
        
    if(map_->mapPoints_.empty()){
        state_ = LOST;
        flog_ << "tracking lost because of empty map at frame " << curr_->id_ << endl;
        return false;
    }
    else
        return true;
}

bool VisualOdometry::setLogFile(const std::string& logpath)
{
    flog_.open(logpath, fstream::out);
    return flog_.good();
}

void VisualOdometry::detectAndComputeFeatures()
{
    if(voType_ == VO_RGBD)//if use imDepth, assign orb right null
        orbRight_=nullptr;
    curr_->setORBextractor(orbLeft_, orbRight_);
    curr_->detectAndComputeFeatures();
}

int VisualOdometry::featureMatchingWithPrev()
{
    ORBmatcher matcher(0.8,true);
    vector<MapPoint::Ptr> vpMapPointMatches;
    size_t mapPointsCountInPrevFrame=0;
    for(size_t i = 0; i < prev_->vpMapPoints_.size(); ++i){
        if(prev_->vpMapPoints_[i] != nullptr)
                mapPointsCountInPrevFrame ++;
    }
    flog_ << "mapPointsCountInPrevFrame: " << mapPointsCountInPrevFrame << std::endl;
    
    int nmatch = matcher.searchByBoW(prev_, curr_, vpMapPointMatches);
    curr_->vpMapPoints_ = vpMapPointMatches;
    return nmatch;
}

int VisualOdometry::poseEstimationOptimization()
{
    numInliers_ = Optimizer::poseOptimization(curr_);
    // Discard outliers
    int nmatches = 0;
    for(int i =0; i<curr_->N_; i++){
        if(curr_->vpMapPoints_[i]){
            if(curr_->vbOutlier_[i]){
                curr_->vpMapPoints_[i]=nullptr;
                curr_->vbOutlier_[i]=false;
                //nmatches--;
            }
            else{
                nmatches++;
            }
        }
    }
    return nmatches;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( numInliers_ < min_inliers_ )
    {
        flog_<<"reject because inlier is too small: "<<numInliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3<double> T_r_c = prev_->Tcw_ * curr_->Tcw_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d translation = curr_->Tcw_.translation() - prev_->Tcw_.translation();
    flog_ << "motion change " << d.norm() << " translation: "<<translation.transpose()<< endl;
    if ( d.norm() > 4.0 ) {
        flog_<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    //1. few match
    if (numInliers_<50){
        return true;
    }
    SE3<double> T_r_c = ref_->Tcw_ * curr_->Tcw_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    //2. large motion from previous key frame
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}
void VisualOdometry::reInitializeFrame()
{
    ref_ = curr_;
    map_->mapPoints_.clear();
    for (size_t i = 0; i<curr_->vKeys_.size(); i++)
    {
        double d = curr_->getDepth(i);
        if (d < 0)
            continue;
        Vector3d p_world = ref_->camera_->pixel2world(
            Vector2d(curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y), curr_->Tcw_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr pNewMp = MapPoint::createMapPoint(
            p_world, n, curr_->descriptors_.row(i).clone(), curr_);
        map_->insertMapPoint(pNewMp);
        curr_->vpMapPoints_[i] = pNewMp;
        curr_->addMapPoint(pNewMp,i);
        pNewMp->addObservation(curr_, i);
        pNewMp->computeDistinctiveDescriptors();
    }
    keyFrameIds_.push_back(curr_->id_);
    map_->insertKeyFrame(curr_);
    prev_ = curr_;
    flog_ << "re init frame " << endl;
}

void VisualOdometry::addKeyFrame()
{
    //1. sort the point by depth
    vector<pair<double, size_t> > vDepthIdx;
    for(size_t i = 0; i < curr_->N_; ++i){
        double z = curr_->getDepth(i);
        if(z>0)
            vDepthIdx.push_back(make_pair(z, i));
    }
    //2. add the map point
    int nPts = 0;
    if(!vDepthIdx.empty()){
        sort(vDepthIdx.begin(),vDepthIdx.end());
        for(size_t j = 0; j < vDepthIdx.size(); ++j){
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;
            MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
            if(pMp==nullptr)
                bCreateNew = true;
            else if(pMp->observations_.empty()){
                bCreateNew = true;
                curr_->vpMapPoints_[i] = nullptr;
            }
            if(bCreateNew){
                Vector3d p_world = curr_->camera_->pixel2world(
                    Vector2d( curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y ),
                    curr_->Tcw_, vDepthIdx[j].first );
                Vector3d n = p_world - curr_->getCamCenter();
                n.normalize();
                MapPoint::Ptr pNewMp = MapPoint::createMapPoint(p_world, n, curr_->descriptors_.row(i).clone(), curr_);
                pNewMp->addObservation(curr_, i);
                curr_->addMapPoint(pNewMp, i);
                pNewMp->computeDistinctiveDescriptors();
                pNewMp->updateNormalAndDepth();
                map_->insertMapPoint(pNewMp);
            }
            nPts++;
            if(nPts>200)
                break;
        }
    }
    flog_ << "new points are added for new key frame: " << nPts << endl;
    
    keyFrameIds_.push_back(curr_->id_);
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
    curr_->computeBoW();
    ///3: local Bundle Adjustment
    map_->localMapping();
    
    flog_ << "translation after local BA: " << curr_->Tcw_.translation().transpose() << std::endl;
    for(size_t i = 0; i < curr_->orderedConnectedKeyFrames_.size(); ++i){
        flog_ << curr_->orderedWeights_[i] << "->" << curr_->orderedConnectedKeyFrames_[i]->id_ << std::endl;
    }
}

void VisualOdometry::validateProjection()
{
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp == nullptr) continue;
        Eigen::Vector3d pos = pMp->pos_;
        Eigen::Vector2d pix1 = curr_->camera_->world2pixel(pos, curr_->Tcw_);
        Eigen::Vector2d pix0(curr_->vKeys_[i].pt.x, curr_->vKeys_[i].pt.y);
        if ((pix1 - pix0).norm() > 5.0)
            flog_ << "large error for projection: " << pix1.transpose()
            << " " << pix0.transpose() << endl;
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points 
    size_t map_points_sz = map_->mapPoints_.size();
    int outframe = 0, lowmatchratio = 0, largeangle = 0;
    for ( auto iter = map_->mapPoints_.begin(); iter != map_->mapPoints_.end(); )
    {
        MapPoint::Ptr pMp = iter->second;
        if ( !curr_->isInFrame(iter->second->pos_) )  {
            iter = map_->mapPoints_.erase(iter);
            outframe++;
            continue;
        }
        float match_ratio = float(pMp->found_times_)/pMp->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ ) {
            iter = map_->mapPoints_.erase(iter);
            lowmatchratio++;
            continue;
        }
        if (pMp->last_frame_seen_ + 10 < curr_->id_){
            iter = map_->mapPoints_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, pMp );
        if ( angle > M_PI/6. )
        {
            iter = map_->mapPoints_.erase(iter);
            largeangle++;
            continue;
        }
        iter++;
    }
    
    if ( map_->mapPoints_.size() > 1000 && map_point_erase_ratio_<0.30)  
    {
        map_point_erase_ratio_ += 0.05;
    }
    else 
        map_point_erase_ratio_ = 0.1;
    flog_ << "map points size change: " << map_points_sz << " to " << map_->mapPoints_.size() 
        << " outframe "<<outframe << " lowmatchratio "<<lowmatchratio << " largeangle " <<largeangle<< endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

void VisualOdometry::updateLocalKeyFrames()
{
    map<Frame::Ptr, int> keyframeCounter;
    for(size_t i = 0; i < curr_->N_; ++i){
        if(curr_->vpMapPoints_[i]!=nullptr){
            MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
            if(!pMp->isBad()){
                const map<Frame::Ptr,size_t>& observations = pMp->observations_;
                for(map<Frame::Ptr,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else{
                curr_->vpMapPoints_[i]=nullptr;
            }
        }
    }
    if(keyframeCounter.empty()) return;
    int max = 0; 
    Frame::Ptr pKFmax = nullptr;
    vLocalKeyFrames_.clear();
    vLocalKeyFrames_.reserve(3*keyframeCounter.size());
    // All keyframes that observe a map point are included in the local map. 
    for(map<Frame::Ptr,int>::const_iterator it = keyframeCounter.begin(); it != keyframeCounter.end(); ++it){
        Frame::Ptr pKF = it->first;
        if(it->second>max){
            max = it->second;
            pKFmax = pKF;
        }
        vLocalKeyFrames_.push_back(pKF);
    }
}

void VisualOdometry::updateLocalMapPoints()
{
    vLocalMapPts_.clear();
    for(auto kf : vLocalKeyFrames_){
        const vector<MapPoint::Ptr> vMPs = kf->vpMapPoints_;
        for(auto mp : vMPs){
            if(mp==nullptr) continue;
            if(mp->track_ref_frame_==curr_->id_) continue;
            if(!mp->isBad()){
                vLocalMapPts_.push_back(mp);
                mp->track_ref_frame_ = curr_->id_;
            }
        }
    }
}

void VisualOdometry::searchLocalMapPoints()
{
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp==nullptr) continue;
        if(pMp->isBad()){
            curr_->vpMapPoints_[i]==nullptr;
        }
        else{
            pMp->increaseVisible();
            pMp->last_frame_seen_ = curr_->id_;
            pMp->track_in_view_ = false;
        }
    }
    int nTomatch = 0;
    for(vector<MapPoint::Ptr>::iterator vit=vLocalMapPts_.begin(), vend=vLocalMapPts_.end(); vit!=vend; vit++){
        MapPoint::Ptr pMp = *vit;
        if(pMp->last_frame_seen_ == curr_->id_) continue;
        if(pMp->isBad()) continue;
        if(curr_->isInFrustum(pMp)){
            pMp->increaseVisible();
            nTomatch++;
        }
    }
    if(nTomatch>0){
        ORBmatcher matcher(0.8);
        int nmatch = matcher.searchByProjection(curr_,vLocalMapPts_,1.2);
        flog_ << "new match found by projection: " << nmatch << std::endl;
    }
}

bool VisualOdometry::trackLocalMap()
{
    updateLocalKeyFrames();
    updateLocalMapPoints();
    searchLocalMapPoints();
    numInliers_ = Optimizer::poseOptimization(curr_);
    flog_ << "translation after track local map: "<<curr_->Tcw_.translation().transpose() << std::endl;
    int ngoodmatch = 0, nmatch = 0;
    for(size_t i = 0; i < curr_->vpMapPoints_.size(); ++i){
        MapPoint::Ptr pMp = curr_->vpMapPoints_[i];
        if(pMp){
            nmatch ++;
            if(curr_->vbOutlier_[i]){
                curr_->vpMapPoints_[i]=nullptr;
            }
            else{
                if(pMp->observations_.empty()==false)
                    ngoodmatch++;
                pMp->found_times_++;
                pMp->last_frame_seen_ = curr_->id_;
            }
        }
    }
    flog_ << "good match: " << ngoodmatch << " total match: " << nmatch << std::endl;
    if(ngoodmatch < 30)
        return false;
    else
        return true;
}


}
