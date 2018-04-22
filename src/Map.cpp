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
#include "gslam/Map.h"
#include "gslam/ORBmatcher.h"
#include <gslam/Optimizer.h>

namespace gslam
{

void Map::insertKeyFrame ( Frame::Ptr frame )
{
    cout<<"Key frame size = "<<keyframes_.size()<<endl;
    if ( keyframes_.find(frame->id_) == keyframes_.end() )
    {
        keyframes_.insert( make_pair(frame->id_, frame) );
    }
    else
    {
        keyframes_[ frame->id_ ] = frame;
    }
    currKF_ = frame;
}

void Map::insertMapPoint ( MapPoint::Ptr mapPoint )
{
    if ( mapPoints_.find(mapPoint->id_) == mapPoints_.end() )
    {
        mapPoints_.insert( make_pair(mapPoint->id_, mapPoint) );
    }
    else 
    {
        mapPoints_[mapPoint->id_] = mapPoint;
    }
}
void Map::mapPointCulling()
{
    list<MapPoint::Ptr>::iterator lit = recentAddedMapPoints_.begin();
    const unsigned long int nCurrentKFid = currKF_->id_;

    const int cnThObs = 3;

    while(lit!=recentAddedMapPoints_.end())
    {
        MapPoint::Ptr pMP = *lit;
        if(pMP->isBad())
        {
            lit = recentAddedMapPoints_.erase(lit);
        }
        else if(pMP->found_times_/pMP->visible_times_<0.25f )
        {
            pMP->good_ = false;
            lit = recentAddedMapPoints_.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->last_frame_seen_)>=10 && pMP->observations_.size()<=cnThObs)
        {
            pMP->good_ = false;
            lit = recentAddedMapPoints_.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->first_frame_seen_)>=12)
            lit = recentAddedMapPoints_.erase(lit);
        else
            lit++;
    }
}
void Map::searchInNeighbor()
{
    int nn=10;
    const vector<Frame::Ptr> vpNeighKFs = currKF_->getBestCovisibilityKeyFrames(nn);
    vector<Frame::Ptr> vpTargetKFs;
    for(vector<Frame::Ptr>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        Frame::Ptr pKFi = *vit;
        if(pKFi->isBad() || pKFi->fuseTargetFrameId_ == currKF_->id_)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->fuseTargetFrameId_= currKF_->id_;

        // Extend to some second neighbors
        const vector<Frame::Ptr> vpSecondNeighKFs = pKFi->getBestCovisibilityKeyFrames(5);
        for(vector<Frame::Ptr>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            Frame::Ptr pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->fuseTargetFrameId_==currKF_->id_ || pKFi2->id_==currKF_->id_)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->fuseTargetFrameId_= currKF_->id_;
        }
    }
    
    ORBmatcher matcher;
    vector<MapPoint::Ptr> vpMapPointMatches = currKF_->vpMapPoints_;
    for(auto vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; ++vit){
        Frame::Ptr pKFi = *vit;
        vector<size_t> fusedMPs = matcher.fuse(pKFi, vpMapPointMatches);
        for(auto idx : fusedMPs){//erase the replaced map points;
            mapPoints_.erase(idx);
        }
    }
    //TODO: Search matches by projection from target KFs in current KF
    vector<MapPoint::Ptr> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    for(auto vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF!=vendKF; ++vitKF){
        Frame::Ptr pKFi = (*vitKF);
        vector<MapPoint::Ptr> vpMPsKFi = pKFi -> getMapPointMatches();
        for(MapPoint::Ptr pMP : vpMPsKFi){
            if(pMP == nullptr)
                continue;
            if(pMP->isBad() || pMP->fuseCandidateForKF_ == currKF_->id_)
                continue;
            pMP->fuseCandidateForKF_ = currKF_->id_;
            vpFuseCandidates.push_back(pMP);
        }
    }
    vector<size_t> fusedMPs = matcher.fuse(currKF_, vpFuseCandidates);
    for(auto idx : fusedMPs){
        mapPoints_.erase(idx);
    }
    
    //update points
    vpMapPointMatches = currKF_->getMapPointMatches();
    for(MapPoint::Ptr pMP : vpMapPointMatches){
        if(pMP!=nullptr){
            if(!pMP->isBad()){
                pMP->computeDistinctiveDescriptors();
                pMP->updateNormalAndDepth();
            }
        }
    }
   
    //update connections
    currKF_->updateConnections();
}

void Map::localMapping()
{
    //0. process new key frame
    const vector<MapPoint::Ptr> vpMapPointMatches = currKF_->getMapPointMatches();
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint::Ptr pMP = vpMapPointMatches[i];
        if(pMP){
            if(!pMP->isBad()){
                if(!pMP->isInFrame(currKF_))
                {
                    pMP->addObservation(currKF_, i);
                    pMP->updateNormalAndDepth();
                    pMP->computeDistinctiveDescriptors();
                }
                else // this can only happen for new points inserted by addKeyFrame
                {
                    recentAddedMapPoints_.push_back(pMP);
                }
            }
        }
    }
    currKF_->updateConnections();
    //1. MapPoint Culling
    mapPointCulling();
    //2. search in neighbor to fuse
    searchInNeighbor();
    //3. optimize map
    if(keyframes_.size()>2){
        Optimizer::localBA(currKF_);
    }
    //4. keyframe culling
}

}
