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
#include "gslam/common_include.h"
#include "gslam/MapPoint.h"
#include "gslam/ORBmatcher.h"
namespace gslam
{

MapPoint::MapPoint()
: id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), good_(true), visible_times_(0), found_times_(0)
{

}

MapPoint::MapPoint ( long unsigned int id, const Vector3d& position, const Vector3d& norm, std::shared_ptr<Frame> frame, const Mat& descriptor )
: id_(id), pos_(position), norm_(norm), good_(true), visible_times_(1), found_times_(1), descriptor_(descriptor),fuseCandidateForKF_(0)
{
    last_frame_seen_ = frame->id_;
    first_frame_seen_ = frame->id_; 
    track_ref_frame_ = 0;
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, Vector3d(0,0,0), Vector3d(0,0,0) )
    );
}

MapPoint::Ptr MapPoint::createMapPoint ( 
    const Vector3d& pos_world, 
    const Vector3d& norm, 
    const Mat& descriptor, 
    std::shared_ptr<Frame> frame )
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, pos_world, norm, frame, descriptor )
    );
}

unsigned long MapPoint::factory_id_ = 0;

void MapPoint::computeDistinctiveDescriptors()
{
    vector<cv::Mat> vDescriptors;
    map<Frame::Ptr,size_t> observations = observations_;
    if(observations.empty()) return;
    vDescriptors.reserve(observations.size());
    for(map<Frame::Ptr, size_t>::iterator mit = observations.begin(); mit != observations.end(); ++mit){
        Frame::Ptr pKF = mit->first;
        if(!pKF->isBad()){
            vDescriptors.push_back(pKF->descriptors_.row(mit->second));
        }
    }
    if(vDescriptors.empty()) return;
    const size_t N = vDescriptors.size();
    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }
    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }
    descriptor_ = vDescriptors[BestIdx].clone();
}
void MapPoint::addObservation(shared_ptr< Frame > frame, size_t i)
{
    if(observations_.count(frame)) return;
    observations_.insert(make_pair(frame, i));
}

void MapPoint::updateNormalAndDepth()
{
    Vector3d normal(0,0,0);
    int n = 0;
    for(auto mit = observations_.begin(), mend = observations_.end(); mit != mend; ++mit){
        Vector3d OWi = mit->first->getCamCenter();
        Vector3d normali = pos_ - OWi;
        normal = normal + normali / normali.norm();
        n++;
    }
    norm_ = normal / n;
    norm_.normalize();
}
bool MapPoint::isInFrame(shared_ptr< Frame > frame)
{
    return observations_.count(frame);
}

bool MapPoint::replace(MapPoint::Ptr pMP)
{
    if(id_ == pMP->id_)
        return false;
    for(auto mit = observations_.begin(), mend = observations_.end(); mit != mend; ++mit){
        Frame::Ptr pKF = mit->first;
        if(!pMP->isInFrame(pKF)){
            pKF->addMapPoint(pMP, mit->second);
            pMP->addObservation(pKF, mit->second);
        }
        else{
            pKF->vpMapPoints_[mit->second] = nullptr;
        }
    }
    pMP->visible_times_ +=visible_times_;
    pMP->found_times_ += found_times_;
    pMP->computeDistinctiveDescriptors();
    return true;
}

int MapPoint::getIndexInFrame(shared_ptr< Frame > pKF)
{
    if(observations_.count(pKF))
        return observations_[pKF];
    else
        return -1;
}

void MapPoint::eraseObservation(shared_ptr< Frame > pKF)
{
    if(observations_.count(pKF))
    {
        int idx = observations_[pKF];

        observations_.erase(pKF);

        if(observations_.size()<=2)
            good_=false;
    }
}


}
