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
#include "gslam/Frame.h"
#include <boost/concept_check.hpp>

namespace gslam
{
shared_ptr<ORBVocabulary>  Frame::pORBvocab_=nullptr;
Frame::Frame()
: id_(-1), timeStamp_(-1), camera_(nullptr)
{

}

Frame::Frame ( long id, double time_stamp, SE3<double> T_c_w, Camera::Ptr camera, Mat color, Mat depth )
: id_(id), timeStamp_(time_stamp), Tcw_(T_c_w), camera_(camera), color_(color), depth_(depth) 
{

}

Frame::~Frame()
{

}

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;
    return Frame::Ptr( new Frame(factory_id++) );
}

double Frame::findDepth ( const cv::KeyPoint& kp )
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return double(d)/camera_->depth_scale_;
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return double(d)/camera_->depth_scale_;
            }
        }
    }
    return -1.0;
}

void Frame::setPose ( const SE3<double>& T_c_w )
{
    Tcw_ = T_c_w;
}


Vector3d Frame::getCamCenter() const
{
    return Tcw_.inverse().translation();
}

bool Frame::isInFrame ( const Vector3d& pt_world )
{
    Vector3d p_cam = camera_->world2camera( pt_world, Tcw_ );
    // cout<<"P_cam = "<<p_cam.transpose()<<endl;
    if ( p_cam(2,0)<0 ) return false;
    Vector2d pixel = camera_->world2pixel( pt_world, Tcw_ );
    return isInFrame(pixel);
}

bool Frame::isInFrame(const Vector2d& pixel)
{
    return pixel(0,0)>0 && pixel(1,0)>0 
        && pixel(0,0)<color_.cols 
        && pixel(1,0)<color_.rows;
}

bool Frame::isInFrustum(MapPoint::Ptr pMp)
{
    pMp->track_in_view_ = false;
    Vector3d P = pMp->pos_;
    Vector3d Pc = Tcw_*P;
    const double &PcX = Pc(0);
    const double &PcY= Pc(1);
    const double &PcZ = Pc(2);
    if(PcZ<0.0) return false;
    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=camera_->fx_*PcX*invz+camera_->cx_;
    const float v=camera_->fy_*PcY*invz+camera_->cy_;
    if(u<0.0 || u > static_cast<double>(color_.cols)) return false;
    if(v<0.0 || v > static_cast<double>(color_.rows)) return false;
    
    Vector3d Pn = pMp->norm_;
    assert(std::abs<double>(Pn.norm()-1.0)<1e-10);
    Vector3d Ow = -Tcw_.rotationMatrix().transpose()*Tcw_.translation();
    Vector3d PO = P - Ow;
    const double viewCos = PO.dot(Pn)/PO.norm();
    if(viewCos < 0.5) return false;
    pMp->track_in_view_ = true;
    pMp->track_proj_x_ = u;
    pMp->track_proj_y_ = v;
    pMp->track_view_cos_= viewCos;  
    return true;
}

vector<size_t> Frame::getFeaturesInAera(float x, float y, float r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N_);
    for(size_t i = 0; i < N_; ++i){
        const cv::KeyPoint &kp = vKeys_[i];
        float dx = kp.pt.x - x;
        float dy = kp.pt.y - y;
        if(fabs(dx)<r && fabs(dy) < r)
            vIndices.push_back(i);
    }
    return vIndices;
}




void Frame::addMapPoint(MapPoint::Ptr pMp, size_t i)
{
    vpMapPoints_[i] = pMp;
}
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

void Frame::computeBoW()
{
    if(BowVec_.empty())
    {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(descriptors_);
        pORBvocab_->transform(vCurrentDesc,BowVec_,featVec_,4);
    }
}

vector< MapPoint::Ptr > Frame::getMapPointMatches()
{
    return vpMapPoints_;
}

void Frame::updateConnections()
{
    map<Frame::Ptr,int> KFcounter;
    for(vector<MapPoint::Ptr>::iterator vit=vpMapPoints_.begin(), vend=vpMapPoints_.end(); vit!=vend; vit++){
        MapPoint::Ptr pMP = *vit;
        if(!pMP)
            continue;
        if(pMP->isBad())
            continue;
        map<Frame::Ptr,size_t> observations = pMP->observations_;

        for(map<Frame::Ptr,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->id_==id_)
                continue;
            KFcounter[mit->first]++;
        }
    }    
    // This should not happen
    if(KFcounter.empty())
        return;
    
    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    Frame::Ptr pKFmax=nullptr;
    int th = 15;

    vector<pair<int,Frame::Ptr> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<Frame::Ptr,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++){
        if(mit->second>nmax){
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th){
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->addConnection(shared_from_this(),mit->second);
        }
    }
    if(vPairs.empty()){
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->addConnection(shared_from_this(), nmax);
    }
    std::sort(vPairs.begin(), vPairs.end());
    list<Frame::Ptr> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++){
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    connectedKeyFrameWeights_ = KFcounter;
    orderedConnectedKeyFrames_ = vector<Frame::Ptr>(lKFs.begin(),lKFs.end());
    orderedWeights_ = vector<int>(lWs.begin(), lWs.end());    
}

void Frame::addConnection(Frame::Ptr frame, const int& weight)
{
    if(connectedKeyFrameWeights_.count(frame))
        connectedKeyFrameWeights_[frame] = weight;
    else if(connectedKeyFrameWeights_[frame]!=weight)
        connectedKeyFrameWeights_[frame] = weight;
    else
        return;
    updateBestCovisibles();
}
vector< Frame::Ptr > Frame::getBestCovisibilityKeyFrames(int N)
{
    if((int)orderedConnectedKeyFrames_.size()<N)
        return orderedConnectedKeyFrames_;
    else
        return vector<Frame::Ptr>(orderedConnectedKeyFrames_.begin(),orderedConnectedKeyFrames_.begin()+N);
}

void Frame::setBadFlag()
{
    if(id_==0) return;
    for(map<Frame::Ptr, int>::iterator mit = connectedKeyFrameWeights_.begin(), mend = connectedKeyFrameWeights_.end();
                            mit != mend; ++mit){
        mit->first->eraseConnection(shared_from_this());
    }
    for(size_t i = 0; i < vpMapPoints_.size(); ++i){
        vpMapPoints_[i]->eraseObservation(shared_from_this());
    }
    connectedKeyFrameWeights_.clear();
    orderedConnectedKeyFrames_.clear();
}

void Frame::eraseConnection(Frame::Ptr frame)
{
    if(connectedKeyFrameWeights_.count(frame) )
        connectedKeyFrameWeights_.erase(frame);
}

void Frame::updateBestCovisibles()
{
    vector<pair<int,Frame::Ptr> > vPairs;
    vPairs.reserve(connectedKeyFrameWeights_.size());
    for(map<Frame::Ptr,int>::iterator mit=connectedKeyFrameWeights_.begin(), mend=connectedKeyFrameWeights_.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    std::sort(vPairs.begin(),vPairs.end());
    list<Frame::Ptr> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++){
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    orderedConnectedKeyFrames_ = vector<Frame::Ptr>(lKFs.begin(),lKFs.end());
    orderedWeights_ = vector<int>(lWs.begin(), lWs.end());        
}

}//namespace
