
#include"gslam/ORBmatcher.h"
#include <boost/concept_check.hpp>

namespace gslam {
    
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}
// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
int ORBmatcher::searchByBoW(Frame::Ptr pKF,Frame::Ptr F, vector<MapPoint::Ptr> &vpMapPointMatches)
{
    F->computeBoW();
    //
    const vector<MapPoint::Ptr>& vpMapPointsKF = pKF->vpMapPoints_;

    vpMapPointMatches.assign(F->vKeys_.size(),static_cast<MapPoint::Ptr>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->featVec_;

    int nmatches=0;
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    
    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F->featVec_.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F->featVec_.end();
    
    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint::Ptr pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->descriptors_.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F->descriptors_.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F->featVec_.lower_bound(KFit->first);
        }
    }
    return nmatches;
}   

int ORBmatcher::searchByProjection(Frame::Ptr F, const vector< MapPoint::Ptr >& vpMapPoints, const float r)
{
    int nmatches=0;
    for(size_t i=0; i<vpMapPoints.size(); i++){
        MapPoint::Ptr pMp = vpMapPoints[i];
        if(pMp->isBad()) continue;
        const vector<size_t> vIndices = F->getFeaturesInAera(pMp->track_proj_x_, pMp->track_proj_y_, r);
        if(vIndices.empty()) continue;
        const cv::Mat MPdescriptor = pMp->getDescriptor();
        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++){
            const size_t idx = *vit;
            ///TODO:
            // how to add mappoint to the map??? Ans: update local map will add mappoint
            if(F->vpMapPoints_[idx])
                if(F->vpMapPoints_[idx]->observations_.empty()==false)
                    continue;
            const cv::Mat &d = F->descriptors_.row(idx);
            const int dist = DescriptorDistance(MPdescriptor,d);
            if(dist<bestDist){
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F->vKeys_[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2){
                bestLevel2 = F->vKeys_[idx].octave;
                bestDist2=dist;
            }
        }
        if(bestDist<=TH_HIGH){
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F->vpMapPoints_[bestIdx]=pMp;
            nmatches++;
        }
    }
    return nmatches;
}

vector<size_t> ORBmatcher::fuse(Frame::Ptr pKF, const std::vector< MapPoint::Ptr >& vpMapPoints, const float th)
{
    vector<size_t> fusedMPs;
    SE3<double> Tcw = pKF->Tcw_;
    const int nMPs = vpMapPoints.size();
    for(int i = 0; i < nMPs; ++i){
        MapPoint::Ptr pMP = vpMapPoints[i];
        if(pMP==nullptr) 
            continue;
        if(pMP->isBad() || pMP->isInFrame(pKF))
            continue;
        Vector3d p3Dw = pMP->pos_;
        Vector3d p3Dc = pKF->camera_->world2camera(p3Dw, Tcw);
        if(p3Dc(2) < 0.0)
            continue;
        Vector2d p2Dc = pKF->camera_->camera2pixel(p3Dc);
        if(!pKF->isInFrame(p2Dc))
            continue;
        Vector3d Ow = pKF->getCamCenter();
        Vector3d PO = p3Dw - Ow;
        Vector3d Pn = pMP->norm_;
        double dist3D = PO.norm();
        if(PO.dot(Pn) < 0.5*PO.norm())
            continue;
        float radius = th * 8;
        vector<size_t> vIndices = pKF->getFeaturesInAera(p2Dc(0), p2Dc(1), radius);
        if(vIndices.empty())
            continue;
        const cv::Mat dMP = pMP->getDescriptor();
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++){
            const size_t idx = *vit;
            const cv::KeyPoint &kp = pKF->vKeys_[idx];
            const cv::Mat &dKF  = pKF->descriptors_.row(idx);
            const int dist = DescriptorDistance(dMP, dKF);
            if(dist < bestDist){
                bestDist = dist;
                bestIdx = idx;
            }
        }
        
        if(bestDist < TH_LOW){
            MapPoint::Ptr pMPinKF = pKF->vpMapPoints_[bestIdx];
            if(pMPinKF==nullptr){
                pMP->addObservation(pKF, bestIdx);
                pKF->addMapPoint(pMP, bestIdx);
            }
            else{
                if(!pMPinKF->isBad()){
                    if(pMPinKF->observations_.size() > pMP->observations_.size())
                        if(pMP->replace(pMPinKF))
                            fusedMPs.push_back(pMP->id_);
                    else
                        if(pMPinKF->replace(pMP))
                            fusedMPs.push_back(pMPinKF->id_);
                }
            }
        }
    }
    return fusedMPs;
}

    
}










