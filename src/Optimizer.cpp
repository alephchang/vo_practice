#include "gslam/Optimizer.h"
#include "gslam/g2o_types.h"
#include "gslam/Frame.h"
#include "gslam/Converter.h" 
#include"Eigen/Dense"
#include<fstream>

namespace gslam {
    Optimizer::Optimizer()
    {
    }


    Optimizer::~Optimizer()
    {
    }

    int Optimizer::optId = 0;
    std::string Optimizer::logPath;

void Optimizer::localBA(vector<unsigned long>& frame_ids, Map::Ptr map)
{
#if 0        
   vector<Frame::Ptr> frames(frame_ids.size());
   for (size_t i = 0; i < frames.size(); ++i) frames[i] = map->keyframes_[frame_ids[i]];
   typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver_6_3;
   typedef g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType> Linear;
   g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
       g2o::make_unique<BlockSolver_6_3>(g2o::make_unique<Linear>()));

   g2o::SparseOptimizer optimizer;
   optimizer.setAlgorithm(solver);
   //nodes of camera pose
   for(size_t i = 0; i < frames.size(); ++i){
       g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
       pose->setId(i);
       Eigen::Quaterniond se3_r(frames[i]->Tcw_.rotationMatrix());
       g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, frames[i]->Tcw_.translation());
       //Tcw.setRotation(se3_r);
       //Tcw.setTranslation(T_c_w_estimated_.translation());
       pose->setEstimate(Tcw);
       optimizer.addVertex(pose);
       if (i == 0)
           pose->setFixed(true);
   }

   //nodes of map point
   // parameter: camera intrinsics
   g2o::CameraParameters* camera = new g2o::CameraParameters(
       frames[0]->camera_->fx_, Eigen::Vector2d(frames[0]->camera_->cx_, frames[0]->camera_->cy_), 0);
   camera->setId(0);

   optimizer.addParameter(camera);
   int index = frames.size();
   int edge_id = 1;
   std::vector<MapPoint::Ptr> map_points;
   for(auto it = map->map_points_.begin(); it != map->map_points_.end(); ++it){
       int id = it->first;
       std::list<std::pair<int, cv::Point2f> > edge_candidate;
       for(size_t i = 0; i < frames.size(); ++i){
           auto it2d = frames[i]->map_points_2d_.find(id);
           if(it2d != frames[i]->map_points_2d_.end()){
               edge_candidate.push_back(std::pair<int, cv::Point2f>(i, it2d->second));
           }
       }
       ///TODO: complete the edge 
       if(edge_candidate.size()>1){//the map point is observed more than once
           //add the map point as node
           g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
           point->setId(index);
           point->setEstimate(it->second->pos_);
           point->setMarginalized(true);
           optimizer.addVertex(point);
           map_points.push_back(it->second);
           //add edges

           for(auto itedge = edge_candidate.begin(); itedge != edge_candidate.end(); ++itedge){
               g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
               e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(index)));
               e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(itedge->first)));
               e->setMeasurement(Eigen::Vector2d(itedge->second.x, itedge->second.y));
               g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
               e->setRobustKernel(rk);
               e->setParameterId(0, 0);
               e->setInformation(Eigen::Matrix2d::Identity());
               e->setId(edge_id++);
               optimizer.addEdge(e);
           }
           index++;
       }
   }
   std::ofstream fou;
   if (!logPath.empty()) {
       std::string path = logPath + "/opt_" + std::to_string(optId) + ".txt";
       fou.open(path.c_str(), std::ios::out);
   }

   if (fou.good()) {
       fou << "pose and points before optimization: " << std::endl;
       fou << " camera pose: " << std::endl;
       for (size_t i = 0; i < frames.size(); ++i) {
           const g2o::SE3Quat& Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
           fou << "camera " << i << " : " << Tcw.rotation().coeffs().transpose() << "\t" << Tcw.translation().transpose() << std::endl;
       }
/*            fou << " map points: " << std::endl;
       for (size_t i = frames.size(); i < index; ++i) {
           const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
           fou << "point " << i << " : " << vPoint.transpose() << std::endl;
       }*/
       auto edges = optimizer.edges();
       for (auto it = edges.begin(); it != edges.end(); ++it) {
           dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->computeError();
           if (dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().norm() < 4.0) continue;
           fou << "point id: " << (*it)->vertex(0)->id() << " camera id: " << (*it)->vertex(1)->id()
               << " pixel locatoin: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->measurement().transpose() 
               << " error: "<< dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().transpose()<< std::endl;
       }
   }

   optimizer.setVerbose(true);
   optimizer.initializeOptimization();
   optimizer.optimize(10);

   if (fou.good()) {
       fou << "pose and points after optimization: " << std::endl;
       fou << " camera pose: " << std::endl;
       for (size_t i = 0; i < frames.size(); ++i) {
           const g2o::SE3Quat& Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
           fou << "camera " << i << " : " << Tcw.rotation().coeffs().transpose() << "\t" << Tcw.translation().transpose() << std::endl;
       }
/*            fou << " map points: " << std::endl;
        for (size_t i = frames.size(); i < index; ++i) {
            const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
            fou << "point " << i << " : " << vPoint.transpose() << std::endl;
        }*/
        auto edges = optimizer.edges();
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            if (dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().norm() < 4.0) continue;
            fou << "point id: " << (*it)->vertex(0)->id() << " camera id: " << (*it)->vertex(1)->id()
                << " pixel locatoin: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->measurement().transpose()
                << " error: " << dynamic_cast<g2o::EdgeProjectXYZ2UV*>(*it)->error().transpose() << std::endl;
        }
    }
    fou.close();
    optId++;
    for (size_t i = 0; i < frames.size(); ++i) {
        g2o::SE3Quat Tcw = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i))->estimate();
        frames[i]->Tcw_ = Sophus::SE3d(Tcw.rotation(), Tcw.translation());
    }
    for (size_t i = frames.size(); i < index; ++i) {
        const Eigen::Vector3d& vPoint = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i))->estimate();
        map_points[i - frames.size()]->pos_ = vPoint;
    }
#endif        
}

void Optimizer::localBA(Frame::Ptr pKF)
{
    //TODO: 
    //1. 1-order neighbor key frames (i.e. local key frames)
    list<Frame::Ptr> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->baId_ = pKF->id_;

    const vector<Frame::Ptr> vNeighKFs = pKF->orderedConnectedKeyFrames_;
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        Frame::Ptr pKFi = vNeighKFs[i];
        pKFi->baId_ = pKF->id_;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }
    //2. local map points seen by pKF
    list<MapPoint::Ptr> lLocalMapPoints;
    for(list<Frame::Ptr>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint::Ptr> vpMPs = (*lit)->getMapPointMatches();
        for(vector<MapPoint::Ptr>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint::Ptr pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->baId_!=pKF->id_)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->baId_=pKF->id_;
                    }
        }
    }
    //3. key frames see local map points but not local key frames (i.e. fixed key frames)
    list<Frame::Ptr> lFixedCameras;
    for(list<MapPoint::Ptr>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<Frame::Ptr,size_t> observations = (*lit)->observations_;
        for(map<Frame::Ptr,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            Frame::Ptr pKFi = mit->first;

            if(pKFi->baId_!=pKF->id_ && pKFi->baFixedId_!=pKF->id_)
            {                
                pKFi->baFixedId_=pKF->id_;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    //4. setup optimizer
    g2o::SparseOptimizer optimizer;
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver_6_3;
    typedef g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType> Linear;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver_6_3>(g2o::make_unique<Linear>()));
    optimizer.setAlgorithm(solver);
    size_t maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<Frame::Ptr>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        Frame::Ptr pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Eigen::Quaterniond se3_r(pKFi->Tcw_.rotationMatrix());
        g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, pKFi->Tcw_.translation());
        vSE3->setEstimate(Tcw);
        vSE3->setId(pKFi->id_);
        vSE3->setFixed(pKFi->id_==0);
        optimizer.addVertex(vSE3);
        if(pKFi->id_>maxKFid)
            maxKFid=pKFi->id_;
    }

    // Set Fixed KeyFrame vertices
    for(list<Frame::Ptr>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        Frame::Ptr pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Eigen::Quaterniond se3_r(pKFi->Tcw_.rotationMatrix());
        g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, pKFi->Tcw_.translation());
        vSE3->setEstimate(Tcw);
        vSE3->setId(pKFi->id_);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->id_>maxKFid)
            maxKFid=pKFi->id_;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<Frame::Ptr> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint::Ptr> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    
    for(list<MapPoint::Ptr>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint::Ptr pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->pos_);
        int id = pMP->id_+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<Frame::Ptr,size_t> observations = pMP->observations_;
        //add edge
        for(map<Frame::Ptr,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            Frame::Ptr pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->vKeys_[mit->second];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->id_)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKFi->vInvLevelSigma2_[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                e->fx = pKFi->camera_->fx_;
                e->fy = pKFi->camera_->fy_;
                e->cx = pKFi->camera_->cx_;
                e->cy = pKFi->camera_->cy_;

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
            }
        }
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(5);
    optimizer.setVerbose(true);
    
    //TODO:
        // Check inlier observations
    if(0){//do more?
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint::Ptr pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive()){
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }
    
    vector<pair<Frame::Ptr,MapPoint::Ptr> > vToErase;
    vToErase.reserve(vpEdgesMono.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint::Ptr pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive()){
            Frame::Ptr pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    for(size_t i=0;i<vToErase.size();i++){
        Frame::Ptr pKFi = vToErase[i].first;
        MapPoint::Ptr pMPi = vToErase[i].second;
        int idx = pMPi->getIndexInFrame(pKFi);
        if(idx>=0) pKFi->vpMapPoints_[idx] = nullptr;
        pMPi->eraseObservation(pKFi);
    }
    
    // Recover optimized data

    //Keyframes
    for(list<Frame::Ptr>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        Frame::Ptr pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->id_));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->Tcw_ = Sophus::SE3d(SE3quat.rotation(), SE3quat.translation());
    }

    //Points
    for(list<MapPoint::Ptr>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint::Ptr pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id_+maxKFid+1));
        pMP->pos_ = vPoint->estimate();
        pMP->updateNormalAndDepth();
    }    
}

    
int Optimizer::poseOptimization(Frame::Ptr pFrame)
{
    g2o::SparseOptimizer optimizer;
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolver_6_3;
    typedef g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType> Linear;
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolver_6_3>(g2o::make_unique<Linear>()));
//    BlockSolver_6_3::LinearSolverType * linearSolver;
//    linearSolver = new g2o::LinearSolverDense<BlockSolver_6_3::PoseMatrixType>();
//    BlockSolver_6_3 * solver_ptr = new BlockSolver_6_3(linearSolver);
//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    //vSE3->setEstimate(Converter::toSE3Quat(pFrame->Tcw_));
    Eigen::Quaterniond se3_r(pFrame->Tcw_.rotationMatrix());
    g2o::SE3Quat Tcw = g2o::SE3Quat(se3_r, pFrame->Tcw_.translation());
    vSE3->setEstimate(Tcw);
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N_;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    const float deltaMono = sqrt(5.991);

    for(int i=0; i<N; i++)
    {
        MapPoint::Ptr pMP = pFrame->vpMapPoints_[i];
        if(pMP!=nullptr)
        {
            nInitialCorrespondences++;
            pFrame->vbOutlier_[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pFrame->vKeys_[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pFrame->vInvLevelSigma2_[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pFrame->camera_->fx_;
            e->fy = pFrame->camera_->fy_;
            e->cx = pFrame->camera_->cx_;
            e->cy = pFrame->camera_->cy_;
            cv::Point3f Xw = pMP->getPositionCV();
            e->Xw[0] = Xw.x;
            e->Xw[1] = Xw.y;
            e->Xw[2] = Xw.z;

            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vnIndexEdgeMono.push_back(i);
        }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->Tcw_));
        optimizer.initializeOptimization(0);
//        optimizer.setVerbose(true);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->vbOutlier_[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->vbOutlier_[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->vbOutlier_[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    Tcw = vSE3_recov->estimate();
    pFrame->Tcw_ = Sophus::SE3d(Tcw.rotation(), Tcw.translation());
#ifdef VO_DEBUG    
    std::ofstream fou;
    logPath = "/work/data/fr1xyz";
    if (!logPath.empty()) {
        std::string path = logPath + "/opt_" + std::to_string(optId) + ".txt";
        fou.open(path.c_str(), std::ios::out);
    }

    if(fou.good()){
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        fou << "vSE3: " << vSE3_recov->estimate() <<endl;
        auto edges = optimizer.edges();
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            g2o::EdgeSE3ProjectXYZOnlyPose* itc = dynamic_cast<g2o::EdgeSE3ProjectXYZOnlyPose*>(*it);
            itc->computeError();
            Eigen::Matrix3d K;
            K << itc->fx, 0.0, itc->cx, 0.0, itc->fy, itc->cy, 0.0, 0.0, 1.0;
            Eigen::Vector3d pt3(itc->Xw[0], itc->Xw[1], itc->Xw[2]);
            pt3 = K * vSE3_recov->estimate()* pt3;
            Eigen::Vector2d projerr(itc->measurement()[0]-pt3[0]/pt3[2], itc->measurement()[1]-pt3[1]/pt3[2]); 
            fou << itc->measurement().transpose() << " -> "
                << itc->Xw.transpose() << " edge error: "
                << itc->error().transpose() << " proj error: " << projerr.transpose() << endl;
            
        }
        fou.close();
    }
#endif    
    return nInitialCorrespondences-nBad;
}
}
