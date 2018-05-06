#include"run_vo.h"
#include<stdio.h>
#include "gslam/g2o_types.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include<boost/timer.hpp>
#include<fstream>
#include"profiler.h"
// -------------- test the visual odometry -------------

int run_vo( int argc, char** argv )
{
    if ( argc != 2 )
    {
        for (int i = 0; i < argc; ++i)
        cout << argv[i] << endl;
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    gslam::Config::setParameterFile ( argv[1] );

    string dataset_dir = gslam::Config::get<string>("dataset_dir");
    string vo_type = gslam::Config::get<string>("vo_type");
    if(vo_type == "stereo")
        gslam::VisualOdometry::voType_ = gslam::VO_STEREO;
    else if(vo_type == "rgbd" )
        gslam::VisualOdometry::voType_ = gslam::VO_RGBD;
    else
        gslam::VisualOdometry::voType_ = gslam::VO_UNKNOW;
    
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    
    gslam::VisualOdometry::Ptr vo(new gslam::VisualOdometry);
    if(vo->setLogFile(dataset_dir + "/log.txt")==false){
        cout << "Faile to create the log file: " << dataset_dir + "log.txt" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files, right_files;
    vector<double> rgb_times, depth_times;
    if(gslam::VisualOdometry::voType_== gslam::VO_STEREO){
        while( !fin.eof()){
            string rgb_time, rgb_file, right_file;
            fin>>rgb_time >> rgb_file >> right_file;
            if(rgb_file.empty()) break;
            rgb_times.push_back(atof(rgb_time.c_str()));
            rgb_files.push_back(dataset_dir+"/"+rgb_file);
            right_files.push_back(dataset_dir+"/"+right_file);
            if(fin.good()==false) break;
        }
    }
    else if(gslam::VisualOdometry::voType_== gslam::VO_RGBD){
        while ( !fin.eof() ) {
            string rgb_time, rgb_file, depth_time, depth_file;
            fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
            if(rgb_file.empty()) break;
            rgb_times.push_back ( atof ( rgb_time.c_str() ) );
            depth_times.push_back ( atof ( depth_time.c_str() ) );
            rgb_files.push_back ( dataset_dir+"/"+rgb_file );
            depth_files.push_back ( dataset_dir+"/"+depth_file );
            cout << "rgb_file: "<<rgb_file <<endl;
            if ( fin.good() == false )
                break;
        }
    }
    else{
        cout << "Unknow vo type: " << vo_type << endl;
        return 1;
    }
    //Load ORB Vocabulary
    string orbVocabDir= gslam::Config::get<string>("orb_vocab_dir");
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    gslam::Frame::pORBvocab_ = std::make_shared<gslam::ORBVocabulary>();
    bool bVocLoad = gslam::Frame::pORBvocab_->loadFromTextFile(orbVocabDir);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << orbVocabDir << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;
    gslam::Camera::Ptr camera ( new gslam::Camera );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    vector<SE3<double> > estimated_pose;
    Mat gray;
    //std::string program=std::string(argv[0])+ "_" + std::to_string(getpid())+".prof";
    //ProfilerStart(program.c_str());
    boost::timer timer;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        if ( color.data==nullptr )
            break;
        cvtColor(color, gray, CV_RGB2GRAY);
        gslam::Frame::Ptr pFrame = gslam::Frame::createFrame();
        pFrame->id_ = i;
        pFrame->camera_ = camera;
        pFrame->imLeft_ = gray;
        pFrame->timeStamp_ = rgb_times[i];
        if(gslam::VisualOdometry::voType_ == gslam::VO_RGBD){
            Mat depth = cv::imread ( depth_files[i], -1 );
            if( depth.data==nullptr ) break;
            pFrame->imDepth_ = depth;
        }
        else{
            Mat right = cv::imread(right_files[i], -1);
            if(right.data==nullptr) break;
            pFrame->imRight_ = right;
        }

        vo->addFrame ( pFrame );

        if ( vo->state_ == gslam::VisualOdometry::LOST ){
            cout << "VO lost!" << endl;
            break;
        }
        estimated_pose.push_back(pFrame->Tcw_);
        // show the map and the camera pose
        Mat img_show = color.clone();
        for ( auto& pt:vo->map_->mapPoints_ )
        {
            gslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel ( p->pos_, pFrame->Tcw_ );
            cv::circle ( img_show, cv::Point2f ( pixel ( 0,0 ),pixel ( 1,0 ) ), 5, cv::Scalar ( 0,255,0 ), 2 );
        }

        cv::imshow ( "image", img_show );
        cv::waitKey ( 1 );
        cout<<endl;
    }
     cout<<"VO costs time per frame: "<<timer.elapsed()/rgb_files.size() <<endl;
    //ProfilerStop();
    ofstream fo(dataset_dir + "/estimatedpose.txt");
    fo.precision(15);
    for (size_t i = 0; i < estimated_pose.size(); ++i) {
        const SE3<double>& se3(estimated_pose[i]);
        fo << rgb_times[i] << "\t" << se3.translation().x()<<" "
            << se3.translation().y() << " "
            << se3.translation().z() << " "
            << se3.unit_quaternion().x()<< " " 
            << se3.unit_quaternion().y() << " " 
            << se3.unit_quaternion().z() << " " 
            << se3.unit_quaternion().w() << endl;
    }
    fo.close();
    return 0;
}
void testSE3QuatError()
{
	Eigen::Quaterniond se3_r;
	g2o::SE3Quat Tcw;
	Tcw.setRotation(se3_r);
}