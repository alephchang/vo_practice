#include<fstream>
#include<iostream>
#include<Eigen/Core>
#include<Eigen/Geometry>

int main(int argc, char** argv)
{
    if(argc != 3){
        std::cout << "usage: convertKittiPoses kitti_times_file kitti_poses_file" << std::endl;
        return 1;
    }
    std::ifstream fitime(argv[1]);
    std::ifstream fi(argv[2]);
    if(!fi.good()){
        std::cout << "fail to open file: "<< argv[1] <<std::endl;
        return 0;
    }
    std::string outpath = std::string(argv[2]) + ".out";
    std::ofstream fo(outpath.c_str());
    fo.precision(15);
    double matData[12];
    int j = 0;
    while(!fi.eof()){
        double timestamp = 0.0;
        fitime >> timestamp;
        for(size_t i = 0; i < 12; ++i)
          fi >> matData[i];
        if(fi.eof()) break;
        Eigen::Map< Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > transMat(matData);
        Eigen::Quaterniond Q(transMat.block<3,3>(0,0));
        Eigen::Vector3d t(transMat.block<3,1>(0,3));
        Q = Q.inverse();
        t = Q*(-t);
        fo << timestamp << " "
           << t[0] << " "<< t[1] << " " << t[2] << " " 
           << Q.x() << " " << Q.y() << " "<<Q.z() << " " << Q.w() << std::endl;
         if(j==1)
             std::cout << transMat << std::endl;
         ++j;
    }
    fitime.close();
    fi.close();
    fo.close();
    return 0;
}