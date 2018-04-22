#include<iostream>
#include"Eigen/Dense"
//#include <eigen3/Eigen/src/Core/Matrix.h>

int main()
{
    Eigen::Matrix3f K;// = Eigen::Matrix3f::Identity();
    K << 517.3, 0.0, 325.1, 0.0, 516.6, 249.7, 0.0, 0.0, 1.0; 
    Eigen::Vector3f pt3;
    pt3 << 0.131611, 0.356976 ,   0.974;
    pt3 << -0.123274, 0.186114, 1.0836;
    pt3 = K * pt3;
    //std::cout << pt3[0] << " " << pt3[1] << " "<<pt3[2] << std::endl;
    Eigen::Vector2f pt2proj(pt3[0]/pt3[2], pt3[1]/pt3[2]);
    Eigen::Vector2f pt2;
    pt2 << 404, 430;
    pt2 <<273.715, 328.458;
    Eigen::Vector2f err = pt2 - pt2proj;
    std::cout << err.transpose() << std::endl;
    return 0;
}