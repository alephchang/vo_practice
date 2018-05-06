
add_executable( run_vo main.cpp run_vo.cpp )
target_link_libraries( run_vo 
        gslam 
        /usr/local/lib/libprofiler.so)
add_executable( run_orb test_orb.cpp)
target_link_libraries( run_orb gslam )
target_link_libraries( run_orb /usr/local/lib/libopencv_xfeatures2d.so)
add_executable( test_stereomatch test_stereomatch.cpp)
target_link_libraries( test_stereomatch gslam )
add_executable(reformat_kitti_poses reformat_kitti_poses.cpp)