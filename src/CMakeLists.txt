add_library( gslam SHARED
    Frame.cpp
    MapPoint.cpp
    Map.cpp
    Camera.cpp
    Config.cpp
    g2o_types.cpp
    VisualOdometry.cpp
    Optimizer.cpp
    ORBextractor.cc
    ORBmatcher.cpp
    Converter.cpp
)

target_link_libraries( gslam
    ${THIRD_PARTY_LIBS}
)
