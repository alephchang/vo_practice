#include"run_vo.h"


int main(int argc, char** argv)
{
    run_vo(argc, argv);
    //testSE3QuatError();
    //validate_result(argc, argv);
}
/*
TODO list:
1. add map_points_ in each frame, accelerate the points match between key frames
2. triangulatepoints for the matchec points for key points
*/