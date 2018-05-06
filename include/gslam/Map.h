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

#ifndef MAP_H
#define MAP_H

#include "gslam/common_include.h"
#include "gslam/Frame.h"
#include "gslam/MapPoint.h"

namespace gslam
{
class Map
{
public:
    typedef shared_ptr<Map> Ptr;
    unordered_map<size_t, MapPoint::Ptr >  mapPoints_;        // all landmarks
    unordered_map<size_t, Frame::Ptr >     keyframes_;         // all key-frames
    Frame::Ptr                          currKF_;
    std::list<MapPoint::Ptr>    recentAddedMapPoints_;

    Map() {}
    
    void insertKeyFrame( Frame::Ptr frame );
    void insertMapPoint( MapPoint::Ptr mapPoint );
    void localMapping();
    void mapPointCulling();
    void searchInNeighbor();
    void keyFrameCulling();
};
}

#endif // MAP_H