# vo_practice
visual odometry practice
The frame is from gaoxiang's slam book. Many techniques in ORB-SLAM2 are transplanted here, like ORB features and matching, local mapping and 4 steps optimization, covisibility graph. 

# Flow
``` js
state=initilize  
for a new frame:
  detect features and compute features description
  if state==initilize:
    compute the depth by stereo and add the points to map.mappoints
    state = OK
  else:
    match features with previous frame
    estimate the pose by PnP
    find more matches by projection
    estimate the pose by PnP
    if pose estimation is accepted:
      if need key frame:
        insert new points to map.mpapoints
        insert current frame to map.keyframes
        update connection(i.e. covisibility)
        fuse map.mappoints,(i.e. merge close mappoints)
        local bundle adjustment for current keyframe
    else://tracking fails, re-init
      estimate the pose by motion
      add the points to map.mappoints
```

# Result
We use several sequences from [KITTI dataset] to show the result.

The evaluation tool is from [TUM tools].

The following table shows the ATE and REP result on three sequence,including 03, 04 and 07.

| sequence id  | 03    | 04    | 07    |
|--------------|-------|-------|-------|
| duration (s) | 82.72 | 28.11 |114.33 |
| rmse (m)     | 1.47  | 0.46  | 3.43  |
| mean (m)     | 1.35  | 0.43  | 3.11  |
| midian (m)   | 1.26  | 0.38  | 2.91  |
| std (m)      | 0.58  | 0.16  | 1.45  |
| min (m)      | 0.38  | 0.14  | 0.58  |
| max (m)      | 4.40 | 1.22  | 6.04   |
| RPE          | 0.57  |0.31   | 0.54  |

The following figure shows the trajectory comparison of sequence 07.
![07_ate_r](https://github.com/alephchang/vo_practice/blob/master/evaluate/result/07_ate_r.png)



[gaoxiang's slambook]: https://github.com/gaoxiang12/slambook
[ORB-SLAM2]:https://github.com/raulmur/ORB_SLAM2
[KITTI dataset]:http://www.cvlibs.net/datasets/kitti/eval_odometry.php
[TUM tools]: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
