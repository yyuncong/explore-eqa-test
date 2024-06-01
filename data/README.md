## HM-EQA dataset

[`questions.csv`](https://github.com/Stanford-ILIAD/explore-eqa/blob/master/data/questions.csv) includes the 500 questions based on 267 scenes from the [HM-3D](https://aihabitat.org/datasets/hm3d-semantics/) dataset. You can load a scene and the corresponding question in Habitat-Sim following the instruction [here](https://github.com/Stanford-ILIAD/explore-eqa/tree/master?tab=readme-ov-file#usage). We treat a pair of the scene from HM-3D and the floor as a single scene in our dataset and do not consider robot navigating to multiple floors in one scenario.

We also determine the initial poses of the robot in each scene by sampling valid poses (e.g., enough clearance from any obstacle), and they are available at [`scene_init_poses.csv`](https://github.com/Stanford-ILIAD/explore-eqa/blob/master/data/scene_init_poses.csv).