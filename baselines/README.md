# vi_rl

The code is still under development and there is a lot of thing needed to done.

The code is in baselines/mbsr.

**Requirement**
- ```pip install -e .```
- Newest version of Gym and mujoco_py, mujoco150
- MPI.

To simple test, please run 
```shell
cs baselines/mbsr
python experiment/train.py
```

There are a number of things to need to improve and test.

**Code part**:
- [ ] Remove unecessary her code in mbsr;
- [ ] Add reward function for all environments: need to change gym code;
- [x] Add necessary configuration params for testing and tunning parameters;
- [ ] Move Backtrace trajectory rollouts code to RolloutWorker;
- [x] Add save policy part code in __setstate__ and __getstate__;
- [ ] Change experiment/play.py and plot.py code for future testing and video generation.
- [x] Remove testing code (fake final states and goals) for backtrace trajectory generation;
- [ ] Add StageArea for better training code.

**Algorithm part**:
- [ ] Current VI part is using score function training, we need to add reparameterization trick code for it, better a LSTM one.
- [ ] Better way of training Transition Model and Backtracing code;
- [x] The initial of training, it seems that we can barely get final successful trajectories (I test for 1000 episodes and get 0) (Solved by loading policy trained by her);
- [ ] Using Amortized goals for training;
- [ ] A2c based to Proximal or trust region policy training
- [ ] The reward function is still too sparse, maybe we need to replace it with euclidean distance (L2 seems to be not very good)
