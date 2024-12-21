# Isaac Lab 学习笔记

## Franka_cabinet_env.py

- F1EnvCf(gDirectRLEnvCfg)

  1. 初始化仿真环境设置
  2. 加载仿真模型或文件
  3. 计算robot 和 object 位姿

- F1Env(DirectRLEnv)

  1. get env local pose

     计算各种各样的位姿

  2. physics step

     apply action，take action

  3. get dones

     检查是否一个episode结束

  4. get rewards

     计算rewards

  5. reset_idx

     重置环境

  6. get observations

### 函数功能

**get_observation函数**

input:

- None

return：

- dof/joint pse
- robot joint velocity
- target error
- object pos and velocity

**compute_intermediate_values函数**

compute robot and drawer`s postion and rotation as state

Input: 

- env_ids: torch.Tensor

Output:

- None

### 环境总结

强化学习的三个重要组成，在这个`Franka_cabinet_env.py`中，是如何定义这三个component?

- state

  通过`get_observations`来获得，包括：

  - robot joint pose and velocity
  - target error
  - object pose and velocity

- action

  通过库函数来算action，在`F1Env`中将action应用pid和clamp

- reward

  1. robot与object之间的距离

  2. 夹爪与object对应程度（正着，夹爪的rot要与object的rot差不多，这样才能对接起来）

  3. 打开的奖励

  4. 动作惩罚

     > 动作越多惩罚越多

  5. 夹爪左右指与object的距离

     > lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
     >
     > rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]

  6. 夹爪手指的惩罚





