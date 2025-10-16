/workspace/server/nvgpu/task_runner.py等中
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

这里取的是不是GPU的在server中的logical id，实际上需要取它的cuda_visible_id，因为可能不一致

是否可以添加TORCH_CUDA_ARCH_LIST, 根据每个卡的情况, Ada, Hopper之类的
另外，是否可以添加CUDA_ARCH_LIST=8.9...(hopper就是9.0)
注意，如果指定的运行包含了多个GPU需要同时指定多个（单个任务比较少见，但不一定不存在）

所有的task现在只传递id，能不能把label等信息也传入，主要是task_label，是不是可以传Task类

Timer的作用是去计时各个阶段的时间，只有log里面各个阶段时间还是用datetime之类，所有task的阶段duration时间需要用Timer