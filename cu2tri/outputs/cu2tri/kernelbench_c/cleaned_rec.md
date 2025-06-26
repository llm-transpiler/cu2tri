# 01_single_op: level1记录
将level1的08挪到了cublas调用里了
删除了55，调用了torch::conv2d
删除57，用了at::conv_transposed2d
删除58，用了at::conv_transpose3d
修改61，forward里面有防御性编程，但其实没必要
删除62，用了torch::conv2d
删除63，用了torch::conv2d
删除64，用了torch::conv_transpose1d
删除66，代码错误
67的cuda比较奇怪
68的cuda也比较奇怪
删除69，用了at::conv_transpose2d
删除71，用了at::conv_transpose2d
删除72，用了at::conv_transpose3d
删除73，pytorch代码有未定义引用
删除78，用了at::conv_transpose2d
79的cuda有点奇怪
81，82的cuda有点奇怪
86的wrapper有点多
删除91，因为没有kernel，直接torch操作了
92的torch实现有点怪
94用了atomAdd和tensor的.div_
96也用了atomAdd
100用了torch::mean，但是保留

删除了13个

# 02
18有atmoAdd
51有torch::sum，但感觉无伤大雅，好像前面也有一个最后torch::mean的，删没删忘了
66有随机性dropout
94的pybind有点奇怪
# 03
49也没保留，有cat,index,einstane