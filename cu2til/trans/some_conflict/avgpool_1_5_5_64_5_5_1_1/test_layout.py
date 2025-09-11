import torch
n = 1
c = 64
h = w = 5
input_nchw = torch.randn(n, c, h, w)
print(input_nchw.shape)
print(input_nchw.stride())
print(input_nchw.is_contiguous(memory_format=torch.channels_last))
print(input_nchw.is_contiguous(memory_format=torch.contiguous_format))

input_nhwc = input_nchw.permute(0, 2, 3, 1)
print(input_nhwc.shape)
print(input_nhwc.stride())
print(input_nhwc.is_contiguous(memory_format=torch.channels_last))
print(input_nhwc.is_contiguous(memory_format=torch.contiguous_format))

input_nhwc_1 = input_nchw.contiguous(memory_format=torch.channels_last)
print(input_nhwc_1.shape)
print(input_nhwc_1.stride())
print(input_nhwc_1.is_contiguous(memory_format=torch.channels_last))
print(input_nhwc_1.is_contiguous(memory_format=torch.contiguous_format))

input_nhwc_2 = input_nhwc.contiguous()
print(input_nhwc_2.shape)
print(input_nhwc_2.stride())
print(input_nhwc_2.is_contiguous(memory_format=torch.channels_last))
print(input_nhwc_2.is_contiguous(memory_format=torch.contiguous_format))
