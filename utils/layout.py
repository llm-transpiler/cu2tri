# https://docs.google.com/document/d/1YfVnSPNMvUQHJx1zKNbEZvhbtcFTnWS6YIEWZLjdukg/edit
def to_col_major(x):
    # convert a row major tensor -> col major with contiguous storage
    return x.transpose(-2, -1).contiguous().transpose(-2, -1)

if __name__ == "__main__":
    import torch
    x = torch.randn(2, 3)
    print(x.shape)
    print(x.stride())
    y = x.T
    print(y.shape)
    print(y.stride())
    print(y.is_contiguous())
    # print(y.is_contiguous(memory_format=torch.contiguous_format))
    # print(y.is_contiguous(memory_format=torch.channels_last))
    # print(y.is_contiguous(memory_format=torch.channels_last_3d))
    y = y.contiguous()
    print(y.shape)
    print(y.stride())
    print(y.is_contiguous())
    
    y = to_col_major(x)
    print(y.shape)
    print(y.stride())
    print(y.is_contiguous())