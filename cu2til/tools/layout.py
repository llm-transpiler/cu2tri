
def convert_nchw_to_nhwc(x_nchw):
    """Convert tensor from NCHW to NHWC layout"""
    return x_nchw.permute(0, 2, 3, 1).contiguous()

def convert_nhwc_to_nchw(x_nhwc):
    """Convert tensor from NHWC to NCHW layout"""  
    return x_nhwc.permute(0, 3, 1, 2).contiguous()
