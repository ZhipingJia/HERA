import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import time
from .quant_util import *
# The measured on-chip noise table (sample_noise_data.py + .npy) is proprietary
# hardware-characterization data and is NOT distributed with this repository.
# Gaussian sample-noise modes work without it; measured-noise modes raise at
# first use unless the user supplies their own sample_noise_data module.
class _SampleNoiseDataPlaceholder:
    def __getattr__(self, name):
        raise RuntimeError(
            "sample_noise_data (measured on-chip noise table) is not distributed "
            "with this repository; provide quantization_and_noise/sample_noise_data.py "
            "with your own measurements to enable measured-noise modes."
        )
try:
    import quantization_and_noise.sample_noise_data as sample_noise_data
except Exception:
    sample_noise_data = _SampleNoiseDataPlaceholder()
from .base_operator import twn_n, twn_n_nolimit
import random
from torch.autograd import Function
device = 'cuda:0'
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# 全局变量缓存所有batch的输入和输出
ONCHIP_INPUTS_CACHE = []
ONCHIP_OUTPUTS_CACHE = []
SAVE_ONCHIP_INPUT = False
SAVE_ONCHIP_OUTPUT = False
REPLACE_ONCHIP_OUTPUT = False
ONCHIP_REPLACEMENT_INDEX = 0


def _resolve_sample_noise_fn(sample_noise_mode):
    if callable(sample_noise_mode):
        return sample_noise_mode

    if sample_noise_mode is None:
        sample_noise_mode = 'sample_noise_2'
    sample_noise_mode = str(sample_noise_mode).lower()
    sample_noise_fns = {
        'sample_noise': sample_noise,
        'real': sample_noise,
        'lookup': sample_noise,
        'sample_noise_2': sample_noise_2,
        'fake': sample_noise_2,
        'fake_clamp': sample_noise_2,
        'sample_noise_3': sample_noise_3,
        'fake_no_output_clamp': sample_noise_3,
    }
    if sample_noise_mode not in sample_noise_fns:
        valid_modes = ', '.join(sorted(sample_noise_fns))
        raise ValueError(f"Unsupported sample_noise_mode '{sample_noise_mode}'. Valid modes: {valid_modes}")
    return sample_noise_fns[sample_noise_mode]


def _as_kwargs(kwargs):
    if kwargs is None:
        return {}
    return dict(kwargs)


# Measured on-chip INT8 weights / reference outputs used only by the
# linear_quant_sample_noise_c160_onchip debugging path.  They are hardware
# characterization data and are not distributed; set the environment variables
# below to your own measurements to enable that path.
import os as _os


class _LazyOnchipArray:
    def __init__(self, env_var, allow_pickle=False):
        self._env_var = env_var
        self._allow_pickle = allow_pickle
        self._data = None

    def _load(self):
        if self._data is None:
            path = _os.environ.get(self._env_var, "")
            if not path or not _os.path.isfile(path):
                raise RuntimeError(
                    f"Measured on-chip data is not distributed with this repository; "
                    f"set {self._env_var} to your own file to enable the on-chip "
                    f"emulation debugging path."
                )
            self._data = np.load(path, allow_pickle=self._allow_pickle)
        return self._data

    def __getitem__(self, key):
        return self._load()[key]


onchip_weight = _LazyOnchipArray("PRIVATE_LLM_ONCHIP_WEIGHT_NPZ")
onchip_output = _LazyOnchipArray("PRIVATE_LLM_ONCHIP_OUTPUT_PKL", allow_pickle=True)

class conv2d_quant_noise(nn.Conv2d):
    def __init__(self,
                 m: nn.Conv2d,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 ):
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_noise, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.a_out_quantizer.int_flag = int_flag
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)
    
    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]  # int * scale
            else:
                input = input[0]
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        x = self._conv_forward(input_q, weight_q, self.bias)
        return self.a_out_quantizer(x)

class conv2d_quant_sample_noise(nn.Conv2d):
    def __init__(self,
                 m: nn.Conv2d,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 sample_out_scale=0.,
                 sample_noise_mode='sample_noise_2',
                 sample_noise_kwargs=None
                 ):
        # 接受 nn.Conv2d 或者已经量化的 Conv2d 层
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_sample_noise, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.w_quantizer.int_flag = int_flag
        self.a_quantizer.int_flag = int_flag
        self.a_out_quantizer.int_flag = int_flag
        self.sample_out_scale = nn.Parameter(torch.tensor(sample_out_scale))
        self.sample_noise_mode = sample_noise_mode
        self.sample_noise_kwargs = _as_kwargs(sample_noise_kwargs)
        self.sample_noise_fn = _resolve_sample_noise_fn(sample_noise_mode)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]  # int * scale
            else:
                input = input[0]

        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)

        weight_int = weight_q[0]
        input_int = input_q[0]

        x = F.conv2d(input_int, weight_int, self.bias,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)

        y = self.sample_noise_fn(x, **self.sample_noise_kwargs)

        out = y * self.sample_out_scale

        return out

class linear_quant_noise(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 ):
        assert isinstance(m, nn.Linear), f"Expected nn.Linear or subclass, got {type(m)}"
        super(linear_quant_noise, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        if self.a_out_quantizer is not None:
            self.a_out_quantizer.int_flag = int_flag
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)
            print('init weight scale:',w_quantizer.s)
        

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]  # int * scale
            else:
                input = input[0]
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        # print('input_q max:',input_q.max())
        # print('input_q min:',input_q.min())
        #print('weight_q.shape',weight_q.shape)
        #print('input_q.shape',input_q.shape)
        #weight_q.register_hook(lambda grad: print('weight_q grad:', grad))
        #input_q.register_hook(lambda grad: print('input_q grad:', grad))
        x = F.linear(
            input_q,
            weight_q,
            self.bias,
        )
        if self.a_out_quantizer is not None:
            output = self.a_out_quantizer(x)
            if isinstance(output, tuple):
                if output[1] != 0.0:
                    output = output[0] * output[1]  # int * scale
                else:
                    output = output[0]
            #x.register_hook(lambda grad: print('x grad:', grad))
        else:
            output = x
        return output

class linear_quant_sample_noise(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 sample_out_scale = 0.,
                 sample_noise_mode='sample_noise_2',
                 sample_noise_kwargs=None
                 ):
        # 接受 nn.Linear 或者已经量化的 Linear 层
        assert isinstance(m, nn.Linear), f"Expected nn.Linear or subclass, got {type(m)}"
        super(linear_quant_sample_noise, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.w_quantizer.int_flag = int_flag
        self.a_quantizer.int_flag = int_flag
        self.a_out_quantizer.int_flag = int_flag
        self.sample_out_scale = nn.Parameter(torch.tensor(sample_out_scale))
        self.sample_noise_mode = sample_noise_mode
        self.sample_noise_kwargs = _as_kwargs(sample_noise_kwargs)
        self.sample_noise_fn = _resolve_sample_noise_fn(sample_noise_mode)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)
        

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale
    
    def forward(self, input):
        #start_time = time.time()
        #print("Start time:", start_time)
        
        # Initialize process_tuple_time to start_time to avoid unbound error
        #process_tuple_time = start_time

        if isinstance(input, tuple):
            #check_tuple_time = time.time()
            #print("Time after checking if input is tuple:", check_tuple_time - start_time)
            
            if input[1] != 0.0:
                input = input[0] * input[1]  # int * scale
            else:
                input = input[0]
            #process_tuple_time = time.time()
            #print("Time after processing tuple input:", process_tuple_time - check_tuple_time)

        weight_q = self.w_quantizer(self.weight)
        #quantize_weight_time = time.time()
        #print("Time after weight quantization:", quantize_weight_time - process_tuple_time)

        input_q = self.a_quantizer(input)
        #quantize_input_time = time.time()
        #print("Time after input quantization:", quantize_input_time - quantize_weight_time)

        weight_int = weight_q[0]
        input_int = input_q[0]

        x = F.linear(input_int, weight_int, self.bias)
        #linear_time = time.time()
        #print("Time after F.linear computation:", linear_time - quantize_input_time)

        y = self.sample_noise_fn(x, **self.sample_noise_kwargs)
        #noise_sample_time = time.time()
        #print("Time after noise sampling:", noise_sample_time - linear_time)

        out = y * self.sample_out_scale
        #end_time = time.time()
        #print("Time after final scaling:", end_time - noise_sample_time)
        #print("Total time for forward pass:", end_time - start_time)

        return out

class linear_quant_sample_noise_debug(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 sample_out_scale = 0.
                 ):
        assert type(m) == nn.Linear
        super(linear_quant_sample_noise_debug, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        print("int_flag:", int_flag)
        self.w_quantizer.int_flag = int_flag
        self.a_quantizer.int_flag = int_flag
        self.a_out_quantizer.int_flag = int_flag
        self.sample_out_scale = nn.Parameter(torch.tensor(sample_out_scale))
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)
        

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale
    
    def forward(self, input):
        weight_q = self.w_quantizer(self.weight)

        input_q = self.a_quantizer(input)

        weight_float = weight_q[0] * 0.5 * self.sample_out_scale / (input_q[1])
        input = input_q[0]*input_q[1]

        out = F.linear(input, weight_float, self.bias)

        return out

class linear_debug(nn.Linear):
    def __init__(self,
                 m: nn.Linear
                 ):
        assert type(m) == nn.Linear
        super(linear_debug, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.weight = nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
        
    
    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)

        return out

    # def forward(self, input):
    #     if isinstance(input, tuple):
    #         if input[1] != 0.0:
    #             input = input[0] * input[1]  # int * scale
    #         else:
    #             input = input[0]
    #     #print("self.weight.requires_grad:", self.weight.requires_grad)
    #     #self.weight.register_hook(lambda grad: print('self.weight grad',grad))
    #     weight_q = self.w_quantizer(self.weight)
    #     # print('type(weight_q)',type(weight_q))
    #     input_q = self.a_quantizer(input)
    #     # print('type(weight_q)',type(input_q))
    #     weight_int = weight_q[0]
    #     input_int = input_q[0]
    #     #print('weight_int dtype:',weight_int.dtype)
    #     #print('input_int dtype:',input_int.dtype)
    #     #input_int.register_hook(lambda grad: print('input_int grad min',grad.min()))
    #     #weight_int.register_hook(lambda grad: print('weight_int grad min',grad.min()))
    #     # print('weight_int.shape:',weight_int.shape)
    #     # print('input_int.shape:',input_int.shape)
    #     x = F.linear(
    #         input_int,
    #         weight_int,
    #         self.bias,
    #     )
    #     # print('self.bias')
    #     # print(self.bias)
    #     # Register hooks to print gradients during backpropagation
    #     # x.register_hook(lambda grad: print('x grad min:', grad.min()))
    #     # print('x max:',x.max())
    #     # print('x min:',x.min())
    #     y = sample_noise(x)
    #     #y = sample_noise_2(x)
    #     #print('y.shape:',y.shape)
    #     # y.register_hook(lambda grad: print('y grad min:', grad.min()))
    #     # print('y max:',y.max())
    #     # print('y min:',y.min())
    #     # out_q = self.a_out_quantizer(y)
    #     # print('type(out_q)',type(out_q))
    #     # out = y*2*weight_q[1]*input_q[1]
    #     out = y*self.sample_out_scale
    #     # print('init scale:',2*weight_q[1]*input_q[1])
    #     # print('now scale:',self.sample_out_scale)
    #     # print('out.shape:',out.shape)
    #     # print('out tensor:',out)
    #     # print('out max:',out.max())
    #     # print('out min:',out.min())
    #     # out.register_hook(lambda grad: print('out grad min:', grad.min()))
    #     return out

def fake_onchip_sample_noise_np(x):
    y = x * 0.5
    output_min, output_max = -127, 127
    y = np.clip(y, output_min, output_max)
    return y

def fake_onchip_fc(layer_name, input_np):
    x_numpy = input_np @ onchip_weight[layer_name]
    y_numpy = fake_onchip_sample_noise_np(x_numpy)
    return y_numpy.astype('int8')

class linear_quant_sample_noise_c160_onchip(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 sample_out_scale = 0.,
                 layer_name = None
                 ):
        assert type(m) == nn.Linear
        super(linear_quant_sample_noise_c160_onchip, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.w_quantizer.int_flag = int_flag
        self.a_quantizer.int_flag = int_flag
        self.a_out_quantizer.int_flag = int_flag
        self.sample_out_scale = nn.Parameter(torch.tensor(sample_out_scale))
        self.layer_name = layer_name
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)
        

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale
    
    def forward(self, input):
        #start_time = time.time()
        #print("Start time:", start_time)
        
        # Initialize process_tuple_time to start_time to avoid unbound error
        #process_tuple_time = start_time

        if isinstance(input, tuple):
            #check_tuple_time = time.time()
            #print("Time after checking if input is tuple:", check_tuple_time - start_time)
            
            if input[1] != 0.0:
                input = input[0] * input[1]  # int * scale
            else:
                input = input[0]
            #process_tuple_time = time.time()
            #print("Time after processing tuple input:", process_tuple_time - check_tuple_time)

        weight_q = self.w_quantizer(self.weight)
        #quantize_weight_time = time.time()
        #print("Time after weight quantization:", quantize_weight_time - process_tuple_time)

        input_q = self.a_quantizer(input)
        #quantize_input_time = time.time()
        #print("Time after input quantization:", quantize_input_time - quantize_weight_time)

        weight_int = weight_q[0]
        input_int = input_q[0]
        #print("c160 onchip!!!", flush=True)
        input_np = input_int.float().cpu().numpy().astype('int8')
        # print(input_np.max())
        # print(input_np.min())
        #print(input_np.dtype, flush=True)
        # print(input_np.shape, flush=True)
        # input_np = sdk.print_onchip_info(self.layer_name, input_np)
        if SAVE_ONCHIP_INPUT:
            ONCHIP_INPUTS_CACHE.append(input_np)
        y_numpy = fake_onchip_fc(self.layer_name, input_np)
            # y_numpy = sdk.onchip_fc(self.layer_name, input_np)
        #print('y_numpy dtype:', flush=True)
        # ONCHIP_OUTPUTS_CACHE.append(y_numpy)
        #print(y_numpy.dtype, flush=True)
        # print(x_numpy.shape, flush=True)
        y_numpy = y_numpy.copy()
        if SAVE_ONCHIP_OUTPUT:
            ONCHIP_OUTPUTS_CACHE.append(y_numpy)
        if REPLACE_ONCHIP_OUTPUT:
            index = ONCHIP_REPLACEMENT_INDEX
            y_numpy = ONCHIP_OUTPUTS[index]
            ONCHIP_REPLACEMENT_INDEX += 1
        # print(weight_int, flush=True)
        # input_tensor = torch.from_numpy(input_np)
        # input_tensor = input_tensor.to(weight_int.device)
        # input_tensor_bf16 = input_tensor.to(torch.bfloat16)

        # x = F.linear(input_tensor_bf16, weight_int, self.bias)
        # print('input_int.dtype:',input_int.dtype)
        # print('weight_int.dtype:',weight_int.dtype)
        # print('x.dtype:',x.dtype)
        # print('input_int.shape:',input_int.shape)
        # print('weight_int.shape:',weight_int.shape)
        # print('x.shape:',x.shape)
        #linear_time = time.time()
        #print("Time after F.linear computation:", linear_time - quantize_input_time)

        y = torch.from_numpy(y_numpy)
        y = y.to(weight_int.device)
        y = y.to(torch.bfloat16)
        #y = sample_noise_2(x)
        #noise_sample_time = time.time()
        #print("Time after noise sampling:", noise_sample_time - linear_time)

        out = y * self.sample_out_scale

        # 保存当前batch的输出
        
        #end_time = time.time()
        #print("Time after final scaling:", end_time - noise_sample_time)
        #print("Total time for forward pass:", end_time - start_time)

        return out


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class SampleNoiseFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # random.seed(1234)
        # torch.random.manual_seed(1234)
        # 将 offchip2onchip_array 移动到 x 的设备并转换为相同的数据类型
        offchip2onchip_array = sample_noise_data.offchip2onchip_array.to(x.device, dtype=x.dtype)

        # 将常量转换为张量，并确保数据类型和设备与 x 一致
        data_min = torch.tensor(sample_noise_data.data_min, dtype=x.dtype, device=x.device)
        data_max = torch.tensor(sample_noise_data.data_max, dtype=x.dtype, device=x.device)
        # print('sample_noise_data.data_max:',sample_noise_data.data_max)
        # print('data_max:',data_max)
        y = x.clamp(data_min, data_max)
        # print('y max:',y.max())
        y = round_pass(y)
        # print('y max 2:',y.max())
        y_shape = y.shape
        y = y.reshape(-1)
        rows_select = y - data_min
        # print('rows_select max:',rows_select.max())
        t = random.randint(0, sample_noise_data.max_fea_num - y.shape[0])
        # print("t")
        # print(t)
        cols_select_all = torch.randint(0, sample_noise_data.sample_num, (sample_noise_data.max_fea_num,), device=x.device)
        # print("cols_select_all:")
        # print(cols_select_all)
        cols_select = cols_select_all[t:t + y.shape[0]]
        # print("cols_select")
        # print(cols_select)
        # 获取 rows_select 的最小值和最大值，并转换为高精度浮点数
        # rows_select_min = rows_select.min().to(torch.float32)
        # rows_select_max = rows_select.max().to(torch.float32)
        # cols_select_min = cols_select.min().to(torch.float32)
        # cols_select_max = cols_select.max().to(torch.float32)

        # # 在索引之前检查索引范围
        # #print('offchip2onchip_array.shape[0]:',offchip2onchip_array.shape[0])
        # assert rows_select_min >= 0 and rows_select_max < offchip2onchip_array.shape[0], f"rows_select out of bounds: min {rows_select_min}, max {rows_select_max}"
        # assert cols_select_min >= 0 and cols_select_max < offchip2onchip_array.shape[1], f"cols_select out of bounds: min {cols_select_min}, max {cols_select_max}"
        y = offchip2onchip_array[rows_select.long(), cols_select.long()]
        #print("sample noise data:", sample_noise_data.offchip2onchip_array)
        y = y.reshape(y_shape)

        # 保存 x 和 alpha，并确保 alpha 的数据类型和设备与 x 一致
        ctx.save_for_backward(x)
        ctx.alpha = torch.tensor(sample_noise_data.noise_map_scale, dtype=x.dtype, device=x.device)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha

        # 将 grad_output 和 alpha 转换为与 x 相同的数据类型和设备
        grad_output = grad_output.to(x.dtype)
        alpha = alpha.to(x.dtype)

        # 计算梯度
        grad_input = grad_output * alpha

        # 确保 grad_input 的数据类型和设备与 x 一致
        grad_input = grad_input.to(device=x.device, dtype=x.dtype)


        return grad_input

def sample_noise(x):
    return SampleNoiseFunction.apply(x)


# class SampleNoiseFunction(Function):
#     @staticmethod
#     def forward(ctx, x):
#         sample_noise_data.offchip2onchip_array = sample_noise_data.offchip2onchip_array.to(x.device)
#         y = x.clamp(sample_noise_data.data_min, sample_noise_data.data_max)
#         y = round_pass(y)
#         y_shape = y.shape
#         y = y.reshape(-1)
#         rows_select = y - sample_noise_data.data_min
#         t = random.randint(0, sample_noise_data.max_fea_num - y.shape[0])
#         cols_select_all = torch.randint(0, sample_noise_data.sample_num, (sample_noise_data.max_fea_num,), device=x.device)
#         cols_select = cols_select_all[t:t + y.shape[0]]
#         y = sample_noise_data.offchip2onchip_array[rows_select.int(), cols_select]
#         y = y.reshape(y_shape)

#         ctx.save_for_backward(x)
#         ctx.alpha = sample_noise_data.noise_map_scale

#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         alpha = ctx.alpha

#         # 直接将梯度乘以系数 alpha
#         grad_input = grad_output * alpha

#         return grad_input

# def sample_noise(x):
#     return SampleNoiseFunction.apply(x)

# def sample_noise(x:Tensor):
#     y = x.clamp(sample_noise_data.data_min,sample_noise_data.data_max)
#     y = round_pass(y)
#     y_shape = y.shape
#     y = y.reshape(-1)
#     rows_select = y-sample_noise_data.data_min
#     t = random.randint(0,sample_noise_data.max_fea_num-y.shape[0])
#     cols_select_all = torch.randint(0,sample_noise_data.sample_num,(sample_noise_data.max_fea_num,)).cuda()
#     cols_select = cols_select_all[t:t+y.shape[0]]
#     # cols_select = torch.randint(0,sample_num,data.shape).cuda()
#     y = sample_noise_data.offchip2onchip_array[rows_select.int(), cols_select]
#     y = y.reshape(y_shape)
#     return y

class SampleNoise2Function(Function):
    @staticmethod
    def forward(ctx, x, scale_factor, output_min, output_max, noise_std):
        data_min = torch.tensor(sample_noise_data.data_min, dtype=x.dtype, device=x.device)
        data_max = torch.tensor(sample_noise_data.data_max, dtype=x.dtype, device=x.device)

        y = x.clamp(data_min, data_max)
        y = y * scale_factor
        if noise_std is not None and torch.any(noise_std != 0):
            y = y + torch.randn_like(y) * noise_std
        y = y.clamp(output_min, output_max)

        ctx.save_for_backward(x, scale_factor)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, scale_factor = ctx.saved_tensors

        grad_input = grad_output * scale_factor
        grad_input = grad_input.to(device=x.device, dtype=x.dtype)

        return grad_input, None, None, None, None


def sample_noise_2(x, scale_factor=None, output_min=-127.0, output_max=127.0, noise_std=20.0):
    if scale_factor is None:
        scale_factor = sample_noise_data.noise_map_scale
    if noise_std is None:
        noise_std = 0.0
    scale_factor = torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    output_min = torch.as_tensor(output_min, dtype=x.dtype, device=x.device)
    output_max = torch.as_tensor(output_max, dtype=x.dtype, device=x.device)
    noise_std = torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    return SampleNoise2Function.apply(x, scale_factor, output_min, output_max, noise_std)

class SampleNoise3Function(Function):
    @staticmethod
    def forward(ctx, x, noise_std):
        # 定义输入截断的范围 [-800, 799]
        data_min = torch.tensor(sample_noise_data.data_min, dtype=x.dtype, device=x.device)
        data_max = torch.tensor(sample_noise_data.data_max, dtype=x.dtype, device=x.device)

        # 对输入进行截断
        y = x.clamp(data_min, data_max)
        
        # 乘以缩放系数 0.5
        y = y * 0.5
        
        # 添加高斯噪声，使用传入的标准差 noise_std
        noise = torch.randn_like(y) * noise_std  # 生成具有指定标准差的高斯噪声
        y = y + noise  # 加上噪声
        
        # 定义输出截断的范围 [-128, 127]
        output_min = torch.tensor(-127.0, dtype=x.dtype, device=x.device)
        output_max = torch.tensor(127.0, dtype=x.dtype, device=x.device)
        
        # 对输出进行截断
        # y = y.clamp(output_min, output_max)
        
        # 保存 x 和缩放系数 0.5，用于反向传播
        ctx.save_for_backward(x)
        ctx.alpha = torch.tensor(0.5, dtype=x.dtype, device=x.device)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 获取在 forward 中保存的变量
        x, = ctx.saved_tensors
        alpha = ctx.alpha

        # 计算梯度，并确保数据类型和设备与 x 一致
        grad_input = grad_output * alpha
        grad_input = grad_input.to(device=x.device, dtype=x.dtype)

        return grad_input, None  # None用于噪声标准差的梯度，因为它不是一个需要求导的参数

# 定义一个函数用于调用 SampleNoise2Function，并允许指定噪声标准差
def sample_noise_3(x, noise_std=0.0):
    return SampleNoise3Function.apply(x, noise_std)


class add_quant(nn.Module): # 输入输出都为[int, s]
    def __init__(self, bit=9, all_positive=False, symmetric=True, quant_method=0, shift=0):
        super(add_quant, self).__init__()
        self.bit = bit
        self.a_out_quantizer = LSQ_act_quantizer(bit=bit, all_positive=all_positive, symmetric=symmetric,
                                                 init_mode='percent', init_percent=0.999)
        self.quant_method = quant_method
        self.shift = shift
        self.int_flag = True
        self.a_out_quantizer.int_flag = True
        print('??????????????????')
    
    def forward(self, input1, input2):
        if not (isinstance(input1, tuple) and isinstance(input2, tuple)):
            raise ValueError('add_quant module need quantized input. ')
        if input1[1] == 0.0 or input2[1] == 0.0:
            raise ValueError('add_quant module need quantized input. ')
        a1_int, s1 = input1
        a2_int, s2 = input2
        if self.quant_method == 0:
            if self.shift == 0:
                if s1 > s2:  # 取较大的
                    s = s1
                else:
                    s = s2
                x = torch.clamp(a1_int + a2_int, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return x, s
                else:
                    return x * s
            elif self.shift == 1:
                if s2 > s1:
                    a1_int, a2_int = a2_int, a1_int
                    s1, s2 = s2, s1
                k = twn_n(s1 / s2)
                a2_int = floor_pass(a2_int / (2 ** k))
                int_x = a1_int + a2_int
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return x, s1
                else:
                    return x * s1
        if self.quant_method == 1: # 带a_out_quantizer,待验证效果
            if self.shift == 0:
                return self.a_out_quantizer(a1_int * s1 + a2_int * s2)
            elif self.shift == 1:
                if s2 > s1:
                    a1_int, a2_int = a2_int, a1_int
                    s1, s2 = s2, s1
                k0 = twn_n(s1 / s2)
                a2_int = floor_pass(a2_int / (2 ** k0))
                int_x = a1_int + a2_int
                k1 = 0
                if self.a_out_quantizer.get_scale() / s1 > 1.:
                    k1 = twn_n(self.a_out_quantizer.get_scale() / s1)
                int_x = floor_pass(int_x / (2 ** k1))
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return x, self.a_out_quantizer.get_scale()
                else:
                    return x * self.a_out_quantizer.get_scale()
            elif self.shift == 2:
                k1 = 0
                k2 = 0
                if self.a_out_quantizer.get_scale() / s1 > 1.:
                    k1 = twn_n(self.a_out_quantizer.get_scale() / s1)
                if self.a_out_quantizer.get_scale() / s2 > 1.:
                    k2 = twn_n(self.a_out_quantizer.get_scale() / s2)
                a1_int = floor_pass(a1_int / (2 ** k1))
                a2_int = floor_pass(a2_int / (2 ** k2))
                int_x = a1_int + a2_int
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return x, self.a_out_quantizer.get_scale()
                else:
                    return x * self.a_out_quantizer.get_scale()

class AdaptiveAvgPool2d_quant(nn.AdaptiveAvgPool2d):
    def __init__(self, m: nn.AdaptiveAvgPool2d, quant_flag=False, bit=9, all_positive=False, symmetric=True):
        assert type(m) == nn.AdaptiveAvgPool2d
        assert m.output_size == (1,1)
        super(AdaptiveAvgPool2d_quant, self).__init__(m.output_size)
        self.quant_flag = quant_flag  # if False: normal module func
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            if input[1] == 0.0:
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            input_int, input_scale = input
            input_size = input_int.size()
            self.size_2D = input_size[2]*input_size[3]
            output = F.adaptive_avg_pool2d(input_int, self.output_size) * self.size_2D
            
            k = twn_n_nolimit(self.size_2D)
            output = floor_pass(output / (2 ** k))
            output = torch.clamp(output, min=self.thd_neg, max=self.thd_pos)
            return output, input_scale
        else:
            if isinstance(input, tuple):
                if input[1] != 0.:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.adaptive_avg_pool2d(input, self.output_size)

# class AvgPool2d_quant(nn.AvgPool2d):
#     def __init__(self, m: nn.AvgPool2d, quant_flag=False, bit=9, all_positive=False, symmetric=True):
#         self.c160_mode = True
#         if self.c160_mode == True:
#             print('C160 mode')
#         assert type(m) == nn.AvgPool2d
#         super(AvgPool2d_quant, self).__init__(m.kernel_size, m.stride, m.padding, m.ceil_mode,
#                                               m.count_include_pad, m.divisor_override)
#         self.quant_flag = quant_flag  # if False: normal module func
#         if isinstance(self.kernel_size, tuple):
#             self.size_2D = self.kernel_size[0] * self.kernel_size[1]
#         else:
#             self.size_2D = self.kernel_size ** 2
#         self.bit = bit
#         if all_positive:
#             assert not symmetric, "Positive quantization cannot be symmetric"
#             # unsigned activation is quantized to [0, 2^b-1]
#             self.thd_neg = 0
#             self.thd_pos = 2 ** bit - 1
#         else:
#             if symmetric:
#                 # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
#                 self.thd_neg = - 2 ** (bit - 1) + 1
#                 self.thd_pos = 2 ** (bit - 1) - 1
#             else:
#                 # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
#                 self.thd_neg = - 2 ** (bit - 1)
#                 self.thd_pos = 2 ** (bit - 1) - 1
    
#     def forward(self, input):
#         if self.quant_flag:
#             #print('quant!')
#             if not isinstance(input, tuple):
#                 raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
#             if input[1] == 0.0:
#                 raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
#             input_int, input_scale = input
#             output_int = F.avg_pool2d(input_int, self.kernel_size, self.stride,
#                             self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override) \
#                      * self.size_2D
            
#             k = twn_n_nolimit(self.size_2D)
#             output_int = floor_pass(output_int / (2 ** k))
#             output_int = torch.clamp(output_int, min=self.thd_neg, max=self.thd_pos)
#             return output_int, input_scale
#         else: 
#             if isinstance(input, tuple):
#                 # print('AVGPOOL TUPLE!!!!!')
#                 if self.c160_mode:
#                     input_data = input[0]
#                 else:
#                     if input[1] != 0.:
#                         input = input[0] * input[1]
#                     else:
#                         input = input[0]
#                 return torch.floor(F.avg_pool2d(input_data, self.kernel_size, self.stride,
#                             self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)),input[1]
#             else:
#                 return torch.floor(F.avg_pool2d(input, self.kernel_size, self.stride,
#                             self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override))
            

class AvgPool2d_quant(nn.AvgPool2d):
    def __init__(self, m: nn.AvgPool2d, quant_flag=False, bit=9, all_positive=False, symmetric=True):
        self.c160_mode = True
        if self.c160_mode == True:
            print('C160 mode')
        assert type(m) == nn.AvgPool2d
        super(AvgPool2d_quant, self).__init__(m.kernel_size, m.stride, m.padding, m.ceil_mode,
                                              m.count_include_pad, m.divisor_override)
    
    def forward(self, input):
        if isinstance(input, tuple):
            #print('AVGPOOL TUPLE!!!!!')
            if self.c160_mode:
                input_data = input[0]
            # else:
            #     if input[1] != 0.:
            #         input = input[0] * input[1]
            #     else:
            #         input = input[0]
            # return F.avg_pool2d(input_data, self.kernel_size, self.stride,
            #             self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
            return F.avg_pool2d(input_data, self.kernel_size, self.stride,
                        self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override),input[1]
        else:
            return F.avg_pool2d(input, self.kernel_size, self.stride,
                        self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)


class BatchNorm2d_quant(nn.BatchNorm2d):
    def __init__(self, m: nn.BatchNorm2d, quant_flag=False, quant_method=1, out_bit=0,
                 w_quantizer=None, bias_quantizer=None, a_out_quantizer=None, *args, **kwargs):
        assert type(m) == nn.BatchNorm2d
        super(BatchNorm2d_quant, self).__init__(m.num_features)
        self.weight = nn.Parameter(m.weight.detach())
        self.bias = nn.Parameter(m.bias.detach())
        self.running_var = m.running_var
        self.running_mean = m.running_mean
        self.track_running_stats = m.track_running_stats
        self.num_batches_tracked = m.num_batches_tracked
        self.w_quantizer = w_quantizer
        self.bias_quantizer = bias_quantizer
        self.a_out_quantizer = a_out_quantizer
        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            # self.w_quantizer.init_scale(m.weight)
            self.w_quantizer.init_scale(m.weight / ((m.running_var ** 0.5) + self.eps))
        self.quant_flag = quant_flag  # if False: normal module func
        self.quant_method = quant_method
        if self.quant_method == 2:
            assert out_bit > 1
            self.thd_neg = - 2 ** (out_bit - 1) + 1
            self.thd_pos = 2 ** (out_bit - 1) - 1
        
    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError("BatchNorm2d_quant with quant_flag=True need quantized input.")
            if input[1] == 0.:
                raise ValueError("BatchNorm2d_quant with quant_flag=True need quantized input.")
        if isinstance(input, tuple):
            self._check_input_dim(input[0])
        else:
            self._check_input_dim(input)
        
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        
        if self.quant_flag:
            input_int, input_scale = input
            input_q = input_int * input_scale
            if self.training:
                tmp = F.batch_norm(  # update running_mean and running_var for inference
                    input_q,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
                batch_mean = torch.mean(input_q.permute(1, 0, 2, 3).reshape(self.num_features, -1), dim=1).detach()
                batch_var = torch.var(input_q.permute(1, 0, 2, 3).reshape(self.num_features, -1), dim=1,
                                      unbiased=False).detach()
            else:
                batch_mean = self.running_mean
                batch_var = self.running_var
            weight_tmp = self.weight / ((batch_var ** 0.5) + self.eps)
            bias_tmp = self.bias - batch_mean * weight_tmp
            weight_tmp_int, weight_tmp_scale = self.w_quantizer(weight_tmp)
            if self.quant_method == 1:
                tmp_scale = (input_scale * weight_tmp_scale).detach()
                bias_tmp_int, bias_tmp_scale = self.bias_quantizer(bias_tmp, tmp_scale)
                output = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1) \
                         + (bias_tmp_int * bias_tmp_scale).reshape(1, self.num_features, 1, 1)
                out_int, out_scale = self.a_out_quantizer(output)
                return out_int, out_scale
            elif self.quant_method == 2:
                output_tmp = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1)
                output_int, output_scale = self.a_out_quantizer(output_tmp)
                bias_tmp_int, bias_tmp_scale = self.bias_quantizer(bias_tmp, output_scale.detach())
                output = torch.clamp(round_pass(output_int) + round_pass(bias_tmp_int), self.thd_neg, self.thd_pos)
                return output, output_scale
            elif self.quant_method == 3: # act_out_quant and weight_quant
                output = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1) \
                         + bias_tmp.reshape(1, self.num_features, 1, 1)
                return self.a_out_quantizer(output)
            elif self.quant_method == 4: # only act_out_quantizer
                output = input_q * weight_tmp.reshape(1, self.num_features, 1, 1) \
                         + bias_tmp.reshape(1, self.num_features, 1, 1)
                return self.a_out_quantizer(output)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

class ReLu_quant(nn.ReLU):
    def __init__(self, m: nn.ReLU):
        assert type(m) == nn.ReLU
        super(ReLu_quant, self).__init__(m.inplace)
    def forward(self, input):
        if isinstance(input, tuple):
            return F.relu(input[0], inplace=self.inplace), input[1]
        else:
            return F.relu(input, inplace=self.inplace)

class MaxPool2d_quant(nn.MaxPool2d):
    def __init__(self, m: nn.MaxPool2d):
        assert type(m) == nn.MaxPool2d
        super(MaxPool2d_quant, self).__init__(kernel_size=m.kernel_size, stride=m.stride,
                                              padding=m.padding, dilation=m.dilation)
        
    def forward(self, input):
        if isinstance(input, tuple):
            # print('maxpool tuple')
            return F.max_pool2d(input[0], self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices),input[1]
        else:
            return F.max_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)

class Dropout_quant(nn.Dropout):
    def __init__(self, m:nn.Dropout):
        assert type(m) == nn.Dropout
        super(Dropout_quant, self).__init__(p=m.p, inplace=m.inplace)
        
    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, tuple):
            return F.dropout(input[0], self.p, self.training, self.inplace), input[1]
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

SAMPLE_FLAG = False
if SAMPLE_FLAG:
    Linear_q = linear_quant_sample_noise
else:
    Linear_q = linear_quant_noise

QuanModuleMapping = {
    nn.Conv2d: conv2d_quant_noise,
    nn.Linear: Linear_q
}
QuanModule = [
    conv2d_quant_noise,
    Linear_q
]

ConvMapping = {nn.Conv2d: conv2d_quant_noise}
FcMapping = {nn.Linear: Linear_q}
BnMapping = {nn.BatchNorm2d: BatchNorm2d_quant}
AvgMapping = {
    nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d_quant,
    nn.AvgPool2d: AvgPool2d_quant,
}
OtherMapping = {
    nn.MaxPool2d: MaxPool2d_quant,
    nn.ReLU: ReLu_quant,
    nn.Dropout: Dropout_quant
}

totalMappingModule = [
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.AdaptiveAvgPool2d,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.ReLU,
    nn.Dropout
]
