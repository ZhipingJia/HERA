import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Function
from .quant_util import *
from .base_operator import twn_n, twn_n_nolimit


DEFAULT_SAMPLE_NOISE_DATA_MIN = -800.0
DEFAULT_SAMPLE_NOISE_DATA_MAX = 799.0
DEFAULT_SAMPLE_NOISE_SCALE = 0.5
DEFAULT_SAMPLE_NOISE_OUTPUT_MIN = -127.0
DEFAULT_SAMPLE_NOISE_OUTPUT_MAX = 127.0


def _as_kwargs(kwargs):
    if kwargs is None:
        return {}
    return dict(kwargs)


def _resolve_sample_noise_fn(sample_noise_mode):
    if callable(sample_noise_mode):
        return sample_noise_mode

    if sample_noise_mode is None:
        sample_noise_mode = 'sample_noise_2'
    sample_noise_mode = str(sample_noise_mode).lower()
    sample_noise_fns = {
        'sample_noise_2': sample_noise_2,
        'sample_noise_2_both': sample_noise_2,
        'fake': sample_noise_2,
        'fake_clamp': sample_noise_2,
        'sample_noise_3': sample_noise_3,
        'fake_no_output_clamp': sample_noise_3,
        'sample_noise_4': sample_noise_4,
        'fake_no_input_clamp': sample_noise_4,
        'sample_noise_no_input_clamp': sample_noise_4,
        'sample_noise_5': sample_noise_5,
        'fake_no_clamp': sample_noise_5,
        'sample_noise_no_clamp': sample_noise_5,
    }
    if sample_noise_mode not in sample_noise_fns:
        valid_modes = ', '.join(sorted(sample_noise_fns))
        raise ValueError(f"Unsupported sample_noise_mode '{sample_noise_mode}'. Valid modes: {valid_modes}")
    return sample_noise_fns[sample_noise_mode]


class SampleNoise2Function(Function):
    @staticmethod
    def forward(ctx, x, scale_factor, output_min, output_max, data_min, data_max, noise_std):
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
        return grad_input, None, None, None, None, None, None


def sample_noise_2(
    x,
    scale_factor=DEFAULT_SAMPLE_NOISE_SCALE,
    output_min=DEFAULT_SAMPLE_NOISE_OUTPUT_MIN,
    output_max=DEFAULT_SAMPLE_NOISE_OUTPUT_MAX,
    data_min=DEFAULT_SAMPLE_NOISE_DATA_MIN,
    data_max=DEFAULT_SAMPLE_NOISE_DATA_MAX,
    noise_std=20.0,
):
    """Fake sample-noise default used by ACIM affinity experiments."""
    if noise_std is None:
        noise_std = 0.0
    scale_factor = torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    output_min = torch.as_tensor(output_min, dtype=x.dtype, device=x.device)
    output_max = torch.as_tensor(output_max, dtype=x.dtype, device=x.device)
    data_min = torch.as_tensor(data_min, dtype=x.dtype, device=x.device)
    data_max = torch.as_tensor(data_max, dtype=x.dtype, device=x.device)
    noise_std = torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    return SampleNoise2Function.apply(x, scale_factor, output_min, output_max, data_min, data_max, noise_std)


def sample_noise_3(x, noise_std=0.0, scale_factor=DEFAULT_SAMPLE_NOISE_SCALE,
                   data_min=DEFAULT_SAMPLE_NOISE_DATA_MIN, data_max=DEFAULT_SAMPLE_NOISE_DATA_MAX):
    """Fake sample-noise variant without output clamp."""
    data_min = torch.as_tensor(data_min, dtype=x.dtype, device=x.device)
    data_max = torch.as_tensor(data_max, dtype=x.dtype, device=x.device)
    scale_factor = torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    if noise_std is None:
        noise_std = 0.0
    noise_std = torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    y = x.clamp(data_min, data_max) * scale_factor
    if torch.any(noise_std != 0):
        y = y + torch.randn_like(y) * noise_std
    return y


def sample_noise_4(
    x,
    noise_std=0.0,
    scale_factor=DEFAULT_SAMPLE_NOISE_SCALE,
    output_min=DEFAULT_SAMPLE_NOISE_OUTPUT_MIN,
    output_max=DEFAULT_SAMPLE_NOISE_OUTPUT_MAX,
    data_min=DEFAULT_SAMPLE_NOISE_DATA_MIN,
    data_max=DEFAULT_SAMPLE_NOISE_DATA_MAX,
):
    """Fake sample-noise variant without input clamp, keeping output clamp."""
    del data_min, data_max
    scale_factor = torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    output_min = torch.as_tensor(output_min, dtype=x.dtype, device=x.device)
    output_max = torch.as_tensor(output_max, dtype=x.dtype, device=x.device)
    if noise_std is None:
        noise_std = 0.0
    noise_std = torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    y = x * scale_factor
    if torch.any(noise_std != 0):
        y = y + torch.randn_like(y) * noise_std
    return y.clamp(output_min, output_max)


def sample_noise_5(
    x,
    noise_std=0.0,
    scale_factor=DEFAULT_SAMPLE_NOISE_SCALE,
    output_min=DEFAULT_SAMPLE_NOISE_OUTPUT_MIN,
    output_max=DEFAULT_SAMPLE_NOISE_OUTPUT_MAX,
    data_min=DEFAULT_SAMPLE_NOISE_DATA_MIN,
    data_max=DEFAULT_SAMPLE_NOISE_DATA_MAX,
):
    """Fake sample-noise variant without input clamp or output clamp."""
    del data_min, data_max, output_min, output_max
    scale_factor = torch.as_tensor(scale_factor, dtype=x.dtype, device=x.device)
    if noise_std is None:
        noise_std = 0.0
    noise_std = torch.as_tensor(noise_std, dtype=x.dtype, device=x.device)
    y = x * scale_factor
    if torch.any(noise_std != 0):
        y = y + torch.randn_like(y) * noise_std
    return y

class conv2d_quant_noise(nn.Conv2d):
    def __init__(self,
                 m: nn.Conv2d,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 ):
        assert type(m) == nn.Conv2d
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


def _dequantize_static_weight(weight, w_quantizer):
    weight_int, scale = w_quantizer.get_int(weight)
    scale = torch.as_tensor(scale, dtype=weight.dtype, device=weight.device)
    return (weight_int.detach().to(dtype=weight.dtype) * scale).detach()


def _quantize_static_weight_int(weight, w_quantizer):
    weight_int, _scale = w_quantizer.get_int(weight)
    return weight_int.detach().to(dtype=weight.dtype)


class conv2d_quant_noise_folded_weight(nn.Conv2d):
    """conv2d_quant_noise with static weight quantization folded into weight."""

    def __init__(self,
                 m: nn.Conv2d,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 folded_weight=None,
                 ):
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_noise_folded_weight, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.a_quantizer = a_quantizer if a_quantizer is not None else NoQuan()
        self.a_out_quantizer = a_out_quantizer if a_out_quantizer is not None else NoQuan()
        if hasattr(self.a_out_quantizer, 'int_flag'):
            self.a_out_quantizer.int_flag = int_flag

        if folded_weight is None:
            folded_weight = m.weight.detach()
        self.weight = nn.Parameter(folded_weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())

    @classmethod
    def from_quant_noise(cls, module):
        folded_weight = _dequantize_static_weight(module.weight, module.w_quantizer)
        return cls(
            module,
            a_quantizer=module.a_quantizer,
            a_out_quantizer=module.a_out_quantizer,
            int_flag=getattr(module.a_out_quantizer, 'int_flag', False),
            folded_weight=folded_weight,
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        input_q = self.a_quantizer(input)
        x = self._conv_forward(input_q, self.weight, self.bias)
        return self.a_out_quantizer(x)


class conv2d_quant_noise_folded_weight_no_act(nn.Conv2d):
    """Folded-weight conv2d_quant_noise with input activation quantizer removed."""

    def __init__(self,
                 m: nn.Conv2d,
                 a_out_quantizer=None,
                 int_flag=False,
                 folded_weight=None,
                 ):
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_noise_folded_weight_no_act, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.a_out_quantizer = a_out_quantizer if a_out_quantizer is not None else NoQuan()
        if hasattr(self.a_out_quantizer, 'int_flag'):
            self.a_out_quantizer.int_flag = int_flag

        if folded_weight is None:
            folded_weight = m.weight.detach()
        self.weight = nn.Parameter(folded_weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())

    @classmethod
    def from_folded_weight(cls, module):
        return cls(
            module,
            a_out_quantizer=module.a_out_quantizer,
            int_flag=getattr(module.a_out_quantizer, 'int_flag', False),
            folded_weight=module.weight.detach(),
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        x = self._conv_forward(input, self.weight, self.bias)
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
                 sample_noise_kwargs=None,
                 init_weight_scale=False
                 ):
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
        if self.a_out_quantizer is not None:
            self.a_out_quantizer.int_flag = int_flag
        sample_out_scale = torch.as_tensor(sample_out_scale, dtype=self.weight.dtype, device=self.weight.device)
        self.sample_out_scale = nn.Parameter(sample_out_scale.detach().clone())
        self.sample_noise_mode = sample_noise_mode
        self.sample_noise_kwargs = _as_kwargs(sample_noise_kwargs)
        self.sample_noise_fn = _resolve_sample_noise_fn(sample_noise_mode)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if init_weight_scale and isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
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


class conv2d_quant_sample_noise_int_weight(nn.Conv2d):
    """conv2d_quant_sample_noise with weight_int stored directly as weight."""

    def __init__(self,
                 m: nn.Conv2d,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=True,
                 sample_out_scale=0.,
                 sample_noise_mode='sample_noise_2',
                 sample_noise_kwargs=None,
                 int_weight=None,
                 ):
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_sample_noise_int_weight, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.a_quantizer = a_quantizer if a_quantizer is not None else NoQuan()
        self.a_out_quantizer = a_out_quantizer
        if hasattr(self.a_quantizer, 'int_flag'):
            self.a_quantizer.int_flag = int_flag
        if self.a_out_quantizer is not None and hasattr(self.a_out_quantizer, 'int_flag'):
            self.a_out_quantizer.int_flag = int_flag

        if int_weight is None:
            int_weight = m.weight.detach()
        self.weight = nn.Parameter(int_weight.detach().clone())
        sample_out_scale = torch.as_tensor(sample_out_scale, dtype=self.weight.dtype, device=self.weight.device)
        self.sample_out_scale = nn.Parameter(sample_out_scale.detach().clone())
        self.sample_noise_mode = sample_noise_mode
        self.sample_noise_kwargs = _as_kwargs(sample_noise_kwargs)
        self.sample_noise_fn = _resolve_sample_noise_fn(sample_noise_mode)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())

    @classmethod
    def from_sample_noise(cls, module):
        int_weight = _quantize_static_weight_int(module.weight, module.w_quantizer)
        return cls(
            module,
            a_quantizer=module.a_quantizer,
            a_out_quantizer=module.a_out_quantizer,
            int_flag=getattr(module.a_quantizer, 'int_flag', True),
            sample_out_scale=module.sample_out_scale.detach(),
            sample_noise_mode=getattr(module, 'sample_noise_mode', 'sample_noise_2'),
            sample_noise_kwargs=getattr(module, 'sample_noise_kwargs', None),
            int_weight=int_weight,
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]

        input_q = self.a_quantizer(input)
        input_int = input_q[0] if isinstance(input_q, tuple) else input_q

        x = F.conv2d(input_int, self.weight, self.bias,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)

        y = self.sample_noise_fn(x, **self.sample_noise_kwargs)
        out = y * self.sample_out_scale
        return out


def _reshape_output_scale_for_weight(scale, weight):
    scale = torch.as_tensor(scale, dtype=weight.dtype, device=weight.device)
    if scale.dim() == 0:
        return scale
    return scale.reshape(-1, *([1] * (weight.dim() - 1)))


def _reshape_output_scale_for_bias(scale, bias):
    scale = torch.as_tensor(scale, dtype=bias.dtype, device=bias.device)
    if scale.dim() == 0:
        return scale
    return scale.reshape_as(bias)


class conv2d_quant_sample_noise_linearized(nn.Conv2d):
    """Linearized sample_noise_5 conv, keeping the input activation quantizer."""

    def __init__(self,
                 m: nn.Conv2d,
                 a_quantizer=None,
                 int_flag=True,
                 folded_weight=None,
                 folded_bias=None,
                 ):
        assert isinstance(m, nn.Conv2d), f"Expected nn.Conv2d or subclass, got {type(m)}"
        super(conv2d_quant_sample_noise_linearized, self).__init__(
                         m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.a_quantizer = a_quantizer if a_quantizer is not None else NoQuan()
        if hasattr(self.a_quantizer, 'int_flag'):
            self.a_quantizer.int_flag = int_flag

        if folded_weight is None:
            folded_weight = m.weight.detach()
        self.weight = nn.Parameter(folded_weight.detach().clone())
        if m.bias is not None:
            if folded_bias is None:
                folded_bias = m.bias.detach()
            self.bias = nn.Parameter(folded_bias.detach().clone())

    @classmethod
    def from_sample_noise_5(cls, module):
        sample_noise_kwargs = getattr(module, 'sample_noise_kwargs', {})
        scale_factor = sample_noise_kwargs.get('scale_factor', DEFAULT_SAMPLE_NOISE_SCALE)
        fold_scale = torch.as_tensor(scale_factor, dtype=module.weight.dtype, device=module.weight.device)
        fold_scale = fold_scale * torch.as_tensor(
            module.sample_out_scale.detach(),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        weight_scale = _reshape_output_scale_for_weight(fold_scale, module.weight)
        folded_weight = module.weight.detach() * weight_scale
        folded_bias = None
        if module.bias is not None:
            bias_scale = _reshape_output_scale_for_bias(fold_scale, module.bias)
            folded_bias = module.bias.detach() * bias_scale
        return cls(
            module,
            a_quantizer=module.a_quantizer,
            int_flag=getattr(module.a_quantizer, 'int_flag', True),
            folded_weight=folded_weight,
            folded_bias=folded_bias,
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]

        input_q = self.a_quantizer(input)
        input_int = input_q[0] if isinstance(input_q, tuple) else input_q

        return F.conv2d(input_int, self.weight, self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


class linear_quant_noise(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 ):
        assert type(m) == nn.Linear
        super(linear_quant_noise, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
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
        x = F.linear(
            input_q,
            weight_q,
            self.bias,
        )
        return self.a_out_quantizer(x)


class linear_quant_noise_folded_weight(nn.Linear):
    """linear_quant_noise with static weight quantization folded into weight."""

    def __init__(self,
                 m: nn.Linear,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 folded_weight=None,
                 ):
        assert isinstance(m, nn.Linear), f"Expected nn.Linear or subclass, got {type(m)}"
        super(linear_quant_noise_folded_weight, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.a_quantizer = a_quantizer if a_quantizer is not None else NoQuan()
        self.a_out_quantizer = a_out_quantizer if a_out_quantizer is not None else NoQuan()
        if hasattr(self.a_out_quantizer, 'int_flag'):
            self.a_out_quantizer.int_flag = int_flag

        if folded_weight is None:
            folded_weight = m.weight.detach()
        self.weight = nn.Parameter(folded_weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())

    @classmethod
    def from_quant_noise(cls, module):
        folded_weight = _dequantize_static_weight(module.weight, module.w_quantizer)
        return cls(
            module,
            a_quantizer=module.a_quantizer,
            a_out_quantizer=module.a_out_quantizer,
            int_flag=getattr(module.a_out_quantizer, 'int_flag', False),
            folded_weight=folded_weight,
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        input_q = self.a_quantizer(input)
        x = F.linear(
            input_q,
            self.weight,
            self.bias,
        )
        return self.a_out_quantizer(x)


class linear_quant_noise_folded_weight_no_act(nn.Linear):
    """Folded-weight linear_quant_noise with input activation quantizer removed."""

    def __init__(self,
                 m: nn.Linear,
                 a_out_quantizer=None,
                 int_flag=False,
                 folded_weight=None,
                 ):
        assert isinstance(m, nn.Linear), f"Expected nn.Linear or subclass, got {type(m)}"
        super(linear_quant_noise_folded_weight_no_act, self).__init__(
                         m.in_features,
                         m.out_features,
                         bias=True if m.bias is not None else False,
                         )
        self.a_out_quantizer = a_out_quantizer if a_out_quantizer is not None else NoQuan()
        if hasattr(self.a_out_quantizer, 'int_flag'):
            self.a_out_quantizer.int_flag = int_flag

        if folded_weight is None:
            folded_weight = m.weight.detach()
        self.weight = nn.Parameter(folded_weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())

    @classmethod
    def from_folded_weight(cls, module):
        return cls(
            module,
            a_out_quantizer=module.a_out_quantizer,
            int_flag=getattr(module.a_out_quantizer, 'int_flag', False),
            folded_weight=module.weight.detach(),
        )

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        x = F.linear(
            input,
            self.weight,
            self.bias,
        )
        return self.a_out_quantizer(x)


class linear_quant_sample_noise(nn.Linear):
    def __init__(self,
                 m: nn.Linear,
                 w_quantizer=None,
                 a_quantizer=None,
                 a_out_quantizer=None,
                 int_flag=False,
                 sample_out_scale=0.,
                 sample_noise_mode='sample_noise_2',
                 sample_noise_kwargs=None,
                 init_weight_scale=False
                 ):
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
        if self.a_out_quantizer is not None:
            self.a_out_quantizer.int_flag = int_flag
        sample_out_scale = torch.as_tensor(sample_out_scale, dtype=self.weight.dtype, device=self.weight.device)
        self.sample_out_scale = nn.Parameter(sample_out_scale.detach().clone())
        self.sample_noise_mode = sample_noise_mode
        self.sample_noise_kwargs = _as_kwargs(sample_noise_kwargs)
        self.sample_noise_fn = _resolve_sample_noise_fn(sample_noise_mode)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

        if init_weight_scale and isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)

    def get_int_weight(self):
        weight_int, scale = self.w_quantizer.get_int(self.weight)
        return weight_int, scale

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]

        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)

        weight_int = weight_q[0]
        input_int = input_q[0]

        x = F.linear(input_int, weight_int, self.bias)
        y = self.sample_noise_fn(x, **self.sample_noise_kwargs)
        out = y * self.sample_out_scale
        return out


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad
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
    
QuanModuleMapping = {
    nn.Conv2d: conv2d_quant_noise,
    nn.Linear: linear_quant_noise
}
QuanModule = [
    conv2d_quant_noise,
    linear_quant_noise
]

ConvMapping = {nn.Conv2d: conv2d_quant_noise}
FcMapping = {nn.Linear: linear_quant_noise}
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
