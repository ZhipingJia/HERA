import munch
from .quant_layer import *
from .quant_util import *
import logging
logger = logging.getLogger()

def get_target_cfg(default_cfg, this_cfg=None): # 输入输出为munch.Munch
    target_cfg = munch.Munch(default_cfg)
    if this_cfg is None:
        return target_cfg
    for k in this_cfg:
        if target_cfg.get(k, None) is not None and type(this_cfg[k]) == munch.Munch:
            target_cfg[k] = get_target_cfg(default_cfg[k], this_cfg[k])
        else:
            target_cfg[k] = this_cfg[k]
    return target_cfg

def get_weight_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        q = NoQuan
    elif target_cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif target_cfg['quant_name'] == 'binary':
        q = Binary_weight_quantizer
    elif target_cfg['quant_name'] == 'lsq':
        q = LSQ_weight_quantizer
    elif target_cfg['quant_name'] == 'lsq_1':
        q = LSQ_weight_quantizer_1
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])

    target_cfg.pop('quant_name')
    return q(**target_cfg)
def get_bias_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        q = NoQuan
    elif target_cfg['quant_name'] == 'fixed_scale':
        q = Bias_quantizer_rows
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])

    target_cfg.pop('quant_name')
    return q(**target_cfg)
def get_act_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        #print('NoQuan!!!!!')
        q = NoQuan
    elif target_cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif target_cfg['quant_name'] == 'binary':
        return Binary_act_quantizer()
    elif target_cfg['quant_name'] == 'binary_std':
        q = Binary_act_quantizer_std
    elif target_cfg['quant_name'] == 'binary_th':
        q = Binary_act_quantizer_th
    elif target_cfg['quant_name'] == 'lsq':
        q = LSQ_act_quantizer
    elif target_cfg['quant_name'] == 'shift3bit':
        q = Shift3bit_act_quantizer
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])
        
    target_cfg.pop('quant_name')
    return q(**target_cfg)

def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
                target_cfg = get_target_cfg(quan_scheduler, quan_scheduler.excepts[name])
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(target_cfg.weight),
                    a_quantizer=get_act_quantizer(target_cfg.act),
                    a_out_quantizer=get_act_quantizer(target_cfg.act_out)
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(quan_scheduler.weight),
                    a_quantizer=get_act_quantizer(quan_scheduler.act),
                    a_out_quantizer=get_act_quantizer(quan_scheduler.act_out)
                )
        elif quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name)) # 模块替换操作
                        break
            else:
                helper(c)

    helper(model)
    return model
def find_modules_to_quantize2(model, quan_args):
    replaced_modules = dict()
    if (not 'conv' in quan_args) or (not 'fc' in quan_args):
        raise ValueError("can not find 'conv' or 'fc' key for quan_args. ")
    mapping_name = ['avgpool', 'bn', 'other']
    for m_n in mapping_name:
        if not m_n in quan_args:
            logger.warning("can not find '{}' key for qan_args, it will use default setting as "
                          "quant_flag: False. ".format(m_n))
    for name, module in model.named_modules():
        if not type(module) in totalMappingModule:
            continue
        if type(module) in QuanModuleMapping:
            layer_type = 'conv' if isinstance(module, nn.Conv2d) else 'fc'
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args[layer_type], quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args[layer_type])
            quant_module = QuanModuleMapping[type(module)](
                module,
                w_quantizer=get_weight_quantizer(
                    target_cfg.weight if 'weight' in target_cfg else {}),
                a_quantizer=get_act_quantizer(target_cfg.act if 'act' in target_cfg else {}),
                a_out_quantizer=get_act_quantizer(
                    target_cfg.act_out if 'act_out' in target_cfg else {}),
                int_flag = target_cfg.int_flag if 'int_flag' in target_cfg else False
            )
            replaced_modules[name] = quant_module
            logger.info("{} layer {} switched to quant_{} with args {}. ".format(layer_type,
                                                                                 name, layer_type, target_cfg))
        elif type(module) in BnMapping:
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args.bn if 'bn' in quan_args else munch.Munch(),
                                            quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args.bn) if 'bn' in quan_args else {}
            logger.info("BN layer {} switched to quant_BN with args {}. ".format(name, target_cfg))
            if 'quant_flag' in target_cfg and target_cfg.quant_flag:
                assert target_cfg.get('weight', None) is not None and \
                       target_cfg.weight.get('quant_name', None) is not None
                assert target_cfg.get('bias', None) is not None and \
                       target_cfg.bias.get('quant_name', None) is not None
                assert target_cfg.get('act_out', None) is not None and \
                       target_cfg.act_out.get('quant_name', None) is not None
                replaced_modules[name] = BnMapping[type(module)](m=module,
                                                                 w_quantizer=get_weight_quantizer(target_cfg.pop('weight')),
                                                                 bias_quantizer=get_bias_quantizer(target_cfg.pop('bias')),
                                                                 a_out_quantizer=get_act_quantizer(target_cfg.pop('act_out')),
                                                                 **dict(target_cfg))
            else:
                replaced_modules[name] = BnMapping[type(module)](m=module)
        elif type(module) in AvgMapping:
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args.avgpool if 'avgpool' in quan_args else munch.Munch(),
                                            quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args.avgpool) if 'avgpool' in quan_args else {}
            replaced_modules[name] = AvgMapping[type(module)](m=module, **dict(target_cfg))
            logger.info("Avg layer {} switched to quant_Avg with args {}. ".format(name, target_cfg))
        else:  # other module
            replaced_modules[name] = OtherMapping[type(module)](m=module)

    return replaced_modules

def find_modules_to_quantize3(model, quan_args):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if 'lora_mobile' in name:  # 仅选择 lora_mobile 层
            target_cfg = munch.Munch(quan_args['lora_mobile'])  # 根据配置文件设置
            quant_module = QuanModuleMapping[type(module)](
                module,
                w_quantizer=get_weight_quantizer(
                    target_cfg.weight if 'weight' in target_cfg else {}),
                a_quantizer=get_act_quantizer(target_cfg.act if 'act' in target_cfg else {}),
                a_out_quantizer=get_act_quantizer(
                    target_cfg.act_out if 'act_out' in target_cfg else {}),
                int_flag=target_cfg.int_flag if 'int_flag' in target_cfg else False
            )
            replaced_modules[name] = quant_module
            logger.info(f"lora_mobile layer {name} switched to quantized with args {target_cfg}.")

    return replaced_modules

def find_modules_to_quantize4(model, quan_args):
    # 从 quan_args 中提取特殊层配置和量化配置
    special_layers = quan_args.special_layers
    quant_config = quan_args.quant_config

    print("\n=== Special Layers ===")
    print(special_layers)
    print("\n=== Quantization Config ===")
    import pprint
    pprint.pprint(quant_config)
    print("===========================\n")

    replaced_modules = dict()
    for name, module in model.named_modules():
        if 'lora_mobile' in name:  # 针对 lora_mobile 层
            if name in special_layers:  # 针对特殊层
                # 根据 SAMPLE_FLAG 决定使用哪种映射
                quant_module_type = linear_quant_noise
                quant_module = quant_module_type(
                    module,
                    w_quantizer=get_weight_quantizer(quant_config["weight"]),
                    a_quantizer=get_act_quantizer(quant_config["act"]),
                    a_out_quantizer=None,  # 可根据需求补充 act_out 的量化器
                    int_flag=quant_config["int_flag"],
                )
                replaced_modules[name] = quant_module
                print(f"Special layer {name} switched to quantized with config from args {quant_config}.")
            else:  # 默认层配置
                target_cfg = munch.Munch(quan_args['lora_mobile'])  # 使用全局配置
                quant_module = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(
                        target_cfg.weight if 'weight' in target_cfg else {}),
                    a_quantizer=get_act_quantizer(target_cfg.act if 'act' in target_cfg else {}),
                    a_out_quantizer=get_act_quantizer(
                        target_cfg.act_out if 'act_out' in target_cfg else {}),
                    int_flag=target_cfg.int_flag if 'int_flag' in target_cfg else False
                )
                replaced_modules[name] = quant_module
                print(f"Default lora_mobile layer {name} switched to quantized with args {target_cfg}.")

    return replaced_modules

def find_modules_to_quantize5(model, quan_args):
    # 提取特殊层配置
    special_layers = quan_args.special_layers
    quant_config = quan_args.quant_config
    lora_lm_head_config = quan_args.lora_lm_head_B  # 提取方式一致

    print("\n=== Special Layers ===")
    print(special_layers)
    print("\n=== Quantization Config ===")
    import pprint
    pprint.pprint(quant_config)
    print("\n=== Lora LM Head Config ===")
    pprint.pprint(lora_lm_head_config)
    print("===========================\n")

    replaced_modules = dict()
    for name, module in model.named_modules():
        print('find name:',name)
        if 'lora_mobile' in name:  # 针对 lora_mobile 层
            if name in special_layers:  # 处理特殊层
                quant_module_type = linear_quant_noise
                quant_module = quant_module_type(
                    module,
                    w_quantizer=get_weight_quantizer(quant_config["weight"]),
                    a_quantizer=get_act_quantizer(quant_config["act"]),
                    a_out_quantizer=None,  # 可根据需求补充 act_out 的量化器
                    int_flag=quant_config["int_flag"],
                )
                replaced_modules[name] = quant_module
                print(f"Special layer {name} switched to quantized with config from args {quant_config}.")
            else:  # 默认层配置
                target_cfg = munch.Munch(quan_args['lora_mobile'])
                quant_module = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(
                        target_cfg.weight if 'weight' in target_cfg else {}),
                    a_quantizer=get_act_quantizer(target_cfg.act if 'act' in target_cfg else {}),
                    a_out_quantizer=get_act_quantizer(
                        target_cfg.act_out if 'act_out' in target_cfg else {}),
                    int_flag=target_cfg.int_flag if 'int_flag' in target_cfg else False
                )
                replaced_modules[name] = quant_module
                print(f"Default lora_mobile layer {name} switched to quantized with args {target_cfg}.")
        elif name == lora_lm_head_config.layer_name:  # 处理 lora_lm_head_B.weight
            quant_module_type = linear_quant_noise
            quant_module = quant_module_type(
                module,
                w_quantizer=get_weight_quantizer(lora_lm_head_config.weight),
                a_quantizer=get_act_quantizer(lora_lm_head_config.act),
                a_out_quantizer=None,  # 输出不量化
                int_flag=lora_lm_head_config.int_flag,
            )
            replaced_modules[name] = quant_module
            print(f"Special layer {name} (lora_lm_head_B.weight) switched to quantized with config: {lora_lm_head_config}.")

    return replaced_modules

def find_modules_to_quantize_by_kl_divergence(model, quan_args):
    """
    遍历模型所有模块，根据配置文件区分处理：
      - 对于名称中包含 'lora_mobile' 的层：
          如果该层名称存在于 quan_args.special_layer_names 中，
          则使用 config_file_2 中的配置（存储在 quan_args.lora_mobile_config_2 下）；
          否则使用默认配置（quan_args.lora_mobile）。
      - 对于名称等于 quan_args.lora_lm_head_B.layer_name 的层，
          则使用 lm_head_file 中的配置（存储在 quan_args.lora_lm_head_B 下）。
    返回一个字典，键为模块完整名称，值为构造好的量化模块。
    """
    replaced_modules = {}
    for name, module in model.named_modules():
        if 'lora_mobile' in name:
            if hasattr(quan_args, "special_layer_names") and name in quan_args.special_layer_names:
                # 使用 config_file_2 中的 lora_mobile 配置
                target_cfg = munch.Munch(quan_args.lora_mobile_config_2)
                quant_type = "SPECIAL"
            else:
                # 使用默认配置
                target_cfg = munch.Munch(quan_args.lora_mobile)
                quant_type = "DEFAULT"
            if quant_type == "DEFAULT":
                quant_module = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(
                        target_cfg.weight
                    ) if 'weight' in target_cfg else None,
                    a_quantizer=get_act_quantizer(
                        target_cfg.act
                    ) if 'act' in target_cfg else None,
                    a_out_quantizer=get_act_quantizer(
                        target_cfg.act_out
                    ) if 'act_out' in target_cfg else None,
                    int_flag=target_cfg.get('int_flag', False)
                )
                replaced_modules[name] = quant_module
                print(f"{quant_type} lora_mobile layer '{name}' quantized using its respective configuration.")
            elif quant_type == "SPECIAL":
                quant_module = linear_quant_noise(
                    module,
                    w_quantizer=get_weight_quantizer(
                        target_cfg.weight
                    ) if 'weight' in target_cfg else None,
                    a_quantizer=get_act_quantizer(
                        target_cfg.act
                    ) if 'act' in target_cfg else None,
                    a_out_quantizer=get_act_quantizer(
                        target_cfg.act_out
                    ) if 'act_out' in target_cfg else None,
                    int_flag=target_cfg.get('int_flag', False)
                )
                replaced_modules[name] = quant_module
                print(f"{quant_type} lora_mobile layer '{name}' quantized using its respective configuration.")
        elif hasattr(quan_args, "lora_lm_head_B") and name == quan_args.lora_lm_head_B.layer_name:
            target_cfg = munch.Munch(quan_args.lora_lm_head_B)
            quant_module = linear_quant_noise(
                module,
                w_quantizer=get_weight_quantizer(target_cfg.weight),
                a_quantizer=get_act_quantizer(target_cfg.act),
                a_out_quantizer=None,  # 输出层不量化
                int_flag=target_cfg.int_flag,
            )
            replaced_modules[name] = quant_module
            print(f"lora_lm_head_B layer '{name}' quantized using lm_head configuration.")
    return replaced_modules

def find_modules_to_quantize_by_kl_divergence_c160_onchip(model, quan_args):
    """
    遍历模型所有模块，根据 quan_args.lora_mobile_layers_mode 中各层的模式配置进行量化：
      - 对于名称中包含 'lora_mobile' 的层：
          * 如果该层模式为 "CGRA"，则使用 quan_args.lora_mobile_config_2 中的配置，并采用 linear_quant_noise。
          * 如果该层模式为 "RRAM_SIM"，则使用 quan_args.lora_mobile 中的配置，并采用 linear_quant_sample_noise。
          * 如果该层模式为 "RRAM_ONCHIP"，则使用 quan_args.lora_mobile 中的配置，并采用 linear_quant_sample_noise_c160_onchip。
      - 对于名称等于 quan_args.lora_lm_head_B.layer_name 的层，
          则使用 lm_head 配置（采用 linear_quant_noise）。
    返回一个字典，键为模块完整名称，值为构造好的量化模块。
    """
    replaced_modules = {}
    # 获取每个 lora_mobile 层的模式字典
    layers_mode_dict = quan_args.lora_mobile_layers_mode

    for name, module in model.named_modules():
        if 'lora_mobile' in name:
            if name not in layers_mode_dict:
                raise KeyError(f"Layer name {name} not found in lora_mobile_layers_mode configuration")
            mode = layers_mode_dict[name]
            if mode == "CGRA":
                # 使用 quan_args.lora_mobile_config_2 配置，并采用 linear_quant_noise 进行量化
                target_cfg = munch.Munch(quan_args.lora_mobile_config_2)
                quant_module = linear_quant_noise(
                    module,
                    w_quantizer=get_weight_quantizer(target_cfg.weight) if 'weight' in target_cfg else None,
                    a_quantizer=get_act_quantizer(target_cfg.act) if 'act' in target_cfg else None,
                    a_out_quantizer=get_act_quantizer(target_cfg.act_out) if 'act_out' in target_cfg else None,
                    int_flag=target_cfg.get('int_flag', False)
                )
                replaced_modules[name] = quant_module
                print(f"CGRA lora_mobile layer '{name}' quantized using lora_mobile_config_2 configuration.")
                # pass
            elif mode == "RRAM_SIM":
                # 使用 quan_args.lora_mobile 配置，并采用 linear_quant_sample_noise 进行量化
                target_cfg = munch.Munch(quan_args.lora_mobile)
                quant_module = linear_quant_sample_noise(
                    module,
                    w_quantizer=get_weight_quantizer(target_cfg.weight) if 'weight' in target_cfg else None,
                    a_quantizer=get_act_quantizer(target_cfg.act) if 'act' in target_cfg else None,
                    a_out_quantizer=get_act_quantizer(target_cfg.act_out) if 'act_out' in target_cfg else None,
                    int_flag=target_cfg.get('int_flag', False),
                    sample_noise_mode=target_cfg.get('sample_noise_mode', 'sample_noise_2'),
                    sample_noise_kwargs=target_cfg.get('sample_noise_kwargs', None)
                )
                # quant_module = linear_debug(
                #     module
                # )
                replaced_modules[name] = quant_module
                print(f"RRAM_SIM lora_mobile layer '{name}' quantized using lora_mobile configuration with linear_quant_sample_noise.")
                pass
            elif mode == "RRAM_ONCHIP":
                # 使用 quan_args.lora_mobile 配置，并采用 linear_quant_sample_noise_c160_onchip 进行量化
                target_cfg = munch.Munch(quan_args.lora_mobile)
                quant_module = linear_quant_sample_noise_c160_onchip(
                    module,
                    w_quantizer=get_weight_quantizer(target_cfg.weight) if 'weight' in target_cfg else None,
                    a_quantizer=get_act_quantizer(target_cfg.act) if 'act' in target_cfg else None,
                    a_out_quantizer=get_act_quantizer(target_cfg.act_out) if 'act_out' in target_cfg else None,
                    int_flag=target_cfg.get('int_flag', False),
                    layer_name=name
                )
                print(f"RRAM_ONCHIP lora_mobile layer '{name}' quantized using lora_mobile configuration with linear_quant_sample_noise_c160_onchip.")
                replaced_modules[name] = quant_module
            else:
                raise ValueError(f"未知的层模式 {mode} for layer {name}")
            

        elif hasattr(quan_args, "lora_lm_head_B") and name == quan_args.lora_lm_head_B.layer_name:
            # pass
            target_cfg = munch.Munch(quan_args.lora_lm_head_B)
            quant_module = linear_quant_noise(
                module,
                w_quantizer=get_weight_quantizer(target_cfg.weight),
                a_quantizer=get_act_quantizer(target_cfg.act),
                a_out_quantizer=None,
                int_flag=target_cfg.int_flag,
            )
            replaced_modules[name] = quant_module
            print(f"lora_lm_head_B layer '{name}' quantized using lm_head configuration.")

    return replaced_modules


def find_modules_to_quantize_unified_cgra(model, quan_args):
    """
    遍历模型：
      1. 所有的 'lora_mobile' 层 -> 统一使用 quan_args.lora_mobile_config_2 (CGRA配置) + linear_quant_noise
      2. 特殊的 'lora_lm_head_B' 层 -> 使用 lm_head 配置 + linear_quant_noise
    """
    replaced_modules = {}

    for name, module in model.named_modules():
        
        # === 情况 1: 所有的 lora_mobile 层 (全部视为 CGRA) ===
        if 'lora_mobile' in name:
            # 直接使用 lora_mobile_config_2 (即 opt.quant_file_2 里的配置)
            target_cfg = munch.Munch(quan_args.lora_mobile_config_2)
            
            # 统一使用 linear_quant_noise 类 (CGRA 层的类)
            # 注意：CGRA 层只有 w_quantizer 和 a_quantizer，没有 sample_out_scale 相关的逻辑
            quant_module = linear_quant_noise(
                module,
                w_quantizer=get_weight_quantizer(target_cfg.weight) if 'weight' in target_cfg else None,
                a_quantizer=get_act_quantizer(target_cfg.act) if 'act' in target_cfg else None,
                a_out_quantizer=None,
                int_flag=target_cfg.get('int_flag', False)
            )
            replaced_modules[name] = quant_module
            # print(f"Unified CGRA: layer '{name}' quantized using config_2.")

        # === 情况 2: 特殊的 LM Head 层 ===
        elif hasattr(quan_args, "lora_lm_head_B") and name == quan_args.lora_lm_head_B.layer_name:
            target_cfg = munch.Munch(quan_args.lora_lm_head_B)
            
            quant_module = linear_quant_noise(
                module,
                w_quantizer=get_weight_quantizer(target_cfg.weight),
                a_quantizer=get_act_quantizer(target_cfg.act),
                a_out_quantizer=None,
                int_flag=target_cfg.int_flag,
            )
            replaced_modules[name] = quant_module
            print(f"Unified CGRA: lm_head layer '{name}' quantized using lm_head configuration.")

    return replaced_modules


def replace_module_by_names2(model, modules_to_replace):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in totalMappingModule:
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name)) # 模块替换操作
                        break
            else:
                helper(c)

    helper(model)
    return model

def replace_module_by_names3(model, modules_to_replace):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in totalMappingModule:
                for full_name, m in model.named_modules():
                    if 'lora_mobile' in full_name and c is m:
                        if full_name in modules_to_replace:  # 确保层在待替换模块列表中
                            child.add_module(n, modules_to_replace.pop(full_name))  # 模块替换操作
                            print(f"Replaced module: {full_name}")
                        break
            else:
                helper(c)

    helper(model)
    return model

def replace_module_by_names5(model, modules_to_replace):
    def helper(child: nn.Module, prefix: str = ""):
        # 遍历子模块
        for n, c in child.named_children():
            if type(c) in totalMappingModule:
                for full_name, m in model.named_modules():
                    if c is m:
                        if full_name in modules_to_replace:  # 确保层在待替换模块列表中
                            child.add_module(n, modules_to_replace.pop(full_name))  # 模块替换操作
                            print(f"Replaced module: {full_name}")
                        break
            else:
                helper(c)

    # 从根模块开始递归替换
    helper(model)
    return model


def prepare_quant_model(
    model,
    train_loader,
    quan_args,
    ):
    # model: float32 model to be quantized
    # train_loader: train_data may be used to act initial
    # quan_args: quantizer args
    modules_to_replace = find_modules_to_quantize(model, quan_args)
    model = replace_module_by_names(model, modules_to_replace)
    if quan_args.init_batch:
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print(name)
        for batch_idx, (inputs, targets) in enumerate(train_loader): ##
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
    return model

def prepare_quant_model2(
    model,
    train_loader,
    quan_args,
    ):
    # model: float32 model to be quantized
    # train_loader: train_data may be used to act initial
    # quan_args: quantizer args
    modules_to_replace = find_modules_to_quantize2(model, quan_args)
    model = replace_module_by_names2(model, modules_to_replace)
    # model.eval()
    if quan_args.init_batch:
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print("Quantizer {} set init_batch_mode True. ".format(name))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
                print("Quantizer {} set init_batch_mode False. ".format(name))
    return model


def prepare_quant_model3(
    model,
    quan_args,
    ):
    modules_to_replace = find_modules_to_quantize3(model, quan_args)
    print('modules_to_replace:',modules_to_replace)
    model = replace_module_by_names3(model, modules_to_replace)
    return model

def prepare_quant_model4(
    model,
    quan_args,
    ):
    modules_to_replace = find_modules_to_quantize4(model, quan_args)
    print('modules_to_replace:',modules_to_replace)
    model = replace_module_by_names3(model, modules_to_replace)
    return model

def prepare_quant_model5(
    model,
    quan_args,
    ):
    modules_to_replace = find_modules_to_quantize5(model, quan_args)
    print('modules_to_replace:',modules_to_replace)
    model = replace_module_by_names5(model, modules_to_replace)
    return model

def prepare_quant_model_by_layer_name(model, quan_args, target_layer_name):
    """
    根据指定的层名称精确量化模型中特定的单个层
     
    :param model: 原始全精度模型
    :param quan_args: 量化相关参数配置
    :param target_layer_name: 目标量化层的完整名称
    :return: 量化后的模型
    """
    replaced_modules = {}

    # 定位并替换指定层
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_cfg = munch.Munch(quan_args['quan']['lora_mobile'])  # 根据配置文件设置
            quant_module = QuanModuleMapping[type(module)](
                module,
                w_quantizer=get_weight_quantizer(
                    target_cfg.weight if 'weight' in target_cfg else {}
                ),
                a_quantizer=get_act_quantizer(
                    target_cfg.act if 'act' in target_cfg else {}
                ),
                a_out_quantizer=get_act_quantizer(
                    target_cfg.act_out
                ) if 'act_out' in target_cfg else None,
                int_flag=target_cfg.int_flag if 'int_flag' in target_cfg else False
            )

            replaced_modules[name] = quant_module
            print(f"Layer {name} switched to quantized with args {target_cfg}.")
    
    print('replaced_modules:')
    print(replaced_modules)

    replace_module_by_names3(model, replaced_modules)

    print(f"Quantized model with layer {name} successfully created.")
    return model

def prepare_quant_model_by_kl_divergence(model, quan_args):
    """
    根据配置文件对模型进行量化：
      1. 调用 find_modules_to_quantize_by_kl_divergence 得到待替换模块字典；
      2. 调用 replace_module_by_names5 递归替换模型中的模块。
    """
    modules_to_replace = find_modules_to_quantize_by_kl_divergence(model, quan_args)
    print('modules_to_replace:',modules_to_replace)
    model = replace_module_by_names5(model, modules_to_replace)
    return model

def prepare_quant_model_by_kl_divergence_c160_onchip(model, quan_args):
    """
    根据配置文件对模型进行量化：
      1. 调用 find_modules_to_quantize_by_kl_divergence 得到待替换模块字典；
      2. 调用 replace_module_by_names5 递归替换模型中的模块。
    """
    modules_to_replace = find_modules_to_quantize_by_kl_divergence_c160_onchip(model, quan_args)
    print('modules_to_replace:',modules_to_replace)
    model = replace_module_by_names5(model, modules_to_replace)
    return model

def prepare_quant_model_unified_cgra(model, quan_args):
    """
    统一将模型量化为 CGRA 模式 + 特殊 LM Head。
    """
    # 调用新的查找函数
    modules_to_replace = find_modules_to_quantize_unified_cgra(model, quan_args)
    print('modules_to_replace:', modules_to_replace.keys()) # 打印 key 预览即可
    
    # 替换逻辑通常是通用的，直接调用即可 (假设 replace_module_by_names5 在外部定义)
    model = replace_module_by_names5(model, modules_to_replace)
    return model


# def prepare_quant_model(
#     model,
#     quant_dict,
#     ):
#     # model: float32 model to be quantized
#     # quant_dict: define the layers to be quantized and noised, {'layer_name': {'w_quant_way': {}, 'act_quant_way': {}, 'w_noise_way': {}}}}
#     for name, layer_module in model.named_modules():
#         if name in quant_dict:
#             if isinstance(layer_module, nn.Conv2d):
#                 quant_conv = conv2d_quant_noise(
#                     layer_module,
#                     w_quant_way=quant_dict[name]['w_quant_way'],
#                     a_quant_way=quant_dict[name]['a_quant_way'],
#                 )
#                 model._modules[name] = quant_conv
#             elif isinstance(layer_module, nn.Linear):
#                 quant_linear = linear_quant_noise(
#                     layer_module,
#                     w_quant_way=quant_dict[name]['w_quant_way'],
#                     a_quant_way=quant_dict[name]['a_quant_way'],
#                 )
#                 model._modules[name] = quant_linear
#             else:
#                 raise ValueError("layer {} can not be quantizes yet! ".format(name))
#     return model
