from speakerlab.models.campplus.DTDNN import CAMPPlus
import torch.onnx
import yaml

if __name__ == "__main__":
    
    # 加载配置文件
    with open("./models/speech_campplus_sv_zh-cn_16k-common/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 使用配置文件中的参数创建模型
    model_conf = config["model_conf"]
    model = CAMPPlus(
        feat_dim=model_conf["feat_dim"],
        embedding_size=model_conf["embedding_size"],
        growth_rate=model_conf["growth_rate"],
        bn_size=model_conf["bn_size"],
        init_channels=model_conf["init_channels"],
        config_str=model_conf["config_str"],
        memory_efficient=model_conf["memory_efficient"]
    )
    
    state_dict_path = "./models/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    dummpy_input = torch.rand(1, 1000, 80)
    
    # 直接导出 ONNX 模型，不使用 JIT 编译
    model.eval()
    torch.onnx.export(
        model,
        dummpy_input,
        './speech_campplus_sv_zh-cn_16k-common.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    
    