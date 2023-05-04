def get_backbone_info(backbone):
    info = {
        "resnet18": {"n_output_channels": 512, "downsample_rate": 4},
        "resnet34": {"n_output_channels": 512, "downsample_rate": 4},
        "resnet50": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnet50_adf_dropout": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnet50_dropout": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnet101": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnet152": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnext50_32x4d": {"n_output_channels": 2048, "downsample_rate": 4},
        "resnext101_32x8d": {"n_output_channels": 2048, "downsample_rate": 4},
        "wide_resnet50_2": {"n_output_channels": 2048, "downsample_rate": 4},
        "wide_resnet101_2": {"n_output_channels": 2048, "downsample_rate": 4},
        "mobilenet_v2": {"n_output_channels": 1280, "downsample_rate": 4},
        "hrnet_w32": {"n_output_channels": 480, "downsample_rate": 4},
        "hrnet_w48": {"n_output_channels": 720, "downsample_rate": 4},
        # 'hrnet_w64': {'n_output_channels': 2048, 'downsample_rate': 4},
        "dla34": {"n_output_channels": 512, "downsample_rate": 4},
    }
    return info[backbone]
