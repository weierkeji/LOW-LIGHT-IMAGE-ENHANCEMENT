import argparse

from utils.parser import create_model, define_dataloader, define_network, define_dataset, parse
from utils.reproducibility import set_seed_and_cudnn


def main(config):
    set_seed_and_cudnn()

    phase = config['phase']
    dataset = define_dataset(config[phase]['dataset'])
    dataloader = define_dataloader(dataset, config[phase]['dataloader']['args'])
    
    # 根据参数选择使用哪种网络模型
    if config.get('use_enhanced_model', False):
        print("Using Enhanced CDAN model with improved architecture for better PSNR...")
        from models.cdan import EnhancedCDAN
        network = EnhancedCDAN()
    else:
        print("Using original CDAN model...")
        network = define_network(config['model']['networks'][0])

    model = create_model(config=config,
                         network=network,
                         dataloader=dataloader
                        )

    if phase == 'train':
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.json', help='Path to the JSON configuration file')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Phase to run (train or test)', default='train')
    parser.add_argument('-e', '--enhanced', action='store_true', help='Use enhanced CDAN model for better PSNR')

    # parser configs
    args = parser.parse_args()
    config = parse(args)
    
    # 添加使用增强模型的标志到配置中
    config['use_enhanced_model'] = args.enhanced

    main(config)