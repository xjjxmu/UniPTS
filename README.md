# UniPTS: A Unified Framework for Proficient Post-Training Sparsity



## Requirements

- Python >= 3.7.4
- Pytorch >= 1.6.1
- Torchvision >= 0.4.1

## Reproduce the Experiment Results

Select a configuration file in `configs` to reproduce the experiment results reported in the paper. For example, to prune ResNet-50 on ImageNet dataset, run:

   `python main.py --config configs/resnet50.yaml --multigpu 0 --checkpoint /path/to/resnet50-0676ba61.pth`

   Note that the `data` and `prune_rate` in the yaml file should be changed to the data path and your target sparse rate. And we point out the pretrained dense model used in UniPTS.


Any problem, feel free to contact jingjingxie@stu.xmu.edu.cn