export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup torchrun --nproc_per_node=8 chexfound/train/train.py \
--config-file chexfound/configs/train/vitl16_ibot333_highres336.yaml \
--output-dir /outputs/chexfound/ibot333_highres336 \
train.dataset_path=CXRDatabase:split=TRAIN:root="/path/to/<ROOT>":extra="/path/to/<EXTRA>" \
&> /outputs/chexfound/ibot333_highres336.log &