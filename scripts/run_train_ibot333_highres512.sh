export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

nohup torchrun --nproc_per_node=1 chexfound/train/train.py \
--config-file chexfound/configs/train/vitl16_ibot333_highres512.yaml \
--output-dir ./outputs/chexfound/ibot333_highres512 \
train.dataset_path=CXRDatabase:split=TRAIN:root="./data":extra="./EXTRA" \
&> /outputs/chexfound/ibot333_highres512.log &

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 chexfound/train/train.py --config-file chexfound/configs/train/vitl16_ibot333_highres512.yaml --output-dir ./outputs/chexfound/ibot333_highres512 train.dataset_path=CXRDatabase:split=TRAIN:root=./data:extra=./EXTRA &> ./outputs/chexfound/ibot333_highres512.log &
