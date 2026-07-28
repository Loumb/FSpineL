export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup torchrun --nproc_per_node=4 chexfound/eval/classification/linear_glori.py \
--batch-size 64 \
--val-epochs 100 \
--epochs 100 \
--image-size 512 \
--save-checkpoint-frequency 20 \
--eval-period-epochs 20 \
--val-metric-type binary_auc \
--config-file /outputs/chexfound/ibot333_highres512/config.yaml \
--pretrained-weights /outputs/chexfound/ibot333_highres512/eval/training_249999/teacher_checkpoint.pth \
--output-dir /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen \
--train-dataset Shenzhen:split=TRAIN:root=/eval/shenzhen \
--val-dataset Shenzhen:split=VAL:root=/eval/shenzhen \
--test-dataset Shenzhen:split=TEST:root=/eval/shenzhen \
 &> /outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen.log &