source venv/bin/activate
python scripts/train.py --dataset kitti \
                       --datapath ./datasets/kitti2015 \
                       --trainlist ./datasets/kitti15_train.txt \
                       --testlist ./datasets/kitti15_val.txt \
                       --epochs 300 \
                       --lrepoch 200:10 \
                       --model gwcnet-g \
                       --log_dir runs \
                       --ckpt_dir checkpoints \
                       --pretrained pretrained/best.ckpt \
                       --batch_size 1 \
                       --test_batch_size 1