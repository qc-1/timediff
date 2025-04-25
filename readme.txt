
Environment:
conda create -n timediff python=3.10
conda activate timediff

Data:
/root/autodl-tmp/datasets/prediction/data_autoformer/ETT-small/ETTh1.csv

Epoch: 99 cost time: 4.252164363861084
Epoch: 99, Steps: 109 | Train Loss: 0.0254163 Val Loss: 0.4039532
        iters: 100, epoch: 100 | loss: 0.0255536
        speed: 0.1106s/iter; left time: 1.1057s
Epoch: 100 cost time: 4.85784912109375
Epoch: 100, Steps: 109 | Train Loss: 0.0251986 Val Loss: 0.4076888
>>>>>>>testing : ETTh1_1440_168_DDPM_ETTh1_ftM_sl1440_ll336_pl168_dt0_TWO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
> test-shape (4320, 7) (4320, 7)
test 2713
Successfully loading trained model!
Elapsed time: 38.29 ms
Elapsed time: 40.90 ms
Elapsed time: 23.75 ms
Elapsed time: 25.37 ms
Elapsed time: 36.69 ms
id_worst 2
top5 [3 2 4 1 0]
mse|mae|rmse|mape|mspe|corr
0.41035238 0.4212738 0.6405875 9.32205 34339.9 25.59765

Parameter Introduction:
--is_training    1:训练    0：测试
--type_sampler none：diffusion step可视化    dmp:非可视化
--dataset_name   数据集名字
--target  预测的目标:target=MT_030 在 features=M 时，不会影响模型输出和mse等指标的计算，只是一个标记,模型输出所有变量，损失（如mse）是对所有变量整体计算的；如果你想只预测 MT_030，需要设置 --features S 或 --features MS，这时 target 才会生效,模型的输出、损失（如mse）、评估等都只针对 MT_030 这一列
--num_vars  数据集的变量数


ETTh1
CUDA_VISIBLE_DEVICES=2 python main_ddpm.py --pretrain_epochs 10 --train_epochs 100 --is_training 1 --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 --ablation_study_F_type Linear  --cond_ddpm_num_layers 30 --ddpm_layers_II 10 --learning_rate 0.0001 --label_len 336


ECL:
python main_ddpm.py --pretrain_epochs 10 --train_epochs 100 --is_training 1 --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 --ablation_study_F_type Linear  --cond_ddpm_num_layers 30 --ddpm_layers_II 10 --learning_rate 0.0001 --label_len 336 --dataset_name ECL --target MT_030 --num_vars 370


Exchange_rate
 python main_ddpm.py --pretrain_epochs 10 --train_epochs 100 --is_training 0 --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 --ablation_study_F_type Linear  --cond_ddpm_num_layers 30 --ddpm_layers_II 10 --learning_rate 0.0001 --label_len 336 --dataset_name Exchange --target Australia --num_vars 8 --type_sampler none --features M

weather: 33






