# MOE-CL form jxzhou

0 warmup stage 
only_for_bert.py: warmup,bert,mixeddata - > 1115-only-bert-formoe-stage0.pth
get_cluster_centers_after_warmup.py: get_2_clusters, bert, mixeddata -> 1115-layer_centers-2expert.pth

CUDA_VISIBLE_DEVICES=2 nohup python -u only_for_bert.py > 1115-only-bert-formoe-stage0.log
python get_cluster_centers_after_warmup.py


1 pretrain stage
moe-stage1.py: moe-stage1, vermilion_model, 2_clusters, Mixdata_1115 ->1115_vermilion_model_satge1.pth
only_for_bert.py: bert-stage1,bert_warmup,Mixdata_1115 - > 1115-only-bert-formoe-stage1.pth
get_replay_for_moe.py: vermilion_model_stage1, 2_clusters, Mixdata_1115 -> 1115-replay_data_vermilion_models-2experts.pth(batch_size == 1)
moe-router-stage1.py: moe-router-stage1, vermilion_model, 2_clusters, Mixdata_1115 ->1115_vermilion_model_satge1.pth


CUDA_VISIBLE_DEVICES=1 nohup python -u moe-stage1.py > 1115_vermilion_model_satge1.log
CUDA_VISIBLE_DEVICES=2 nohup python -u only_for_bert.py > 1115-only-bert-formoe-stage1.log
CUDA_VISIBLE_DEVICES=1 nohup python -u moe-router-stage1.py > 1120_rose_model_satge1.log
python get_replay_for_moe.py


2 incremental pretrain stage
moe-stage2.py: moe-stage2, vermilion_model, 2_clusters, ACLForLM_1103,replay_data ->1115_vermilion_model_satge2.pth
only_for_bert.py: bert-stage2,bert_stage1,ACLForLM_1103 - > 1115-only-bert-formoe-stage2.pth


CUDA_VISIBLE_DEVICES=1 nohup python -u moe-stage2.py > 1115_vermilion_model_satge2.log
CUDA_VISIBLE_DEVICES=2 nohup python -u only_for_bert.py > 1115-only-bert-formoe-stage2.log

