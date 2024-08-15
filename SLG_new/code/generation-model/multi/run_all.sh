CUDA_VISIBLE_DEVICES=0 python __main__.py train Configs/Base_asl_csl.yaml & 
CUDA_VISIBLE_DEVICES=2 python __main__.py train Configs/Base_csl_asl.yaml &
CUDA_VISIBLE_DEVICES=3 python __main__.py train Configs/Base_asl_dgs.yaml &
CUDA_VISIBLE_DEVICES=4 python __main__.py train Configs/Base_dgs_asl.yaml &
CUDA_VISIBLE_DEVICES=5 python __main__.py train Configs/Base_dgs_csl.yaml &
CUDA_VISIBLE_DEVICES=6 python __main__.py train Configs/Base_csl_dgs.yaml 
