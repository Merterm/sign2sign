CUDA_VISIBLE_DEVICES=1 python -m signjoey test configs/eval_pipeline/sign_eval_csl_dgs.yaml --output_path pipeline_source_output/csl_dgs_src &
CUDA_VISIBLE_DEVICES=2 python -m signjoey test configs/eval_pipeline/sign_eval_csl_asl.yaml --output_path pipeline_source_output/csl_asl_src &
CUDA_VISIBLE_DEVICES=3 python -m signjoey test configs/eval_pipeline/sign_eval_asl_csl.yaml --output_path pipeline_source_output/asl_csl_src &
CUDA_VISIBLE_DEVICES=4 python -m signjoey test configs/eval_pipeline/sign_eval_asl_dgs.yaml --output_path pipeline_source_output/asl_dgs_src &
CUDA_VISIBLE_DEVICES=5 python -m signjoey test configs/eval_pipeline/sign_eval_dgs_csl.yaml --output_path pipeline_source_output/dgs_csl_src & 
CUDA_VISIBLE_DEVICES=6 python -m signjoey test configs/eval_pipeline/sign_eval_dgs_asl.yaml --output_path pipeline_source_output/dgs_asl_src
