bash train.sh -d ./data/wikitext-2-raw-v1

bash ./scripts/run_merge_dora.sh \
  --model_name_or_path data4elm/Llama-400M-12L \
  --lora_model_path output_models/finetune \
  --output_model_path output_models/finetune_merged

cd lm-evaluation-harness

lm_eval --model hf \
  --model_args pretrained=../output_models/finetune_merged,trust_remote_code=True,cache_dir=~/.cache \
  --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
  --device cuda:0 \
  --batch_size 1 \
  --log_samples \
  --output_path ./eval_results/test_elmb

cd ..