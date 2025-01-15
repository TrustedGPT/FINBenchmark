# export MKL_THREADING_LAYER=GNU
# export MKL_SERVICE_FORCE_INTEL=1
# export OPENAI_API_KEY=''
# export OPENAI_API_BASE=http://0.0.0.0:8081/v1

echo "evaluating instruct ..."
python test.py --model_type api --model_path $1  --resume --out_name instruct.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/instruct_final_zh.json --eval instruct --prompt_type json --model_display_name  $1

echo "evaluating review ..."
python test.py --model_type api --model_path $1  --resume --out_name review.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/review_final_zh.json --eval review --prompt_type str --model_display_name  $1

echo "evaluating plan ..."
python test.py --model_type api --model_path $1  --resume --out_name plan.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/plan_final_zh.json --eval plan --prompt_type json --model_display_name  $1

echo "evaluating reason ..."
python test.py --model_type api --model_path $1  --resume --out_name resaon.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/reason_final_zh.json --eval reason --prompt_type str --model_display_name  $1

echo "evaluating retrieve ..."
python test.py --model_type api --model_path $1  --resume --out_name retrieve.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/retrieve_final_zh.json --eval retrieve --prompt_type str --model_display_name  $1

echo "evaluating understand ..."
python test.py --model_type api --model_path $1  --resume --out_name understand.json --out_dir ../$2/teval_output/ --dataset_path data1/agent_final数据/understand_final_zh.json --eval understand --prompt_type str --model_display_name  $1
