

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
# output_name="LSTP-Blip2-flant5-xl-ivt"
output_name="LSTP-Instructblip-Vicuna"
pred_path="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/gpt3-0.25"
output_json="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/results.json"
api_key="xxx"
api_base="xxx"
num_tasks=8



python -m eval.evaluate \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}