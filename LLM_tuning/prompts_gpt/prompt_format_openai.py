import json

input_file = "preprocessing/participant_prompts.jsonl"
output_file = "participant_prompts_openai_format.jsonl"

system_msg = "Du bist ein hilfreicher Assistent, der klassifiziert, ob eine Person ihre Krankenversicherung wechseln wird oder nicht."

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": data["prompt"]},
            {"role": "assistant", "content": data["completion"]}
        ]
        json.dump({"messages": messages}, outfile, ensure_ascii=False)
        outfile.write('\n')
