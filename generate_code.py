# coding=utf-8

import argparse
import csv
import json
import os
import random
import re
import time

import torch
from peft import PeftModel
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

# chat template
default_system = 'You are a helpful assistant. 你是一个乐于助人的助手。'
templates = {
    'default': {
        'prefix': '<|im_start|>system\n' + default_system +
        '<|im_end|>\n<|im_start|>user\n',
        'suffix': '<|im_end|>\n<|im_start|>assistant\n'
    },
    'llama1': {
        'prefix': 'Human: \n',
        'suffix': '\n\nAssistant: \n'
    },
    'llama2': {
        'prefix': '<s>[INST] <<SYS>>\n' + default_system + '\n<</SYS>>\n\n',
        'suffix': ' [/INST]'
    },
    'llama3': {
        'prefix':
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + default_system +
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n',
        'suffix':
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    },
    'baichuan': {
        'prefix': '<reserved_102>',
        'suffix': '<reserved_103>'
    },
    'Yi-34B': {
        'prefix': '',
        'suffix': ''
    }
}

categories = {
    'financial_knowledge': {
        'finance_single': '4-single',
        'finance_multiple': '4-multiple',
        'insurance_single': '4-single',
        'insurance_multiple': '4-multiple',
        'accounting_single': '4-single',
        'accounting_multiple': '4-multiple',
        'banking_single': '4-single',
        'banking_multiple': '4-multiple',
        'qualification_single': '4-single',
        'qualification_multiple': '4-multiple',
    },
    'financial_services': {
        'semantic_similarity': '2-single',
        'intent_recognition': '4-single',
        'business_compliance': '4-single1',#业务合规额外处理
        'action_recognition': 'list',
    },
    'financial_agency': {
        'text_matching': 'number',
        'distress_prediction': 'set',
    },
}

name_en2zh = {
    'financial_knowledge': '金融知识',
    'finance_single': '金融学-单选',
    'finance_multiple': '金融学-多选',
    'insurance_single': '保险学-单选',
    'insurance_multiple': '保险学-多选',
    'accounting_single': '会计学-单选',
    'accounting_multiple': '会计学-多选',
    'banking_single': '银行学-单选',
    'banking_multiple': '银行学-多选',
    'qualification_single': '金融从业资格-单选',
    'qualification_multiple': '金融从业资格-多选',
    'financial_services': '金融业务',
    'semantic_similarity': '文本相似',
    'intent_recognition': '意图识别',
    'business_compliance': '业务合规',
    'emotion_transfer': '情感迁移',
    'dialogue_summarization': '对话摘要',
    'question_answering': '业务问答',
    'action_recognition': '动作识别',
    'financial_agency': '金融智能体',
    'text_matching': '文段匹配',
    'distress_prediction': '风险评估',
}


class Runner:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_answer(self, prompt: str, generation_config={}):
        input_ids = self.tokenizer.encode(prompt,
                                          return_tensors='pt').to('cuda')
        try:
            output = self.model.generate(inputs=input_ids,
                                         return_dict_in_generate=True,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         generation_config=generation_config,
                                         num_return_sequences=1)
            output_text = self.tokenizer.decode(output.sequences[0],
                                                skip_special_tokens=True)
        except Exception:
            output_text, _ = self.model.chat(
                self.tokenizer, "请解释一下资产负债率", history=None)
        return output_text


    def run_answer_batch(self, prompts: list, generation_config={}):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer(
            prompts, return_tensors='pt', padding=True).to('cuda')

        # If the model is wrapped in DataParallel, access the underlying module
        model_to_use = self.model.module if isinstance(
            self.model, torch.nn.DataParallel) else self.model

        generated_ids = model_to_use.generate(
            **input_ids,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_config
        )
        outputs = self.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True)
        return outputs


class Runnervllm:

    def __init__(self, llm):
        if "Tongyi-Finance-14B" in llm:
            self.llm = LLM(model=llm,
                        tensor_parallel_size=len(args.gpus.split(',')),
                        trust_remote_code=True,
                        gpu_memory_utilization=0.80)
        else:    
            self.llm = LLM(model=llm,
                        tensor_parallel_size=len(args.gpus.split(',')),
                        gpu_memory_utilization=0.95)

    def run_answer(self, prompts: list, temperature=1.0):
        sampling_params = SamplingParams(temperature=temperature,
                                         presence_penalty=1.0,
                                         max_tokens=64)
        texts = []
        for prompt in prompts:
            message = [{
                'role': 'system',
                'content': default_system
            }, {
                'role': 'user',
                'content': prompt
            }]
            text = tokenizer.apply_chat_template(message,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            texts.append(text)
        outputs = self.llm.generate(texts, sampling_params, use_tqdm=False)
        return outputs


def gen_prompt(sample, task, cot=False, include_answer=False):
    if args.num_few_shot and not include_answer:
        prompt = instruction_example
    else:
        prompt = ''
    if eval_method.endswith('single') or eval_method.endswith('single1'):
        if cot:
            prompt += f'以下是关于{name_en2zh[task]}的单项选择题，请分析并选出正确答案。\n题目：'
        else:
            prompt += f'以下是关于{name_en2zh[task]}的单项选择题，请直接给出正确答案的选项。\n题目：'
    if eval_method.endswith('multiple'):
        if cot:
            prompt += f'以下是关于{name_en2zh[task]}的多项选择题，请分析并选出正确答案。\n题目：'
        else:
            prompt += f'以下是关于{name_en2zh[task]}的多项选择题，请直接给出正确答案的选项。\n题目：'
    prompt += sample['question']
    if eval_method.endswith('single') or eval_method.endswith('multiple') or eval_method.endswith('single1'):
        for i in range(int(eval_method[0])):
            prompt += f'\n{chr(i + 65)}. {sample[chr(i + 65)]}'
        if cot and not include_answer:
            prompt += '\n逐步分析并给出答案选项：'
        else:
            prompt += '\n答案是：'
    else:
        prompt += '\n不用解释，直接给出答案，答案是：'
    if include_answer:
        prompt += sample['answer']
    return prompt


def extract_multiple_choice(response):
    choices = ''.join(chr(i + 65) for i in range(int(eval_method[0])))
    print(f"Response: {response}") 
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 3),
        (r'答案(是|为)选项 ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 2),
        (r'故?选择?：? ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
        (r'([ABCDE]+(?:[\/,，、和] ?[ABCDE])*) ?选?项都?(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 2),
        (r'答案(应该)?(是|为)([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 3),
        (r'选项 ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*) ?都?(是|为)?正确', 1),
        (r'选择答案 ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
        (r'答案?：?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
        (r'([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)(选?项)?都?是?符合题意', 1),
        (r'答案选项：? ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
        (r'答案(选项)?为(.*?)([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 3),
        (r'选项([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)都?是最恰当的', 1),
        (r'选项([ABCDE]+(?:[\/,，、和] ?[ABCDE])*).*最恰当', 1),
        (r'选项([ABCDE]+(?:[\/,，、和] ?[ABCDE])*).*最能恰当', 1),
        (r'选项([ABCDE]+(?:[\/,，、和] ?[ABCDE])*).*最能', 1),
        (r'最恰当.*是选项([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
        (r'correct answer is.*?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            answer = ''.join(ch for ch in answer if ch in choices)
            return answer
    # 2. Recursive match
    patterns = [
        (r'([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)(.*?)都?当选', 1),
        (r'([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)(.*?)都?正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                answer = ''.join(ch for ch in answer if ch in choices)
                m = re.search(pattern, m.group(0)[1:], re.M)
            return answer
    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            answer = ''.join(ch for ch in answer if ch in choices)
            return answer
    # 4. Check the only mentioned choices
    pattern = r'^[^ABCDE]*([ABCDE]+(?:[\/,，、和] ?[ABCDE])*)[^ABCDE]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        answer = ''.join(ch for ch in answer if ch in choices)
        return answer
    answer = random.sample(choices, random.randint(1, int(eval_method[0])))
    return ''.join(answer)


def extract_choice(response):
    choices = ''.join(chr(i + 65) for i in range(int(eval_method[0])))
    print(f"Response: {response}")  # Debugging output
    if response[0] in choices:
        return response[0]
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCDE])', 3),
        (r'答案(是|为)选项 ?([ABCDE])', 2),
        (r'故?选择?：? ?([ABCDE])', 1),
        (r'([ABCDE]) ?选?项(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCDE])', 2),
        (r'答案(应该)?(是|为)([ABCDE])', 3),
        (r'选项 ?([ABCDE]) ?(是|为)?正确', 1),
        (r'选择答案 ?([ABCDE])', 1),
        (r'答案?：?([ABCDE])', 1),
        (r'([ABCDE])(选?项)?是?符合题意', 1),
        (r'答案选项：? ?([ABCDE])', 1),
        (r'答案(选项)?为(.*?)([ABCDE])', 3),
        (r'选项([ABCDE])是最恰当的', 1),
        (r'选项([ABCDE]).*最恰当', 1),
        (r'选项([ABCDE]).*最能恰当', 1),
        (r'选项([ABCDE]).*最能', 1),
        (r'最恰当.*是选项([ABCDE])', 1),
        (r'correct answer is.*([ABCDE])', 1),
        (r'([ABCDE]),', 1),
        (r'([ABCDE]).', 1),
    ]

    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            if answer in choices:
                return answer
            else:
                print(f"Warning: Found answer '{answer}' not in choices.")

    patterns_recursive = [
        (r'([ABCDE])(.*?)当选', 1),
        (r'([ABCDE])(.*?)正确', 1),
    ]

    for pattern, idx in patterns_recursive:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            if answer in choices:
                return answer
            else:
                print(
                    f"Warning: Found recursive answer '{answer}' not in choices.")

    print("No valid choice found, returning a random choice.")
    return choices[random.randint(0, int(eval_method[0]) - 1)]


def extract_number(response):
    matches = re.findall(r'\d+', response)
    if not matches:
        return ''
    return matches[0]


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_template', type=str, default='llama3')
    parser.add_argument('--peft_model_path', type=str, default='')
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--num_few_shot', type=int, default=0)
    parser.add_argument('--test_cases', type=int, default=99999)
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--overwrite_output_file', action='store_true')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print detailed information of each example.')
    args = parser.parse_args()
    print(args)

    # load model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    random.seed(42)   
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                padding_side='left',
                                                trust_remote_code=True)
    prompt_end = tokenizer.decode(tokenizer.encode(
        templates[args.model_template]['suffix']),
        skip_special_tokens=True)
    if args.use_vllm:
        run = Runnervllm(args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=(torch.bfloat16
                         if torch.cuda.is_bf16_supported() else torch.float32))

        if args.peft_model_path:
            model = PeftModel.from_pretrained(
                model,
                args.peft_model_path,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float32))
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     model = torch.nn.DataParallel(model).to('cuda')
        # else:
        #     model.to('cuda')
        run = Runner(model, tokenizer)
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    config_updates = {
        'do_sample': True,
        'max_new_tokens': 1024,
        'min_new_tokens': 1,
        'temperature': 0.2,
        'repetition_penalty': 1.1
    }
    for key, value in config_updates.items():
        setattr(generation_config, key, value)

    # load tasks
    if args.task == 'all':
        tasks = [i for v in categories.values() for i in v]
    elif args.task in categories:
        tasks = categories[args.task]
    elif args.task in [i for j in categories.values() for i in j]:
        tasks = [args.task]
    else:
        assert args.task in categories

    # run evaluation tasks
    for task in tasks:
        save_file = os.path.join(args.output_path, f'{task}.json')
        if os.path.exists(save_file) and not args.overwrite_output_file:
            print(f'\033[31mSkip:\033[0m {name_en2zh[task]}')
            continue
        print(f'\033[36mGenerate:\033[0m {name_en2zh[task]}')
        # load data
        data = []
        if args.do_test:
            with open(os.path.join('eval_data', 'test', f'{task}.csv'),
                      'r',
                      encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i in csv_reader:
                    data.append(i)
        else:
            with open(os.path.join('eval_data', 'val', f'{task}.csv'),
                      'r',
                      encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i in csv_reader:
                    data.append(i)
        data = data[:args.test_cases]
        for k, v in categories.items():
            if task in v:
                eval_method = categories[k][task]
                generation_config.temperature = 0.01
                break

        with torch.no_grad():
            # preprocess data
            if args.num_few_shot:
                data_example = []
                with open(os.path.join('eval_data', 'dev', f'{task}.csv'),
                          'r',
                          encoding='utf-8') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for i in csv_reader:
                        data_example.append(i)
                data_example = data_example[:args.num_few_shot]
                instruction_example = '\n\n'.join([
                    gen_prompt(sample, task=task, include_answer=True)
                    for sample in data_example
                ]) + '\n\n'
            if args.use_vllm:
                instructions = [
                    gen_prompt(sample, task=task, cot=args.cot)
                    for sample in data
                ]
            else:
                instructions = [
                    templates[args.model_template]['prefix'] +
                    gen_prompt(sample, task=task, cot=args.cot) +
                    templates[args.model_template]['suffix'] for sample in data
                ]

            # generate answers
            if args.use_vllm:
                # use vllm
                outputs_all = []
                for i in trange(0,
                                len(instructions),
                                args.batch,
                                disable=args.verbose):
                    prompts = instructions[i:i + args.batch]
                    outputs = run.run_answer(
                        prompts, temperature=generation_config.temperature)
                    for j in range(min(len(outputs), args.batch)):
                        output = outputs[j].outputs[0].text
                        if args.verbose:
                            print(prompts[j], output)
                        if eval_method.endswith('single'):
                            output = extract_choice(output)
                        elif eval_method.endswith('multiple'):
                            output = extract_multiple_choice(output)
                        elif eval_method == 'number':
                            output = extract_number(output)
                        outputs_all.append(output)
                        print(output)
                for i, j in enumerate(outputs_all):
                    data[i]['output'] = j
            elif args.batch > 1:
                # batch generate
                outputs_all = []
                for i in trange(0,
                                len(instructions),
                                args.batch,
                                disable=args.verbose):
                    prompts = instructions[i:i + args.batch]
                    outputs = run.run_answer_batch(prompts, generation_config)
                    for j in range(min(len(outputs), args.batch)):
                        output = outputs[j]
                        output = output[output.index(prompt_end) +
                                        len(prompt_end):]
                        if args.verbose:
                            print(prompts[j], output)
                        if eval_method.endswith('single'):
                            output = extract_choice(output)
                        elif eval_method.endswith('multiple'):
                            output = extract_multiple_choice(output)
                        elif eval_method == 'number':
                            output = extract_number(output)
                        outputs_all.append(output)
                for i, j in enumerate(outputs_all):
                    data[i]['output'] = j
            else:
                for i, j in enumerate(tqdm(instructions,
                                           disable=args.verbose)):
                    output = run.run_answer(j, generation_config)
                    output = output[output.index(prompt_end) +
                                    len(prompt_end):]
                    if args.verbose:
                        print(j, output)
                    if eval_method.endswith('single'):
                        output = extract_choice(output)
                    elif eval_method.endswith('multiple'):
                        output = extract_multiple_choice(output)
                    elif eval_method == 'number':
                        output = extract_number(output)
                    data[i]['output'] = output

        if args.output_path:
            if not os.path.exists(args.output_path):
                os.mkdir(args.output_path)
            save_file = os.path.join(args.output_path, f'{task}.json')
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'\033[32mTime:\033[0m {execution_time:.2f}s')
