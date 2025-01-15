# coding=utf-8

import argparse
import csv
import json
import os
import random
import re
import time
from threading import Thread

from openai import OpenAI
from tqdm import tqdm, trange
from zhipuai import ZhipuAI

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

    def __init__(self, model, api, url):
        self.model = model
        self.api = api
        self.url = url
        self.temperature = 0.2

    def run_answer(self,
                   prompt: str,
                   system='You are a helpful assistant. 你是一个乐于助人的助手。'):
        if 'glm' in self.model:
            client = ZhipuAI(api_key=self.api)
        else:
            client = OpenAI(api_key=self.api, base_url=self.url)
        response = None
        count = 0
        while response is None and count < 5:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': system
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        },
                    ],
                    temperature=self.temperature).choices
            except Exception as e:
                print(e)
            if count:
                time.sleep(15)
            count += 1
        if response is None:
            raise ValueError('API Error')
        return response[0].message.content

    def run_thread(self, dic, system):
        output = self.run_answer(dic['question'], system)
        dic['output'] = output

    def run_answer_batch(self,
                         prompts: list,
                         system='You are a helpful assistant. 你是一个乐于助人的助手。'):
        data = [{'question': i} for i in prompts]
        threads = []
        for i in data:
            threads.append(Thread(target=self.run_thread, args=(i, system)))
            threads[-1].daemon = True
        for i in threads:
            i.start()
        for i in threads:
            i.join()
        outputs = [i['output'] for i in data]
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


def extract_choice(response):
    choices = ''.join(chr(i + 65) for i in range(int(eval_method[0])))
    print(f"Response: {response}") 
    if response[0] in choices:
        return response[0]
    # 1. Single match
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
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer
    # 2. Recursive match
    patterns = [
        (r'([ABCDE])(.*?)当选', 1),
        (r'([ABCDE])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer
    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCDE])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer
    # 4. Check the only mentioned choices
    pattern = r'^[^ABCDE]*([ABCDE])[^ABCDE]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        if answer not in choices:
            answer = ''
        return answer
    return choices[random.randint(0, int(eval_method[0]) - 1)]


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


def extract_number(response):
    matches = re.findall(r'\d+', response)
    if not matches:
        return ''
    return matches[0]


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_api', type=str, required=True)
    parser.add_argument('--model_url',
                        type=str,
                        default='https://api.openai.com/v1')
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
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
    random.seed(42)
    run = Runner(model=args.model_name, api=args.model_api, url=args.model_url)

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
                run.temperature = 0.01
                break

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
        instructions = [
            gen_prompt(sample, task=task, cot=args.cot) for sample in data
        ]

        # generate answers
        if args.batch > 1:
            # batch generate
            outputs_all = []
            for i in trange(0,
                            len(instructions),
                            args.batch,
                            disable=args.verbose):
                prompts = instructions[i:i + args.batch]
                outputs = run.run_answer_batch(prompts)
                for j in range(min(len(outputs), args.batch)):
                    output = outputs[j]
                    if args.verbose:
                        print(prompts[j], output)    
                    if eval_method.endswith('single'):
                        output = extract_choice(output)
                    elif eval_method.endswith('multiple'):
                        output = extract_multiple_choice(output)
                    elif eval_method == 'number':
                        output = extract_number(output)
                    elif eval_method.endswith('single1'):   
                        if output is None:
                            output = "None" 
                    outputs_all.append(output)
            for i, j in enumerate(outputs_all):
                data[i]['output'] = j
        else:
            for i, j in enumerate(tqdm(instructions, disable=args.verbose)):
                output = run.run_answer(j)
                if args.verbose:
                    print(j, output)
                if eval_method.endswith('single'):
                    output = extract_choice(output)
                elif eval_method.endswith('multiple'):
                    output = extract_multiple_choice(output)
                elif eval_method == 'number':
                    output = extract_number(output)
                elif eval_method.endswith('single1'):   
                        if output is None:
                            output = "None"     
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
