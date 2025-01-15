meta_template_dict = dict(
    internlm = [
        dict(role='system', begin='<|System|>:', end='\n'),
        dict(role='user', begin='<|User|>:', end='\n'),
        dict(role='function', begin='<|System|>:', end='\n'),
        dict(
            role='assistant',
            begin='<|Bot|>:',
            end='<eoa>\n',
            generate=True)
    ],
    #这个也不对
    llama2 = [
        dict(role='system', begin='[INST]', end='[\INST]'),
        dict(role='user', begin='[INST]', end='[\INST]'),
        dict(role='function', begin='[INST]', end='[\INST]'),
        dict(role='assistant',
                begin='',
                end='</s>',
                generate=True),
    ],
    #这个是对的，输出里面没有乱七八糟的token
    llama2_zh = [
        dict(role='system', begin='<s>[INST]<<SYS>>', end='<</SYS>>\n'),
        dict(role='user', begin='<s>[INST]', end='[/INST]'),
        dict(role='function', begin='<s>[INST]', end='[/INST]'),
        dict(role='assistant',
                begin='',
                end='</s>',
                generate=True),
    ],
    #不对
    llama3 = [
        dict(role='system', begin='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n', end='<|eot_id|>'),
        dict(role='user', begin='<|start_header_id|>user<|end_header_id|>\n', end='<|eot_id|>'),
        dict(role='function', begin='<|start_header_id|>user<|end_header_id|>\n', end='<|eot_id|>'),
        dict(role='assistant',
                begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
                end='<|eot_id|>',
                generate=True),
    ],
    qwen = [
        dict(role='user', api_role='user', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='system', api_role='system', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='function', api_role='user', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='assistant',
                api_role='assistant',
                begin='\n<|im_start|>assistant\n',
                end='<|im_end|>',
                generate=True),
    ],
    vicuna = [
        dict(role='user', begin='user: ', end='\n'),
        dict(role='system', begin='user: ', end='\n'),
        dict(role='function', begin='user: ', end='\n'),
        dict(role='assistant',
                begin='assistant: ',
                end='\n',
                generate=True),
    ],
    # we use chat APITemplateParser for chatglm due to some specific designs from ChatGLM
    chatglm = [
        dict(role='system', api_role='user'),
        dict(role='user', api_role='user'),
        dict(role='function', api_role='user'),
        dict(role='assistant',
            api_role='assistant',
            generate=True)
    ],
    baichuan = [
        dict(role='user', begin='<reserved_106>', end='\n'),
        dict(role='system', begin='<reserved_106>', end='\n'),
        dict(role='function', begin='<reserved_106>', end='\n'),
        dict(role='assistant', begin='<reserved_107>', end='\n',
                generate=True),
    ]
)
