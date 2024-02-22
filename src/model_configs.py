import os

VIRTUAL_LLM_URL = os.environ.get(
    'VIRTUAL_LLM_URL', 'http://localhost:10000/v1/')

BASEDIR = 'BASEDIR'

MODELS = {
    'llama-2': {
        'model_name': 'llama-2',
        'model_path': f'{BASEDIR}/Llama-2-7b-chat-hf',
        'tokenizer_path': f'{BASEDIR}/Llama-2-7b-chat-hf',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_name': 'vicuna',
        'model_path': f'{BASEDIR}/vicuna-13b-v1.5',
        'tokenizer_path': f'{BASEDIR}/vicuna-13b-v1.5',
        'conversation_template': 'vicuna-v1.5'
    },
    'mistral': {
        'model_name': 'mistral',
        'model_path': f'{BASEDIR}/mistral-7b',
        'tokenizer_path': f'{BASEDIR}/mistral-7b',
        'conversation_template': 'mistral'
    },
    'gpt-3.5-turbo-0613': {
        'model_name': 'gpt-3.5-turbo-0613',
        'conversation_template': 'gpt-3.5-turbo'
    },
    'gpt-3.5-turbo-1106': {
        'model_name': 'gpt-3.5-turbo-1106',
        'conversation_template': 'gpt-3.5-turbo'
    },
    'gpt-3.5-turbo': {
        'model_name': 'gpt-3.5-turbo',
        'conversation_template': 'gpt-3.5-turbo'
    }
}

OPENAI_API_KEY = "API-KEY"
