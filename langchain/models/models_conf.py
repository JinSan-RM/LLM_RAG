KO_EN_MODEL_PATH = "../models/ko_en"
EN_KO_MODEL_PATH = "../models/en_ko"


class ModelParam():
    def __init__(self, model: str):
        self.model = model

    def param_set(self):
        model_max_token = None
        final_summary_length = None
        max_tokens_per_chunk = None
        if self.model == 'bllossom':
            model_max_token = 2048
            final_summary_length = 1000
            max_tokens_per_chunk = 1000
        elif self.model == 'solar':
            model_max_token = 4096
            final_summary_length = 3000
            max_tokens_per_chunk = 3000
        elif self.model == 'llama3.2':
            model_max_token = 8192
            final_summary_length = 6000
            max_tokens_per_chunk = 6000
        elif self.model == 'EEVE':
            model_max_token = 4096
            final_summary_length = 3000
            max_tokens_per_chunk = 3000
        elif self.model == 'EXAONE':
            model_max_token = 2048
            final_summary_length = 1000
            max_tokens_per_chunk = 1000

        return model_max_token, final_summary_length, max_tokens_per_chunk
