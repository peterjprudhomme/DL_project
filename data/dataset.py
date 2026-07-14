from datasets import load_dataset


class HarmBenchDataset:

    def __init__(self, config="standard"):
        # config can be ['contextual', 'copyright', 'standard']
        # standard has 200 prompts, need to test other configs

        # must be logged into HuggingFace with approved account to work
        # I used "hf auth login" in my VS Code terminal to login to my HuggingFace account with a token
        self.dataset = load_dataset(
            "walledai/HarmBench",
            config
        )

    def __len__(self):
        return len(self.dataset["train"]["prompt"])

    def get_example(self, index):
        return self.dataset["train"]["prompt"][index]
    
    def get_prompts(self):
        # returns list of prompts
        return self.dataset["train"]["prompt"]
    
    def get_categories(self):
        # returns categories of prompts (in same order as get_prompts list)
        return self.dataset["train"]["category"]

