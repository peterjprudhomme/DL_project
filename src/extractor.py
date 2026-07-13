from .activation_sample import ActivationSample


class ActivationExtractor:

    def __init__(self, llm):

        self.llm = llm

    def process_prompt(
        self,
        prompt,
        layers
    ):
        """
        Process a single prompt.
        """
        
        response = self.llm.generate(prompt)
        activations = self.llm.get_activations(
            prompt,
            layers
        )

        sample = ActivationSample(
            prompt=prompt,
            response=response,
            activations=activations
        )

        return sample

    def process_dataset(self, prompts, layers):

        dataset = []

        for prompt in prompts:
            sample = self.process_prompt(
                prompt,
                layers
            )
            dataset.append(sample)

        return dataset