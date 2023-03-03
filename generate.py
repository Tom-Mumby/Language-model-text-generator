from transformers import AutoTokenizer, AutoModelForCausalLM


class Generate:
    """Generate text from model"""

    def __init__(self, tokenizer, trainer=None, model=None):
        # check a trainer or a model has been passed
        assert trainer is not None or model is not None
        # check both a trainer and a model have not been passed
        assert not (trainer is not None and model is not None)

        self.tokenizer = tokenizer
        self.trainer = trainer
        self.model = model

        # set model if trainer has been passed
        if self.model is None:
            self.model = self.trainer.model

    def do(self, output_dir=None, prompt=None):
        """Pass a trainer or a model to generate text"""

        # tokenize input ids if passed
        input_ids = None
        if prompt is not None:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # generate output from model
        sample_outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            min_length=25,
            max_length=150,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=3
        )

        output_string = "\n"

        # add epoch information
        if self.trainer is not None:
            output_string += "Epoch " + str(self.trainer.state.epoch) + '\n'

        # decode each sample
        for sample_output in sample_outputs:
            output_string += 60 * '-' + '\n'
            output_string += self.tokenizer.decode(sample_output, skip_special_tokens=True) + '\n'

        # print output text to screen
        print(output_string)

        # if output directory set, write to file
        if output_dir is not None:
            file_dir = output_dir + "/training_test_generated.txt"
            with open(file_dir, 'a') as f:
                f.write(output_string)


if __name__ == "__main__":
    generate = Generate(AutoTokenizer.from_pretrained("gpt2"), model=AutoModelForCausalLM.from_pretrained("PATH_TO_MODEL"))
    generate.do(prompt="Lets go ")



