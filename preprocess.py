from datasets import Dataset, DatasetDict


class Preprocess:

    @staticmethod
    def process(tokenizer, text_file):
        """Tokenize a given text file"""

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        def group_texts(examples):
            """Group dataset into lengths of the block size
            Taken from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py"""

            # concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # drop the small remainder
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        def read_file_by_eot():
            """Read a text file, splitting by an end of text marker"""
            with open(text_file, "r") as file:
                content = file.readlines()
            content_list = []
            item = ""
            for line in content:
                item += line
                if "<|endoftext|>" in line:
                    content_list.append(item)
                    item = ""

            return content_list

        def read_file_by_split(maxsplit=100):
            """Read a text file, splitting by number of lines"""
            with open(text_file, "r") as file:
                content = file.readlines()

            content_list = []
            item = ""
            for i in range(len(content)):
                item += content[i]
                if i % maxsplit == 0:
                    content_list.append(item)
                    item = ""

            return content_list

        # split text file into chunks demarcated by <|endoftext|> token and add these to a file
        # item_list = read_file_by_eot()

        # split text file into chunks of a given length and add these to a file
        item_list = read_file_by_split()

        print("Total number of parts text file has been split into")
        print(len(item_list))
        # Fraction of to use in training set, the rest will go in the validation set. There is no test set in use in this script.
        train_percentage = 0.9

        # split up the list containing the chunks of text
        cut_position = round(len(item_list) * train_percentage)
        train_data = {'text': item_list[:cut_position]}
        validation_data = {'text': item_list[cut_position:]}

        print("Number in training set:")
        print(len(train_data['text']))
        print("Number in validation set:")
        print(len(validation_data['text']))

        # create the text dataset
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(train_data),
                'validation': Dataset.from_dict(validation_data)
            }
        )
        # Another way to load dataset
        # raw_datasets = load_dataset("text", data_files="PATH_TO_TEXT_FILE", sample_by="paragraph")

        print("----Datasets----")
        print(datasets)

        # tokenize these data sets, using map to speed up the process !!try without
        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

        print("----Tokenized Datasets----")
        print(tokenized_datasets)
        # block_size = int(tokenizer.model_max_length / 1)
        # set block size to split dataset into, default for GPT-2 is 1024
        block_size = int(1024 / 1)
        print("----Block Size----")
        print(block_size)

        # group datasets into specified block size
        lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=1)

        print("----Resized----")
        print(lm_datasets)
        print("----Length of block in dataset----")
        print(len(lm_datasets["train"][-1]["input_ids"]))

        # check can decode correctly
        print("----Decoded example----")
        print(tokenizer.decode(lm_datasets["validation"][-1]["input_ids"]))
        # return dataset
        return lm_datasets
