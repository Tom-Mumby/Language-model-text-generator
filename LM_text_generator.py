from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
from preprocess import Preprocess

from os.path import exists

from train import Train
from config import Config

# Folder to write model to
output_dir = Config.output_dir
# Text file containing text to fine-tune or train from scratch
text_file = Config.text_file_location
from_scratch = Config.from_scratch

if not exists(output_dir):
    print("Output directory does not exist")
    exit()
if not exists(text_file):
    print("Text file does not exist")
    exit()

print("Output directory")
print(output_dir)
print("Text file location")
print(text_file)

tokenizer_file_name = output_dir + "/byte-level-BPE_son.tokenizer.json"
trainer_state_path = output_dir + '/trainer_state.json'

if from_scratch is False:  # fine-tuning gpt2
    print("----Fine-tuning model----")
    # tokenizer from GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # if a previously fine-tuned model exists
    if exists(trainer_state_path):  # set model name to previously trained
        print("Loading existing model")
        model_name = output_dir
    else:  # set model name to load gpt2 124M parameter model
        print("Loading GPT-2 124M parameter model")
        model_name = "gpt2"
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
else:  # training from scratch
    print("----Training from scratch----")
    if exists(tokenizer_file_name):  # model has already starting training
        print("Loading existing model")
        # load model and tokenizer from directory
        model = AutoModelForCausalLM.from_pretrained(output_dir, use_cache=False)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file_name)
    else:  # no model exists
        print("Setting up new model")
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2Config
        print("Creating tokenizer----")
        tokenizer = ByteLevelBPETokenizer()
        # train tokenizer on given text file, setting vocab to be around 52,000 (similar to GPT-2)
        print("Training tokenizer")
        tokenizer.train(files=text_file, vocab_size=52_000, min_frequency=2, special_tokens=["<|endoftext|>"])
        # save tokenizer so can be used for further training
        tokenizer.save(tokenizer_file_name)
        print("Creating model config")
        # uses same architecture as GPT-2
        config = GPT2Config(vocab_size=52_000, max_length=1024, use_cache=False)
        # create model from configuration
        print("Creating model")
        model = AutoModelForCausalLM.from_config(config=config)

    print("----Number of Model Parameters----")
    print(model.num_parameters())

# tokenize datasets
tokenized_datasets = Preprocess.process(tokenizer, text_file)
# train and run analysis
Train.do_train(model, tokenizer, tokenized_datasets, output_dir)
