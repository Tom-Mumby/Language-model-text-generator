class Config:
    """Sets model directory and location of text file as well as type of training"""
    # output directory to save the model to
    output_dir = "/home/thomas/New_LM/tiny_shakespeare"
    # location of text file to fine-tune or train from scratch using
    text_file_location = "/home/thomas/New_LM/text_files/tiny_shakespeare.txt"
    # set to True if training from skratch and False if fine-tuning GPT2
    from_scratch = False
