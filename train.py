from transformers import Trainer, TrainingArguments, default_data_collator

import random

from analyse import Analyse
from generate import Generate


class Train:

    @staticmethod
    def do_train(model, tokenizer, lm_datasets, output_dir):
        """Train model using a given tokenizer and dataset"""

        num_train_epochs = 6

        # seed_data = random.randint(0, 2**32-1)
        # arguments to pass to the trainer
        training_args = TrainingArguments(
            output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=1.372e-4,
            # seed=seed_data,
            # weight_decay=0.001,
            # learning_rate=0.001,
            weight_decay=0.01,
            num_train_epochs=num_train_epochs,
            save_total_limit=1,
            save_strategy='epoch',
            save_steps=1,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_checkpointing=True,
            report_to=None,
            logging_steps=1,
            do_eval=True,
            eval_steps=1,
            load_best_model_at_end=True
            # disable_tqdm=True
        )

        def preprocess_logits_for_metrics(logits, labels):
            """Taken from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py"""
            """Remove extra tensors from logits if needed"""
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        import evaluate
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            """Adapted from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py"""
            print("----computing metrics----")
            generate.do(output_dir=output_dir)

            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics, but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            if len(trainer.state.log_history) > 1:
                analyse.analyse_logs(trainer.state.log_history)
            return metric.compute(predictions=preds, references=labels)

        # set up trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # used to analyse training metrics and produce helpful graphs
        analyse = Analyse(output_dir)
        # used to generate sample text from the trained model
        generate = Generate(tokenizer, trainer=trainer)

        train_dataloader = trainer.get_train_dataloader()
        num_train_steps = len(train_dataloader)
        print("----Number of training steps----")
        print(num_train_steps)
        trainer.create_optimizer_and_scheduler(num_train_steps)

        def set_schedule(lr_type):
            """Returns a learning rate schedule"""

            if lr_type == "CONSTANT":
                from transformers import get_constant_schedule
                return get_constant_schedule(trainer.optimizer)
            if lr_type == "LINEAR":
                from transformers import get_linear_schedule_with_warmup
                return get_linear_schedule_with_warmup(trainer.optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=num_train_steps)
            if lr_type == "COSINE":
                from transformers import get_cosine_schedule_with_warmup
                return get_cosine_schedule_with_warmup(trainer.optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=num_train_steps)

        # Here can choose learning rate schedule to use
        trainer.lr_scheduler = set_schedule(lr_type="CONSTANT")

        print("----Training----")
        # train the model
        trainer.train()
        # save model
        trainer.save_model()
        # generate example text
        generate.do(output_dir=output_dir)

        # analyse results and plot graphs
        analyse.analyse_logs(trainer.state.log_history, show_graph=True)



