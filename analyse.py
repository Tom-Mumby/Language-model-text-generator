import json
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os


class Analyse:
    """Runs analysis on the metrics generated during training"""

    def __init__(self, output_dir):
        # directory where trained model is
        self.output_dir = output_dir

        # path of the file containing the trainer state information
        self.trainer_state_path = 'trainer_state.json'
        self.logs_path = 'logs.json'

        self.start_epoch = 0
        self.start_step = 0

        self.previous_logs = []
        # checks if model had already been trained
        if os.path.isfile(self.output_dir + "/" + self.logs_path):
            print("Has been pre trained")
            # loads trainer state
            self.previous_logs = self.__load_json(self.logs_path)
            # sets the information about the starting state
            self.start_epoch = round(self.previous_logs[-1]['epoch'])
            self.start_step = self.previous_logs[-1]['step'] + 1

        print("Starting from epoch")
        print(self.start_epoch)

    def analyse_logs(self, new_logs, show_graph=False):
        """Processes log files created during the training, finds the minimum validation loss and maximum accuracy"""

        if len(new_logs) > 0:
            import copy
            new_logs_copy = copy.deepcopy(new_logs)
            # training has finished
            if 'train_runtime' in new_logs_copy[-1]:
                end_state = new_logs_copy[-1]
                self.__write_json(end_state, self.trainer_state_path)
                del new_logs_copy[-1]

            # if needed increase epoch to match start epoch
            for entry in new_logs_copy:
                entry['epoch'] += self.start_epoch
                entry['step'] += self.start_step
        else:
            new_logs_copy = []

        # write new logs to file
        output_list_copy = self.previous_logs + new_logs_copy
        self.__write_json(output_list_copy, self.logs_path)

        # for training data
        self.train_epoch = []
        self.train_loss = []
        self.train_lr = []

        # for evaluation data
        self.eval_epoch = []
        self.eval_loss = []
        self.eval_accuracy = []

        # loop over entries training output
        for i in range(len(output_list_copy)):
            current_dict = output_list_copy[i]
            if 'loss' in current_dict:  # is a trainer state entry
                self.train_epoch.append(current_dict['epoch'])
                self.train_loss.append(current_dict['loss'])
                self.train_lr.append(current_dict['learning_rate'])

            else:  # is an evaluation state entry
                self.eval_epoch.append(current_dict['epoch'])
                self.eval_loss.append(current_dict['eval_loss'])
                self.eval_accuracy.append(current_dict['eval_accuracy'])

        def plot_graph():
            # plot graph
            fig = plt.figure()
            fig.set_size_inches(14, 6)
            # create subplots
            ax_loss = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
            ax_accuracy = plt.subplot2grid((2, 2), (0, 1))
            ax_lr = plt.subplot2grid((2, 2), (1, 1))

            # add evaluation loss to first subplot
            ax_loss.plot(self.eval_epoch, self.eval_loss, label="Evaluation loss")

            # add training loss to first subplot
            ax_loss.plot(self.train_epoch, self.train_loss, 'tab:orange', label="Training loss")

            try:
                ax_loss.plot(self.train_epoch, signal.savgol_filter(self.train_loss, window_length=50, polyorder=3), 'tab:green', label="Smoothed training loss")
            except ValueError:
                print("Not enough points to run smoothing")

            ax_loss.set_title('Loss', loc="left", size=15)
            ax_loss.legend(loc="lower left")

            # plot evaluation accuracy
            ax_accuracy.plot(self.eval_epoch, self.eval_accuracy)
            ax_accuracy.set_title('Evaluation Accuracy', loc="left")

            # plot learning rate graph
            ax_lr.plot(self.train_epoch, self.train_lr, 'tab:red')
            ax_lr.set_title('Learning Rate', loc="right")
            ax_lr.ticklabel_format(scilimits=(0, 0), useMathText=True)
            plt.tight_layout()

            try:
                # add max and min labels
                min_text = "Min validation loss: {:.3f} at epoch: {:.1f}".format(min(self.eval_loss), self.eval_epoch[np.argmin(self.eval_loss)])
                ax_loss.text(0.5, 0.96, min_text, transform=ax_loss.transAxes)
                max_text = "Max accuracy: {:.3f} at epoch: {:.1f}".format(max(self.eval_accuracy), self.eval_epoch[np.argmax(self.eval_accuracy)])
                ax_accuracy.text(0.56, 0.1, max_text, transform=ax_accuracy.transAxes)

            except ValueError:
                print("No validation data exists yet")

            plt.savefig(self.output_dir + "/graphs.png")

            if show_graph:
                plt.show()

            plt.close()

        plot_graph()

    def __load_json(self, filename):
        """Loads a json file"""
        with open(self.output_dir + '/' + filename, 'r') as f:
            return json.load(f)

    def __write_json(self, data_to_write, filename):
        """Writes to a json file"""
        with open(self.output_dir + '/' + filename, 'w') as f:
            json.dump(data_to_write, f, indent=2)


if __name__ == "__main__":
    # run analysis on a specified model
    analyse = Analyse("PATH_TO_MODEL")
    analyse.analyse_logs([], show_graph=True)
