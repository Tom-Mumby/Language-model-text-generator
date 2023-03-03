import json
import matplotlib.pyplot as plt
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

        print("Epoch")
        print(self.start_epoch)

    def __load_json(self, filename):
        """Loads a json file"""
        with open(self.output_dir + '/' + filename, 'r') as f:
            return json.load(f)

    def __write_json(self, data_to_write, filename):
        """Writes to a json file"""
        with open(self.output_dir + '/' + filename, 'w') as f:
            json.dump(data_to_write, f, indent=2)

    def process_logs(self, new_logs, show_graph=False):
        """Processes log files created during the training, finds the minimum validation loss and maximum accuracy"""

        import copy
        if len(new_logs) > 0:
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

        # for finding max and min values
        min_val_loss = 9999
        val_epoch = 0
        max_accuracy = 0
        accuracy_epoch = 0

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

                # check if current values are better than the stored max and min values
                if current_dict['eval_loss'] < min_val_loss:
                    min_val_loss = current_dict['eval_loss']
                    val_epoch = current_dict['epoch']
                if current_dict['eval_accuracy'] > max_accuracy:
                    max_accuracy = current_dict['eval_accuracy']
                    accuracy_epoch = current_dict['epoch']

        print("Minimum Validation Loss:\t" + str(min_val_loss) + " at Epoch:\t" + str(round(val_epoch)))
        print("Maximum accuracy:\t" + str(max_accuracy) + " at Epoch:\t" + str(round(accuracy_epoch)))

        # plot graph
        fig = plt.figure()
        fig.set_size_inches(14, 6)
        # create subplots
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        # add evaluation loss to first subplot
        ax1.plot(self.eval_epoch, self.eval_loss, label="Evaluation Loss")

        # add training loss to first subplot
        ax1.plot(self.train_epoch, self.train_loss, 'tab:orange', label="Training Loss")
        ax1.set_title('Loss')
        ax1.legend(loc="upper right")
        # plot evaluation accuracy
        ax2.plot(self.eval_epoch, self.eval_accuracy)
        ax2.set_title('Evaluation Accuracy')
        # plot learning rate graph
        ax3.plot(self.train_epoch, self.train_lr, 'tab:red')
        ax3.set_title('Learning Rate')
        plt.tight_layout()

        if show_graph:
            plt.show()

        plt.savefig(self.output_dir + "/graphs.png")
        plt.close()


if __name__ == "__main__":
    # run analysis on a specified model
    analyse = Analyse("PATH_TO_MODEL")
    analyse.process_logs([], show_graph=True)
