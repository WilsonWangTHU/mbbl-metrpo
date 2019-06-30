import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_LENGTH = 99999


def get_plot_data_from_single_experiment(exp_dir, file_name, column_x, column_y, start_index=1):
    try:
        data = pd.read_csv(os.path.join(exp_dir, file_name))
    except pd.errors.EmptyDataError:
        return [], []

    data_y = data[column_y][:options.x_max]
    if options.column_y_transformation == "softplus":
        data_y = [np.log(np.exp(datum_y) + 1) for datum_y in data_y]

    if column_x == "Iteration":
        data_x = range(start_index, len(data_y) + start_index)
    else:
        data_x = data[column_x][:options.x_max]
    return data_x, data_y


def add_curve_for_experiment(data_dir, label=None, start_index=1):
    if os.path.exists(os.path.join(data_dir, options.file_name)):
        plot_indices_single_exp, plot_data_single_exp = get_plot_data_from_single_experiment(
            data_dir, options.file_name, options.column_x, options.column_y, start_index=start_index)
        plot_indices = [plot_indices_single_exp]
        plot_data = [plot_data_single_exp]
    else:
        plot_data = []
        plot_indices = []
        for sub_name in os.listdir(data_dir):
            sub_dir = os.path.join(data_dir, sub_name)
            if os.path.isdir(sub_dir) and "seed" in sub_name and "viz" not in sub_name and \
                    os.path.exists(os.path.join(sub_dir, options.file_name)):
                print("=> Obtaining data from subdirectory {}".format(sub_dir))
                plot_indices_single_exp, plot_data_single_exp = get_plot_data_from_single_experiment(
                    sub_dir, options.file_name, options.column_x, options.column_y, start_index=start_index)
                plot_indices.append(plot_indices_single_exp)
                plot_data.append(plot_data_single_exp)

    add_curve(plot_data, plot_indices, label)


def add_curve_for_experiment_with_dir_substr(cur_dir, dir_substr, label=None, start_index=1):
    print("=> Obtaining data in directory {} containing sub string {}".format(cur_dir, dir_substr))
    plot_data = []
    plot_indices = []
    for sub_name in os.listdir(cur_dir):
        sub_dir = os.path.join(cur_dir, sub_name)
        if dir_substr in sub_dir:
            plot_indices_single_exp, plot_data_single_exp = get_plot_data_from_single_experiment(
                sub_dir, options.file_name, options.column_x, options.column_y, start_index=start_index)
            plot_indices.append(plot_indices_single_exp)
            plot_data.append(plot_data_single_exp)

    add_curve(plot_data, plot_indices, label)


def create_plot():
    plt.figure()
    plt.title(options.title)


def save_plot():
    plt.xlabel(options.column_x)
    plt.ylabel(options.column_y)
    plt.savefig(os.path.join(options.save_fig_dir, options.save_fig_filename))
    plt.close()


def add_curve(data, indices, label):
    longest_length = len(data[0])
    longest_length_indices = indices[0]
    for i in range(1, len(data)):
        if len(data[i]) > longest_length:
            longest_length = len(data[i])
            longest_length_indices = indices[i]
    mean_data = []
    std_data = []
    for itr in range(longest_length):
        itr_values = []
        for curve_i in range(len(data)):
            if itr < len(data[curve_i]):
                itr_values.append(data[curve_i][itr])
        mean_data.append(np.mean(itr_values))
        std_data.append(np.std(itr_values))

    mean_data = np.array(mean_data)
    std_data = np.array(std_data)

    if label is None:
        plt.plot(longest_length_indices, mean_data)
    else:
        plt.plot(longest_length_indices, mean_data, label=label)
    plt.fill_between(longest_length_indices, mean_data + std_data, mean_data - std_data, alpha=0.3)


def add_legend():
    plt.legend(loc='best')


def use_option_or_input_label(option_argument, prompt_content, substr=False):
    if option_argument is None:
        if substr:
            return input("Enter label for data in directory with sub string {}: ".format(prompt_content))
        return input("Enter label for data in directory '{}': ".format(prompt_content))
    return option_argument


def plot_in_dir_with_substr(sub_str, option_argument):
    if sub_str is not None:
        label = use_option_or_input_label(option_argument, sub_str, substr=True)
        add_curve_for_experiment_with_dir_substr(options.data_dir, sub_str, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_dir_start_index", type=int, default=1)
    parser.add_argument("--multi_curve", type=bool, default=False)
    parser.add_argument("--use_dir_substr", type=bool, default=False)
    parser.add_argument("--dir1_substr", type=str, default=None)
    parser.add_argument("--dir1_substr_curve_name", type=str, default=None)
    parser.add_argument("--dir2_substr", type=str, default=None)
    parser.add_argument("--dir2_substr_curve_name", type=str, default=None)
    parser.add_argument("--dir3_substr", type=str, default=None)
    parser.add_argument("--dir3_substr_curve_name", type=str, default=None)
    parser.add_argument("--dir4_substr", type=str, default=None)
    parser.add_argument("--dir4_substr_curve_name", type=str, default=None)
    parser.add_argument("--dir5_substr", type=str, default=None)
    parser.add_argument("--dir5_substr_curve_name", type=str, default=None)
    parser.add_argument("--dir6_substr", type=str, default=None)
    parser.add_argument("--dir6_substr_curve_name", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="Iteration")
    parser.add_argument("--column_y", type=str, default="AverageReturn")
    parser.add_argument("--column_y_transformation", type=str, default=None)

    parser.add_argument("--title", type=str, default="Training Average Reward Curve")
    parser.add_argument("--save_fig_filename", type=str, default="plot.png")
    parser.add_argument("--save_fig_dir", type=str, default=os.path.curdir)
    parser.add_argument("--x_max", type=int, default=MAX_LENGTH)
    parser.add_argument("--extra_dir1", type=str, default=None)
    parser.add_argument("--extra_dir1_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir1_start_index", type=int, default=1)
    parser.add_argument("--extra_dir2", type=str, default=None)
    parser.add_argument("--extra_dir2_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir2_start_index", type=int, default=1)
    parser.add_argument("--extra_dir3", type=str, default=None)
    parser.add_argument("--extra_dir3_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir3_start_index", type=int, default=1)
    parser.add_argument("--extra_dir4", type=str, default=None)
    parser.add_argument("--extra_dir4_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir4_start_index", type=int, default=1)
    parser.add_argument("--extra_dir5", type=str, default=None)
    parser.add_argument("--extra_dir5_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir5_start_index", type=int, default=1)
    parser.add_argument("--extra_dir6", type=str, default=None)
    parser.add_argument("--extra_dir6_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir6_start_index", type=int, default=1)
    parser.add_argument("--extra_dir7", type=str, default=None)
    parser.add_argument("--extra_dir7_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir7_start_index", type=int, default=1)
    parser.add_argument("--extra_dir8", type=str, default=None)
    parser.add_argument("--extra_dir8_curve_name", type=str, default=None)
    parser.add_argument("--extra_dir8_start_index", type=int, default=1)
    parser.add_argument("--default_curve_name", type=str, default='default-curve')

    options = parser.parse_args()

    create_plot()
    if options.multi_curve:
        if options.use_dir_substr:
            plot_in_dir_with_substr(options.dir1_substr, options.dir1_substr_curve_name)
            plot_in_dir_with_substr(options.dir2_substr, options.dir2_substr_curve_name)
            plot_in_dir_with_substr(options.dir3_substr, options.dir3_substr_curve_name)
            plot_in_dir_with_substr(options.dir4_substr, options.dir4_substr_curve_name)
            plot_in_dir_with_substr(options.dir5_substr, options.dir5_substr_curve_name)
            plot_in_dir_with_substr(options.dir6_substr, options.dir6_substr_curve_name)
        else:
            for sub_name in os.listdir(options.data_dir):
                sub_dir = os.path.join(options.data_dir, sub_name)
                if os.path.isdir(sub_dir) and "viz" not in sub_dir:
                    label = input("Enter label for data in directory '{}': ".format(sub_dir))
                    add_curve_for_experiment(sub_dir, label=label)
        add_legend()
    else:
        if options.extra_dir1:
            label = use_option_or_input_label(options.extra_dir1_curve_name, options.extra_dir1)
            add_curve_for_experiment(options.extra_dir1, label=label, start_index=options.extra_dir1_start_index)
        if options.extra_dir2:
            label = use_option_or_input_label(options.extra_dir2_curve_name, options.extra_dir2)
            add_curve_for_experiment(options.extra_dir2, label=label, start_index=options.extra_dir2_start_index)
        if options.extra_dir3:
            label = use_option_or_input_label(options.extra_dir3_curve_name, options.extra_dir3)
            add_curve_for_experiment(options.extra_dir3, label=label, start_index=options.extra_dir3_start_index)
        if options.extra_dir4:
            label = use_option_or_input_label(options.extra_dir4_curve_name, options.extra_dir4)
            add_curve_for_experiment(options.extra_dir4, label=label, start_index=options.extra_dir4_start_index)
        if options.extra_dir5:
            label = use_option_or_input_label(options.extra_dir5_curve_name, options.extra_dir5)
            add_curve_for_experiment(options.extra_dir5, label=label, start_index=options.extra_dir5_start_index)
        if options.extra_dir6:
            label = use_option_or_input_label(options.extra_dir6_curve_name, options.extra_dir6)
            add_curve_for_experiment(options.extra_dir6, label=label, start_index=options.extra_dir6_start_index)
        if options.extra_dir7:
            label = use_option_or_input_label(options.extra_dir7_curve_name, options.extra_dir7)
            add_curve_for_experiment(options.extra_dir7, label=label, start_index=options.extra_dir7_start_index)
        if options.extra_dir8:
            label = use_option_or_input_label(options.extra_dir8_curve_name, options.extra_dir8)
            add_curve_for_experiment(options.extra_dir8, label=label, start_index=options.extra_dir8_start_index)
        add_curve_for_experiment(options.data_dir, label=options.default_curve_name, start_index=options.data_dir_start_index)
        add_legend()
    save_plot()
