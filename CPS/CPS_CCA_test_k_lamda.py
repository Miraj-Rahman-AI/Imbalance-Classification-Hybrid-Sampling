import random
import math
from dataprocess import data_process
from keypattern import clean_pattern_identify
from keypattern import critical_pattern_identify
from oversampling import oversampling1
from oversampling import oversampling2
from Safe_region import Construct_safe_region


def CPS(train_data, k, lambda1, lambda2):
    num_of_majority, num_of_minority, majority_index, minority_index, distance_matrix = data_process(train_data)

    print('parameter k:' + str(k) + ' parameter alpha:' + str(lambda2) + ' parameter beta:' + str(lambda1))
    critical_index = []
    critical_pattern = []
    for i in range(num_of_minority):
        pattern_index = minority_index[i]
        if critical_pattern_identify(pattern_index, minority_index, distance_matrix, k, lambda1) == 1:
            critical_pattern.append(train_data[pattern_index])

    cleaning_index = []
    for i in range(num_of_majority):  # lambda2 for cleaning majority samples
        pattern_index = majority_index[i]
        if clean_pattern_identify(pattern_index, majority_index, distance_matrix, k, lambda2) == 1:
            cleaning_index.append(pattern_index)

    if num_of_majority - num_of_minority > len(cleaning_index):
        cleaning_index_final = cleaning_index
        oversampling = 1
    else:
        clean_num = num_of_majority - num_of_minority
        cleaning_index_final = random.sample(cleaning_index, clean_num)

        oversampling = 0

    #  cleaning majority samples
    train_data_after_cleaning = []
    for i in range(len(train_data)):
        if i not in cleaning_index_final:
            train_data_after_cleaning.append(train_data[i])

    oversampling_pattern = []
    if oversampling == 1 and len(critical_pattern) != 0:
        num_of_majority, num_of_minority, majority_index, minority_index, distance_matrix = data_process(
            train_data_after_cleaning)
        for i in range(num_of_minority):
            if train_data_after_cleaning[minority_index[i]] in critical_pattern:
                critical_index.append(minority_index[i])

        # critical_index
        safe_region_list, length_of_safe_region = Construct_safe_region(critical_index, majority_index, minority_index,
                                                                        distance_matrix)

        oversampling_num = num_of_majority - num_of_minority

        oversampling_num_list = []

        num_all = 0
        for i in range(len(length_of_safe_region)):
            num_all = num_all + length_of_safe_region[i]

        for i in range(len(safe_region_list)):
            oversampling_num_list.append(0)

        index = 0
        rest_num = oversampling_num
        while rest_num > 0:
            if index == 0:
                for i in range(len(safe_region_list)):
                    length = math.floor(oversampling_num * length_of_safe_region[i] / num_all)
                    while rest_num > 0 and length > 0:
                        oversampling_num_list[i] = oversampling_num_list[i] + 1
                        rest_num = rest_num - 1
                        length = length - 1
                index = 1
            else:
                while rest_num > 0:
                    select = random.randint(0, len(safe_region_list) - 1)
                    oversampling_num_list[select] = oversampling_num_list[select] + 1
                    rest_num = rest_num - 1

        for i in range(len(safe_region_list)):
            list = safe_region_list[i]
            for j in range(oversampling_num_list[i]):
                select = random.randint(1, len(list) - 1)
                center = list[0]  # 
                index = list[select]  #
                if index in majority_index:
                    pattern1 = train_data_after_cleaning[center].copy()
                    del pattern1[-1]
                    pattern2 = train_data_after_cleaning[index].copy()
                    del pattern2[-1]
                    # sample generation with minority border pattern and close majority sample
                    new_pattern = oversampling2(pattern1, pattern2)
                    oversampling_pattern.append(new_pattern)
                elif index in minority_index:
                    # sample generation with minority border pattern and minority sample
                    pattern1 = train_data_after_cleaning[center].copy()
                    del pattern1[-1]
                    pattern2 = train_data_after_cleaning[index].copy()
                    del pattern2[-1]
                    new_pattern = oversampling1(pattern1, pattern2)
                    oversampling_pattern.append(new_pattern)

    data_classify = train_data_after_cleaning + oversampling_pattern

    train_label_resampled = []
    train_pattern_resampled = []

    for i in range(len(data_classify)):
        pattern = data_classify[i].copy()
        train_label_resampled.append(pattern.pop())
        train_pattern_resampled.append(pattern)

    return train_pattern_resampled, train_label_resampled
