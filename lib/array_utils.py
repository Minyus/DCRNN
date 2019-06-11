import numpy as np


def every_n_reduce_mean(arr, n=2):
    remainder = len(arr) % n
    if remainder:
        arr = arr[remainder:]
        print('Warning: dropped remainder of {} samples.'.format(remainder))
    return np.mean(arr.reshape(-1, n), axis=-1)


def partitioned_reduce_mean(arr, part_list=[1, 8, 96, 672], rest_to_one=True):
    out_list = []
    for i in range(len(part_list)):
        if i + 1 < len(part_list):
            part_arr = arr[-part_list[i+1]:]
            reduced_part_arr = every_n_reduce_mean(part_arr, part_list[i])
            arr = arr[:-part_list[i+1]]
        else:
            if rest_to_one:
                reduced_part_arr = np.mean(arr, keepdims=True)
            else:
                reduced_part_arr = every_n_reduce_mean(arr, part_list[i])
        out_list.append(reduced_part_arr)
    out_arr = np.array([])
    for arr in out_list.__reversed__():
        out_arr = np.append(out_arr, arr, axis=0)
    return out_arr


if __name__ == '__main__':
    arr = np.arange(20)
    reduced_arr = partitioned_reduce_mean(arr, [1, 2, 8])
    print(reduced_arr)