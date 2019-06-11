import numpy as np


def every_n_reduce_mean(arr, n=2):
    remainder = arr.shape[0] % n
    if remainder:
        arr = arr[remainder:, ...]
        print('Warning: dropped remainder of {} samples.'.format(remainder))
    ex_arr = np.expand_dims(arr, axis=1)
    s = list(ex_arr.shape)
    s[0] = s[0] // n
    s[1] = n
    return np.mean(np.reshape(ex_arr, tuple(s)), axis=1, keepdims=False)


def partitioned_reduce_mean(arr, part_list=[1, 8, 96, 672], rest_to_one=True):
    out_list = []
    for i in range(len(part_list)):
        if i + 1 < len(part_list):
            part_arr = arr[-part_list[i+1]:, ...]
            reduced_part_arr = every_n_reduce_mean(part_arr, part_list[i])
            arr = arr[:-part_list[i+1], ...]
        else:
            if rest_to_one:
                reduced_part_arr = np.mean(arr, axis=0, keepdims=True)
            else:
                reduced_part_arr = every_n_reduce_mean(arr, part_list[i])
        out_list.append(reduced_part_arr)
    out_arr = None
    for arr in out_list.__reversed__():
        if out_arr is None:
            out_arr = arr
        else:
            out_arr = np.append(out_arr, arr, axis=0)
    return out_arr


if __name__ == '__main__':
    arr = np.arange(40)
    # arr = arr.reshape(10, 2, 2)
    # arr = arr.reshape(20, 1, 2)
    arr = arr.reshape(40, 1, 1)
    reduced_arr = partitioned_reduce_mean(arr, [1, 2, 8])
    print(reduced_arr)