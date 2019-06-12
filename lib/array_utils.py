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


def ex_partitioned_reduce_mean(arr, index, part_size_list=[8, 96, 672, 568]):
    size = sum(part_size_list)
    assert arr.shape[0] >= size
    arr = arr[index:index+size, ...]
    part_list = [1] + part_size_list[:-1]
    return partitioned_reduce_mean(arr, part_list)


def get_seq_len(part_size_list):
    seq_len = ex_partitioned_reduce_mean(np.arange(sum(part_size_list)), 0, part_size_list).shape[0]
    return seq_len


def broadcast_last_dim(arr1d, num_broadcast):
    arr2d = np.expand_dims(arr1d, -1)
    arr2d = np.tile(arr2d, (1, num_broadcast))
    return arr2d


#%%
if __name__ == '__main__':
    arr = np.arange(40)
    # arr = arr.reshape(10, 2, 2)
    # arr = arr.reshape(20, 1, 2)
    arr = arr.reshape(40, 1, 1)
    reduced_arr = partitioned_reduce_mean(arr, [1, 2, 8])
    print(reduced_arr)

#%%
if __name__ == '__main__':
    if 0:
        arr = np.arange(40)
        part_size_list=[2, 8, 30]
    if 1:
        arr = np.arange(1344)
        part_size_list = [8, 96, 672, 568]
    reduced_arr = ex_partitioned_reduce_mean(arr, 10, part_size_list)
    print(reduced_arr)

    seq_len = get_seq_len(part_size_list)
    print(seq_len)
