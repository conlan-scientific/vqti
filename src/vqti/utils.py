import numpy as np
def print_histogram(values, preset_bins=None, density=False):

    bins = 10
    block_char = '\u2591'
    start_char = '\u2595'
    max_block_length = 64

    if isinstance(values, pd.Series):
        values = values.values

    elif isinstance(values, np.ndarray):
        pass

    elif isinstance(values, list):
        values = np.array(values)

    else:
        raise TypeError('Need list, np.ndarray, or pd.Series ...')

    all_equal = np.all(values == values[0])
    if all_equal:
        x_const = values[0]
        bins = 1

    if preset_bins is None:
        hist_y, hist_x = np.histogram(values, bins=bins, density=density)
    else:
        hist_y, hist_x = np.histogram(values, bins=preset_bins, density=density)

    max_y = np.max(hist_y)

    hist_str = ''
    
    for i in range(hist_y.shape[0]):
        y = hist_y[i]
        y0 = f'{y:<8}'
        x0 = f'{hist_x[i]:<8.1f}'
        x1 = f'{hist_x[i+1]:<8.1f}'
        n_blocks = int(round(max_block_length * (y / max_y)))
        block_str = n_blocks * block_char
        block_str += (max_block_length - len(block_str)) * ' '

        if not all_equal:
            x_str = f'[{x0} - {x1}]'
        else:
            x_str = f'[Constant:  {x_const:<8.1f}]'

        # print(f'{start_char}{block_str} {y0} {x_str}')
        hist_str += f'{start_char}{block_str} {y0} {x_str} \n'
    
    # print()
    return hist_str


if __name__ == '__main__':
    print_histogram(np.arange(1, 100))

    

