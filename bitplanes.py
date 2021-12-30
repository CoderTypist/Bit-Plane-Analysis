import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import List, Tuple


class g:
    
    SAVE = True
    VERBOSE = True
    
    SMALL = True
    LARGE = True
    CUMULATIVE = True
    
    SMALL_WINDOW_SIZE = 100
    LARGE_WINDOW_SIZE = 1000


def get_red(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    return np.array([ np.uint8(p[0]) for p in pix ])


def get_green(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    return np.array([ np.uint8(p[1]) for p in pix ])


def get_blue(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    return np.array([ np.uint8(p[2]) for p in pix ])


def get_color_bit_planes(color: np.ndarray) -> List[List[np.uint8]]:
    
    bit_planes = [ [] for i in np.arange(8) ]
    
    for c in color:
        bit_planes[7].append(1 if np.bitwise_and(c,128) else 0)
        bit_planes[6].append(1 if np.bitwise_and(c,64) else 0)
        bit_planes[5].append(1 if np.bitwise_and(c,32) else 0)
        bit_planes[4].append(1 if np.bitwise_and(c,16) else 0)
        bit_planes[3].append(1 if np.bitwise_and(c,8) else 0)
        bit_planes[2].append(1 if np.bitwise_and(c,4) else 0)
        bit_planes[1].append(1 if np.bitwise_and(c,2) else 0)
        bit_planes[0].append(np.bitwise_and(c,1))
    
    return bit_planes


def get_bit_plane_percentages(bit_planes: List[List[np.uint8]], window_size=100) -> List[float]:
    
    bit_plane_percentages = [ [] for i in range(8) ]
    
    if g.VERBOSE:
        if 0 == window_size:
            print('\t- cumulative:')
        else:
            print(f'\t- window size {window_size}:')
        print('\t\t- ', end='')
            
    for i_bp in np.arange(len(bit_planes)):
        
        if g.VERBOSE: print(f'{i_bp} ', end='')
        bp = bit_planes[i_bp]
        bp_sum = 0
        
        for i_val in np.arange(len(bp)):
            bp_sum += bp[i_val]
            
            # cumulative percentages
            if 0 == window_size:
                bit_plane_percentages[i_bp].append(bp_sum/(i_val+1))
            
            # percentage within the window
            else:
                i_start = i_val - window_size
                
                if i_start <= 0:
                    i_start = i_val + 1
                    
                i_end = i_val + 2
                current_window_size = (i_end-1) - i_start
                bit_plane_percentages[i_bp].append(np.sum(bp[i_start:i_end])/current_window_size)
                
    print()
    return bit_plane_percentages
    
    
def color_percentages(color_vals, color_name, base_name=None) -> None:

    bit_planes = get_color_bit_planes(color_vals)
    
    bit_plane_small_window_percentages = None
    bit_plane_large_window_percentages = None
    bit_plane_cumulative_percentages = None
    
    if g.SMALL: bit_plane_small_window_percentages = get_bit_plane_percentages(bit_planes, window_size=g.SMALL_WINDOW_SIZE)
    if g.LARGE: bit_plane_large_window_percentages = get_bit_plane_percentages(bit_planes, window_size=g.LARGE_WINDOW_SIZE)
    if g.CUMULATIVE: bit_plane_cumulative_percentages = get_bit_plane_percentages(bit_planes, window_size=0)
    
    fig, axs = plt.subplots(4,2, figsize=(15, 10), constrained_layout=True)
    
    title = f'{color_name} - Averages: '
    windows = []
    if g.SMALL: windows.append(g.SMALL_WINDOW_SIZE)
    if g.LARGE: windows.append(g.LARGE_WINDOW_SIZE)
    if g.CUMULATIVE: windows.append('cumulative')
    title += str(windows)
    
    fig.suptitle(title, fontsize=30)
    axs_fontsize=25
    
    small_window_alpha=0.2
    large_window_alpha=0.4
    
    for i in np.arange(8):
        r = i % 4
        c = i // 4
        if g.SMALL: axs[r,c].plot(bit_plane_small_window_percentages[i], color='gray', alpha=small_window_alpha)
        if g.LARGE: axs[r,c].plot(bit_plane_large_window_percentages[i], color='gray', alpha=large_window_alpha)
        if g.CUMULATIVE: axs[r,c].plot(bit_plane_cumulative_percentages[i], color=color_name)
        axs[r,c].set_title(f'bit plane {i}', fontsize=axs_fontsize)
        axs[r,c].set_xlabel('Number of Bytes')
        axs[r,c].set_ylabel('Average')
    
    if g.SAVE:
        fname = 'results'
        if base_name: fname += f'_{base_name}'
        fname += f'_{color_name.lower()}'
        if g.SMALL: fname += f'_{g.SMALL_WINDOW_SIZE}'
        if g.LARGE: fname += f'_{g.LARGE_WINDOW_SIZE}'
        if g.CUMULATIVE: fname += '_cumulative'
        fname += '.png'
        plt.savefig(fname)
    
    plt.show()


def pix_percentages(pix, base_name=None):

    if g.VERBOSE: print('RED:')
    color_percentages(get_red(pix), 'Red', base_name=base_name)
    
    if g.VERBOSE: print('GREEN:')
    color_percentages(get_green(pix), 'Green', base_name=base_name)
    
    if g.VERBOSE: print('BLUE:')
    color_percentages(get_blue(pix), 'Blue', base_name=base_name)


def set_config(save=True, verbose=True,small=True, large=True, cumulative=True, small_window_size=100, large_window_size=1000):
     g.SAVE = save
     g.VERBOSE = verbose
     g.SMALL = small
     g.LARGE = large
     g.CUMULATIVE = cumulative
     g.SMALL_WINDOW_SIZE = small_window_size
     g.LARGE_WINDOW_SIZE = large_window_size
        
    
def analyze(fname, save=True, verbose=True, small=True, large=True, cumulative=True, small_window_size=100, large_window_size=1000):
    
    set_config(save=save,
               verbose=verbose,
               small=small,
               large=large,
               cumulative=cumulative,
               small_window_size=small_window_size,
               large_window_size=large_window_size)
    
    base_name = fname.split('.')[0]
    
    im = Image.open(fname)
    pix = list(im.getdata())
    pix_percentages(pix, base_name=base_name)
