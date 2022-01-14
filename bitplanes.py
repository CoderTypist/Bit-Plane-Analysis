from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
    
    
class g:
    
    '''
    Description
    -----------
    Stores global configuration settings for the program.
    All functions modify and rely on these global settings.
    
    '''
    
    # save resulting figures to files
    SAVE = True
    # show the progress of the program
    VERBOSE = True
    
    # show the small moving/block averages
    SMALL = True
    # show the large moving/block averages
    LARGE = True
    # show the cumulative averages
    CUMULATIVE = True
    
    # size of the smaller window/block
    # - specifies the size of the moving window if SMALL_MOVING = True
    # - specifies the size of the blocks if SMALL_MOVING = False
    SMALL_WINDOW_SIZE = 100
    # size of the larger window/block
    # - specifies the size of the moving window if LARGE_MOVING = True
    # - specifies the size of the blocks if LARGE_MOVING = False
    LARGE_WINDOW_SIZE = 1000
    
    # whether to calculate moving or block averages for the smaller size
    # - moving averages are calculated if SMALL_MOVING = True
    # - block averages are calculated if SMALL_MOVING = False
    SMALL_MOVING = True
    # whether to calculate moving or block averages for the larger size
    # - moving averages are calculated if LARGE_MOVING = True
    # - block averages are calculated if LARGE_MOVING = False
    LARGE_MOVING = True


class Block:
    
    '''
    Description
    -----------
    Contains information regarding blocks of contiguous bits in a bit plane.
    
    '''
    
    def __init__(self, mean, std, start, size):
        
        '''
        Params
        ------
        mean: float
            The mean of bit values within the block (between 0 and 1).
            
        std: float
            The standard deviation of bit values within the block (between 0 and 1).
            
        start: int
            The index of the bit within the bit plane.
        
        size: int
            The number of bits within the block.

        '''
        
        self.mean = mean
        self.std = std
        self.start = start
        self.size = size


def get_red(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    
    '''
    Description
    -----------
        Returns a numpy array of the red value for each pixel.
    
    Params
    ------
        pix: List[Tuple[int,int,int]]
            The pixel values for the image.
    
    Returns
    -------
        np.ndarray[np.uint8]
            The red values for each pixel.
            Color values, which are ints, are typecasted to np.uint8 to allow for bitwise operations.
    
    '''
    
    # for each pixel, get the red value and typecast it to an unsigned 8-bit integer
    return np.array([ np.uint8(p[0]) for p in pix ])


def get_green(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    
    '''
    Description
    -----------
        Returns a numpy array of the green value for each pixel.
    
    Params
    ------
        pix: List[Tuple[int,int,int]]
            The pixel values for the image.
    
    Returns
    -------
        np.ndarray[np.uint8]
            The green values for each pixel.
            Color values, which are ints, are typecasted to np.uint8 to allow for bitwise operations.
    
    '''
    
    # for each pixel, get the green value and typecast it to an unsigned 8-bit integer
    return np.array([ np.uint8(p[1]) for p in pix ])


def get_blue(pix: List[Tuple[int,int,int]]) -> np.ndarray:
    
    '''
    Description
    -----------
        Returns a numpy array of the blue value for each pixel.
    
    Params
    ------
        pix: List[Tuple[int,int,int]]
            The pixel values for the image.
    
    Returns
    -------
        np.ndarray[np.uint8]
            The blue values for each pixel.
            Color values, which are ints, are typecasted to np.uint8 to allow for bitwise operations.
    
    '''
    
    # for each pixel, get the blue value and typecast it to an unsigned 8-bit integer
    return np.array([ np.uint8(p[2]) for p in pix ])


def get_color_bit_planes(color: np.ndarray) -> List[List[np.uint8]]:
    
    '''
    Description
    -----------
    Extracts bits from each bit plane.
    
    Params
    ------
    color: np.ndarray[np.uint8]
        Contains the values for a single color (i.e. red, green, or blue)
    
    Returns
    -------
    List[List[np.uint8]]
        The bits from each bit plane.
        
        The outter list contains 8 elements, where each element is a list 
            containing the values for a bit plane. 
        
    '''
    
    bit_planes = [ [] for i in np.arange(8) ]
    
    for c in color:
        bit_planes[7].append(1 if np.bitwise_and(c,128) else 0) # & 0x80
        bit_planes[6].append(1 if np.bitwise_and(c,64) else 0)  # & 0x40
        bit_planes[5].append(1 if np.bitwise_and(c,32) else 0)  # & 0x20
        bit_planes[4].append(1 if np.bitwise_and(c,16) else 0)  # & 0x10
        bit_planes[3].append(1 if np.bitwise_and(c,8) else 0)   # & 0x08
        bit_planes[2].append(1 if np.bitwise_and(c,4) else 0)   # & 0x04
        bit_planes[1].append(1 if np.bitwise_and(c,2) else 0)   # & 0x02
        bit_planes[0].append(np.bitwise_and(c,1))               # & 0x01
    
    return bit_planes


def get_bit_plane_moving_averages(bit_planes: List[List[np.uint8]], window_size) -> List[List[float]]:
    
    '''
    Description
    -----------
    Calculates the moving averages along each bit plane.
    
    Params
    ------
    bit_planes: List[List[np.uint8]]
        The bits from each bit plane.
        
        The outter list contains 8 elements, where each element is a list 
            containing the values for a bit plane.
    
    Returns
    -------
    List[List[float]]
        The moving averages from each bit plane.
        
        The outter list contains 8 elements, where each element is a list 
            containing the moving averages for a bit plane. 
    
    '''

    bit_plane_averages = [ [] for i in range(8) ]
    
    # output average type
    if g.VERBOSE:
        if 0 == window_size:
            print('\t- cumulative:')
        else:
            print(f'\t- window size {window_size}:')
        print('\t\t- ', end='')
    
    # loop over each bit plane
    for i_bp in np.arange(8):
        
        # output current bit plane index
        if g.VERBOSE: print(f'{i_bp} ', end='')
        # get bit plane
        bp = bit_planes[i_bp]
        # cumulative sum of bits (only used for the cumulative averages)
        bp_sum = 0
        
        # loop over each bit in the bit plane
        for i_val in np.arange(len(bp)):
            
            # add bit (0 or 1) to the sum
            bp_sum += bp[i_val]
            
            # cumulative averages
            if 0 == window_size:
                bit_plane_averages[i_bp].append(bp_sum/(i_val+1))
            
            # moving averages
            else:
                # starting index of the moving window
                i_start = i_val - window_size
                
                # towards the beginning, the window size will be smaller than the specified size
                # because of this, i_start needs to be set to 0 as the window grows
                if i_start <= 0:
                    i_start = i_val + 1
                
                # numpy indexing does not include the last number, so add 1
                i_end = i_val + 1
                # the size of the current window
                current_window_size = i_end - i_start
                # the average within the window
                bit_plane_averages[i_bp].append(np.sum(bp[i_start:i_end])/current_window_size)
                
    print()
    return bit_plane_averages


def get_bit_plane_block_averages(bit_planes: List[List[np.uint8]], blk_size) -> List[List[Block]]:
    
    '''
    Description
    -----------
    Calculates the block averages along each bit plane.
    
    Params
    ------
    bit_planes: List[List[np.uint8]]
        The bits from each bit plane.
        
        The outter list contains 8 elements, where each element is a list 
            containing the values for a bit plane.
    
    Returns
    -------
    List[List[Block]]
        Statistics for each block from each bit plane.
        Statistics for each block are stored within a Block object.
        
        The outter list contains 8 elements, where each element is a list 
            containing the block averages/stds (stored within Block objects) for a bit plane. 
        
    '''
    
    bit_plane_block_averages: List[List[Block]] = [ [] for i in range(8) ]
    
    # output average type
    if g.VERBOSE:
        if 0 == blk_size:
            print('\t- cumulative:')
        else:
            print(f'\t- block size {blk_size}:')
        print('\t\t- ', end='')
    
    # loop over each bit plane
    for i_bp in np.arange(8):
        
        # output current bit plane index
        if g.VERBOSE: print(f'{i_bp} ', end='')
        # get bit plane
        bp = bit_planes[i_bp]
        len_bp = len(bp)
        
        for i_start in np.arange(0, len_bp, blk_size):
            
            to_end = len_bp - i_start
            
            i_end = 0
            cur_blk_size = 0
            
            if to_end < blk_size:
                i_end = len_bp
                cur_blk_size = to_end
            else:
                i_end = i_start + blk_size
                cur_blk_size = blk_size
                
            blk = bp[i_start:i_end]
            mean = np.mean(blk)
            std = np.std(blk)
            bit_plane_block_averages[i_bp].append(Block(mean, std, i_start, cur_blk_size))
    
    print()
    return bit_plane_block_averages

    
def color_averages(color_vals: np.ndarray, color_name: str, base_name=None) -> None:
    
    '''
    Description
    -----------
    Graphs the averages along each bit plane for the received color.
    
    Whether block, moving, and/or cumulative averages are calculated is determined by the global settings.
    The global settings are set by calling functions.
    
    Params
    ------
    color_vals: np.ndarray[np.uint8]
        The color values for each pixel (for a single color).
    
    color_name: str
        The name of the color being analyzed (either 'Red', 'Green', or 'Blue')
    
    base_name: str = None
        Part of the filename leading up to the first '.'
        Used to construct the name of the output file containing the graph.
        
    Returns
    -------
    None
    
    '''

    bit_planes = get_color_bit_planes(color_vals)
    
    bit_plane_small_averages = None
    bit_plane_large_averages = None
    bit_plane_cumulative_averages = None
    
    if g.SMALL:
        if g.SMALL_MOVING: bit_plane_small_averages = get_bit_plane_moving_averages(bit_planes, g.SMALL_WINDOW_SIZE)
        else: bit_plane_small_averages = get_bit_plane_block_averages(bit_planes, g.SMALL_WINDOW_SIZE)
        
    if g.LARGE: 
        if g.LARGE_MOVING: bit_plane_large_averages = get_bit_plane_moving_averages(bit_planes, g.LARGE_WINDOW_SIZE)
        else: bit_plane_large_averages = get_bit_plane_block_averages(bit_planes, g.LARGE_WINDOW_SIZE)
        
    if g.CUMULATIVE: bit_plane_cumulative_averages = get_bit_plane_moving_averages(bit_planes, 0)
    
    fig, axs = plt.subplots(4,2, figsize=(15, 10), constrained_layout=True)
    
    title = f'{color_name} - Averages: '
    sizes = []
    if g.SMALL: sizes.append(g.SMALL_WINDOW_SIZE)
    if g.LARGE: sizes.append(g.LARGE_WINDOW_SIZE)
    if g.CUMULATIVE: sizes.append('cumulative')
    title += str(sizes)
    
    fig.suptitle(title, fontsize=30)
    axs_fontsize=25
    
    small_window_alpha=0.2
    large_window_alpha=0.4
    
    for i in np.arange(8):
        r = i % 4
        c = i // 4
        
        if g.LARGE:
            if g.LARGE_MOVING:
                axs[r,c].plot(bit_plane_large_averages[i], color='gray', alpha=large_window_alpha)
            else:
                for blk in bit_plane_large_averages[i]:  
                    
                    rect = Rectangle((blk.start, blk.mean-(blk.std/2)), 
                                     blk.size, blk.std, 
                                     facecolor='gray', edgecolor='black',
                                     alpha=large_window_alpha)
                    axs[r,c].add_patch(rect)
                    
                    axs[r,c].plot([blk.start, blk.start+blk.size],
                                  [blk.mean, blk.mean],
                                  linewidth=3, color='cyan')
        
        if g.SMALL: 
            if g.SMALL_MOVING:
                axs[r,c].plot(bit_plane_small_averages[i], color='gray', alpha=small_window_alpha)
            else:
                for blk in bit_plane_small_averages[i]:  
                    
                    rect = Rectangle((blk.start, blk.mean-(blk.std/2)), 
                                     blk.size, blk.std, 
                                     facecolor='gray', edgecolor='black',
                                     alpha=small_window_alpha)
                    axs[r,c].add_patch(rect)
                    
                    axs[r,c].plot([blk.start, blk.start+blk.size],
                                  [blk.mean, blk.mean],
                                  linewidth=3, color='orange')
        
        if g.CUMULATIVE: axs[r,c].plot(bit_plane_cumulative_averages[i], color=color_name)
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


def pix_averages(pix: List[Tuple[int,int,int]], base_name=None) -> None:
    
    '''
    Description
    -----------
    Creates figures showing the block/moving/cumulative averages along each bit plane for each color.
    
    Params
    ------
    pix: List[Tuple[int,int,int]]
        The pixel values for the image.
    
    base_name: str
        Part of the filename leading up to the first '.'
    
    Returns
    -------
    None
    
    '''

    if g.VERBOSE: print('RED:')
    color_averages(get_red(pix), 'Red', base_name=base_name)
    
    if g.VERBOSE: print('GREEN:')
    color_averages(get_green(pix), 'Green', base_name=base_name)
    
    if g.VERBOSE: print('BLUE:')
    color_averages(get_blue(pix), 'Blue', base_name=base_name)


def set_config(save=True, verbose=True,
               small=True, large=True, cumulative=True,
               small_window_size=100, large_window_size=1000,
               small_moving=True, large_moving=True):
    
    '''
    Description
    -----------
    Sets the global settings used by all functions.
    
    Params
    ------
    save: bool = True
        Save the resulting figures.
    
    verbose: bool = True
        Output program progress to the console.
    
    small: bool = True
        Calculate small moving/block averages.
    
    large: bool = True
        Calculate large moving/block averages.
    
    cumulative: bool = True
        Calculate cumulative averages.
    
    small_window_size: int = 100
        Size of the smaller window/block
        - specifies the size of the moving window if small_moving = True
        - specifies the size of the blocks if small_moving = False
    
    large_window_size: int = 1000
        Size of the larger window/block
        - specifies the size of the moving window if large_moving = True
        - specifies the size of the blocks if large_moving = False
    
    small_moving: bool = True
        Whether to calculate moving or block averages for the smaller size
        - moving averages are calculated if small_moving = True
        - block averages are calculated if small_moving = False
    
    large_moving: bool = True
        Whether to calculate moving or block averages for the larger size
        - moving averages are calculated if large_moving = True
        - block averages are calculated if large_moving = False
        
    Returns
    -------
        None
        
    '''
    
    g.SAVE = save
    g.VERBOSE = verbose
    g.SMALL = small
    g.LARGE = large
    g.CUMULATIVE = cumulative
    g.SMALL_WINDOW_SIZE = small_window_size
    g.LARGE_WINDOW_SIZE = large_window_size
    g.SMALL_MOVING = small_moving
    g.LARGE_MOVING = large_moving
        
    
def analyze(fname, save=True, verbose=True,
            small=True, large=True, cumulative=True,
            small_window_size=-1, large_window_size=-1,
            small_moving=False, large_moving=False):
    
    '''
    Description
    -----------
    Calculate and display the block/moving/cumulative averages along each bit plane for each color.
    
    Params
    ------
    fname: str
        Path to the image file.
    
    save: bool = True
        Save the resulting figures.
    
    verbose: bool = True
        Output program progress to the console.
    
    small: bool = True
        Calculate small moving/block averages.
    
    large: bool = True
        Calculate large moving/block averages.
    
    cumulative: bool = True
        Calculate cumulative averages.
    
    small_block_size: int = -1
        Size of the smaller blocks.
        - specifies the size of the moving window if large_moving = True
        - specifies the size of the blocks if large_moving = False
            - When set to -1, the size of the smaller blocks is
                 dynamically determined by analyze()
    
    large_block_size: int = -1
        Size of the larger blocks.
        - specifies the size of the moving window if large_moving = True
        - specifies the size of the blocks if large_moving = False
            - When set to -1, the size of the smaller blocks is
                 dynamically determined by analyze()
    
    small_moving: bool = True
        Whether to calculate moving or block averages for the smaller size
        - moving averages are calculated if small_moving = True
        - block averages are calculated if small_moving = False
    
    large_moving: bool = True
        Whether to calculate moving or block averages for the larger size
        - moving averages are calculated if large_moving = True
        - block averages are calculated if large_moving = False
    
    Returns
    -------
    None
    
    '''
    
    im = Image.open(fname)
    pix = list(im.getdata())
    
    if -1 == small_window_size:
        num_pixels = len(pix)
        small_window_size = num_pixels // 40
    
    if -1 == large_window_size:
        num_pixels = len(pix)
        large_window_size = num_pixels // 8
        
    set_config(save=save,
               verbose=verbose,
               small=small,
               large=large,
               cumulative=cumulative,
               small_window_size=small_window_size,
               large_window_size=large_window_size,
               small_moving=small_moving,
               large_moving=large_moving)
    
    base_name = fname.split('.')[0]
    pix_averages(pix, base_name=base_name)
    

def cumulative_averages(fname, save=True, verbose=True):
    
    '''
    Description
    -----------
    Calculate and display the cumulative averages along each bit plane for each color.
    
    Params
    ------
    fname: str
        Path to the image file.
    
    save: bool = True
        Save the resulting figures.
    
    verbose: bool = True
        Output program progress to the console.
    
    Returns
    -------
        None
        
    '''
    
    analyze(fname,
            save=save,
            small=False,
            large=False,
            cumulative=True)


def block_averages(fname, save=True, verbose=True,
                   small=True, large=True, cumulative=True,
                   small_block_size=-1, large_block_size=-1):
    
    '''
    Description
    -----------
    Calculate and display the block averages along each bit plane for each color.
    The height of each block in the resulting of the graph is the standard deviation within the block.
    
    Params
    ------
    fname: str
        Path to the image file.
    
    save: bool = True
        Save the resulting figures.
    
    verbose: bool = True
        Output program progress to the console.
    
    small: bool = True
        Calculate small block averages.
    
    large: bool = True
        Calculate large block averages.
    
    cumulative: bool = True
        Calculate cumulative averages.
    
    small_block_size: int = -1
        Size of the smaller blocks.
        When set to -1, the size of the smaller blocks is
            dynamically determined by analyze()
    
    large_block_size: int = -1
        Size of the larger blocks.
        When set to -1, the size of the larger blocks is
            dynamically determined by analyze()
    
    Returns
    -------
        None
        
    '''

    analyze(fname,
            save=save,
            small=small,
            large=large,
            cumulative=cumulative,
            small_window_size=small_block_size,
            large_window_size=large_block_size,
            small_moving=False, large_moving=False)


def moving_averages(fname, save=True, verbose=True,
                    small=True, large=True, cumulative=True,
                    small_window_size=100, large_window_size=1000):
    
    '''
    Description
    -----------
    Calculate and display the moving averages along each bit plane for each color.
    
    Params
    ------
    fname: str
        Path to the image file.
    
    save: bool = True
        Save the resulting figures.
    
    verbose: bool = True
        Output program progress to the console.
    
    small: bool = True
        Calculate small moving averages.
    
    large: bool = True
        Calculate large moving averages.
    
    cumulative: bool = True
        Calculate cumulative averages.
    
    small_window_size: int = 100
        Size of the smaller window.
    
    large_window_size: int = 1000
        Size of the larger window.
    
    Returns
    -------
        None
        
    '''
    
    analyze(fname,
            save=save,
            small=small,
            large=large,
            cumulative=cumulative,
            small_window_size=small_window_size,
            large_window_size=large_window_size,
            small_moving=True,
            large_moving=True)
