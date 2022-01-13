# Bit-Plane-Analysis
---
### Description
Calculate block, moving, or cumulative averages along each bit plane for each color (r,g,b). Averages will be between 0 and 1 since each bit plane consists of 0's and 1's. 

---
### Example
```
import bitplanes as bp

# (recommendeed)
# calculate the averages within contiguous segments (a.k.a blocks) of the image
bp.block_averages('some_image.png')

# (not recommended, SLOW)
# calculate the moving averages
bp.moving_averages('some_image.png')

# calculate the cumulative averages
bp.cumulative_averages('some_image.png')
```

---
### Note
It is recommended to use __block_averages()__ instead of __moving_averages()__. Block averages are quick. Moving averages can take a LONG time.  __block_averages()__ will divide images into segments and calculate the average for each segment. If using a window/block size of 1000, calculating block averages will be 1000 times faster than calculating the moving averages. 

The cumulative average is shown by the solid colored line.
The moving average with a window size of 100 is shown in light gray.
The moving average with a window size of 1000 is shown in dark gray.

---
### Example Output - Block Averages
![Block Averages Along Red Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_red_51840_259200_cumulative.png)
![Block Averages Along Green Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_green_51840_259200_cumulative.png)
![Block Averages Along Blue Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_blue_51840_259200_cumulative.png)

---
### Example Output - Moving Averages
![Moving Averages Along Red Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_red_100_1000_cumulative.png)
![Moving Averages Along Green Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_green_100_1000_cumulative.png)
![Moving Averages Along Blue Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/figures/results_pikachu_blue_51840_259200_cumulative.png)

---
### Motivation
Each color in an image serves as a channel where information can be hidden. Furthermore, each bit plane serves as a channel for hiding information. Since there are 3 different colors and 8 bitplanes per color, there are 24 different channels in which information can be hidden. The moving average is graphed in addition to the cumulative average since information can be hidden anywhere along a bitplane. This program is meant to evaluate the effectiveness of various steganography algorithms and to analyze images that are suspected of containing embedded messages. 

---
## Usage
* Use __bp.block_averages()__ to calculate block averages.
* Use __bp.moving_averages()__ to calculate moving averages (this is __SLOW__).
* Use __bp.cumulative_averages()__ to calculate cumulative averages.
* Only use __bp.analyze()__ for advanced usage.

---
### __bp.block_averages()__
##### Default Setings
```
bp.block_averages('some_image.png')
```
##### Disable Small Blocks
```
bp.block_averages('some_image.png', small=False)
```
##### Disable Large Blocks
```
bp.block_averages('some_image.png', large=False)
```
##### Adjust Block Sizes
```
bp.block_averages('some_image.png', small_block_size=1000, large_block_size=10000)
```
##### Do not Save Figures
```
bp.block_averages('some_image.png', save=False)
```

---
### bp.moving_averages()
##### Default Setings
```
bp.moving_averages('some_image.png')
```
##### Disable Small Blocks
```
bp.moving_averages('some_image.png', small=False)
```
##### Disable Large Blocks
```
bp.moving_averages('some_image.png', large=False)
```
##### Adjust Block Sizes
```
bp.moving_averages('some_image.png', small_block_size=1000, large_block_size=10000)
```
##### Do not Save Figures
```
bp.moving_averages('some_image.png', save=False)
```
---
### bp.cumulative_averages()
##### Only show the cumulative averages
```
bp.cumulative_averages('some_image.png')
```

---
### bp.analyze()
Meant for advanced usage. With __bp.analyze()__, it is possible to show _both_ block and moving averages. __bp.block_averages()__ and __bp.moving_averages()__ are friendly interfaces that call __bp.analyze()__

##### Small Moving Averages and Large Block Averages
```
# By setting large_moving to False, block averages will be used instead
# By setting large_window_size to -1, the window size is dynamically determined
bp.analyze('some_image.png',
            large_moving=False, small_window_size=-1))
```
##### Small Block Averages and Large Moving Averages
```
# By setting small_moving to False, block averages will be used instead
# By setting small_window_size to -1, the window size is dynamically determined
bp.analyze('some_image.png',
            small_moving=False, small_window_size=-1)
```
