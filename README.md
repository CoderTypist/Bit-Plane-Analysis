# Bit-Plane-Analysis

### Description
Show the moving average and cummulative average along each bit plane for each color (r,g,b). Averages will be between 0 and 1 since each bit plane consists of 0's and 1's.

### Usage
```
import bitplanes as bp
bp.analyze('some_image.png')
```

### Example Output
The cumulative average is shown by the solid colored line.
The moving average with a window size of 100 is shown in light gray.
The moving average with a window size of 1000 is shown in dark gray.

![Averages Along Red Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/results_pikachu_red_10_100_cumulative.png)
![Averages Along Green Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/results_pikachu_green_10_100_cumulative.png)
![Averages Along Blue Bit Planes](https://github.com/CoderTypist/Bit-Plane-Analysis/blob/main/results_pikachu_blue_10_100_cumulative.png)

### Disclaimer
Analysis is painstakingly slow for larger images. For example, analysis is fast for a 100 KB image, but takes notably longer for a 1.5 MB image. 

### Adjusting Default Configuration

##### Only plot the cumulative average
```
bp.analyze('some_image.png', small=False, large=False)
```

##### Do no save figures to files
```
bp.analyze('some_image.png', save=False)
```

##### Adjust size of smaller moving window
```
bp.analyze('some_image.png', small_window_size=20)
```
##### Adjust size of the larger moving window
```
bp.analyze('some_image.png', large_window_size=500)
```

### Motivation
Each color in an image serves as a channel where information can be hidden. Furthermore, each bit plane serves as a channel for hiding information. Since there are 3 different colors and 8 bitplanes per color, there are 24 different channels in which information can be hidden. The moving average is graphed in addition to the cumulative average since information can be hidden anywhere along a bitplane. This program is meant to evaluate the effectiveness of various steganography algorithms and to analyze images that are suspected of containing embedded messages. 
