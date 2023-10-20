# Pitch

This folder contains some basic options to create a football pitch in python. <br>

Based on inspirations I found online which include the work of FCPython and McKay Johns and the packages mplsoccer and floodlight I also created my own simple function to create a custom pitch. <br>

## My_Pitch.py

**create_custom_Pitch** (grasscol='#6aa84f', pitchcol = '#38761d', linecol='white', figs=(10.5, 6.5), l=105, w=65):<br>

**Parameters:** 

+ *grasscol (color)*: Color of the pitch within the pitch limits
+ *pitchcol (color)*: Color of the area around the pitch
+ *linecol (color)*: Color of lines
+ *figs (tuple of floats)*: figsize
+ *l (float/int)*: length of the pitch in meters (should match the dimensions of your data!)
+ *w (float/int)*: width of the pitch in meters (should match the dimensions of your data!)

**Returns:**

+ *pitch* (figure)
  

My own function used pitch dimensions based on the following image [1] in meters: <br>

<img src="https://github.com/DavidB1999/My_Football_Analytics/blob/main/Basics/Pitch/Fu%C3%9Fballfeld.png" width="600" />




## Credits

FCPython: https://fcpython.com/visualisation/drawing-pitchmap-adding-lines-circles-matplotlib <br>
Floodlight: https://floodlight.readthedocs.io/en/latest/index.html <br>
McKay Johns: https://www.youtube.com/watch?v=55k1mCRyd2k <br>
mplsoccer: https://mplsoccer.readthedocs.io/en/latest/index.html <br>
Friends of Tracking | Laurie Shaw: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

## References

[1] - https://de.wikipedia.org/wiki/Fu%C3%9Fballregeln <br>
