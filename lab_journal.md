bcn 2018-08-16
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Half of the image is not really usable because we lose resolution at the
far end. I think it will get better when I take less of the frame for
the perspective transform

I want more of the good portion of the frame (the lower one) to count
for the fit

But maybe look at it after transforming back first

*Correction*: I had to make the destination rectangle simply bigger!

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
I reduced the margin of the windows to 15 to filter out more of the
noise coming from shadows.

I can clearly see that the assumptions for the perspective transform are
not fully working. As the car goes over humps, the lines go from an
opening angle to parallel to a closing angle.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Increased again to 30 as other parts of the movie got jumpy with 15

bcn 2018-08-17
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
I had to introduce a counter for the number of unsane pictures to
prevent deadlocks.
