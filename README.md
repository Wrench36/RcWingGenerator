# RcWingGenerator
Automatically generate Gcode for the Tom Stanton Rc Wing 3D Printing Method

Firstly, This project takes it's inspiration from Tom Stanton's 3D printed wing method.
Tom Stanton's video showing his CAD based method:

https://www.youtube.com/watch?v=QJjhMan6T_E

I can't say for certain if he created it, or cited a source.

This project takes the CAD aspect out of the equasion and directly generates the same geometry and gcode from some basic design specifications and 3D printer settings.
Note: Most of the code was written by ChatGPT 5, and debugged in combination with CGPT and myself.

The UI:
![UI](Images/Screenshot_20260606_100959.png)

Example output preview:
![UI](Images/Screenshot_20260606_101322.png)

Dependencies:
    pip install PyQt6 matplotlib numpy

The UI consists of two panels, Left and Right. The left panel is the input panel, and the right is the matPlotLib display.

Input Panel
Broken into 
- Printer
- Airfoil Section
- Spars
- Ribbing
- Wing Profile
- Save/Load/Export

Printer
- Typical print settings similar to any slicer.
- Note: As this is currently an engineering Alpha, the validity of these settings is yet to be confirmed.

Airfoil Section
- Source Type: NACA 4-digit or local .dat
- Source: enter 4-digit NACA profile (defualt 4412) or select local .dat

Spars
- Confogurable table for spars
- Enter placement along Chord % if making a model to match the wing
- Enter LE distance if matching an existing model. Measure form leading edge to spar center (be sure to make a 1 layer test print to verify)
- Diameter: diameter in MM of the hole. This is the nozzle-center diameter, so be sure to add some buffer with a test print.

Ribbing
- This is where the real magic happens. The generator will create ribs at the given angle to the chord line (or 1/2 line width/layer, whichever is less)
- Rib Spacing: Distane between ribs in mm
- Rib Angle: Angle from the chord to the rib
- Family A: Ribs that run from leading edge to tail as layers increase (pointed rearward)
- Family B: Ribs that run from trailing edge to lead as layers increase (poiting forward)

Wing Profile
- A table of sections to print
- Span start of the section
- Chord: Chord length of the section (dimention critical chords will require a test print as they are nozzle-centered)
- Layers interpolate between chord lengths between two given spans, so two spans with the same chord will form a straight section, different chords will produce a tapered section

Save/Load/Export
- Origin X: X position of the leading edge centerpoint
- Origin Y: Y position of the leading edge centerpoint
- Z Offset: Offset from bed
- Output file: Default filename
- Save Setup: Saves a json file of the currently configured wing
- Load Setup: Loads a json file (currently partially implemented)
- Export G-code: Export a 3D printable G-code file
- Export X-mirrored: Exports a G-code file with X axis mirrored, for printing starboard wings.

Design note:
- Use Spar placement option Chord mm if printing a wing in multiple files to be sure spar pockets will line up. The entered value should be the measurement from the leading edge to the center of the pocket from the end ofthe previous section.


Update:
- I've refactored how the drawing process works to simplify and squash a lot of bugs. Rather than tryign to draw the perimeters of each object with adds and subtracts, the codebase now draws primitives for each object (top and bottom, inner and outer, ribs, spars, spar webs) and traces around them. This means that all the complicated add/subtract operations are done much more reliably, and the inner/outer rib features now function properly. This is the _area_walk version.
- Added primitive checkboxes to viewer
- Added navigation toolbar to the viewer
- This version should have working json loading
- Printer options now have X and Y size, and an auto center button in the Gcode section to fix the annoying centering issues
- Mirror option goes along bounding box center, so the same centering values can be used for port/starboard wings



Todo:
- automatic section splitting for Z height max
- General debug and updates from feedback
