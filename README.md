# What is BCDH Toolbox

The BCDH Toolbox is a Python script which can be used in Agisoft Metashape. This code was tested in Version 2.2.1. 
There are two main categories - Analysis and Profile. In Analysis you are able to plot Histograms where you are able to deeper
understand the quality of your project. Metashape by itself only shows numeric values, but in terms of quality inspection for bundle adjustment, you want
a distribution of your data. In Profile you are able to generate and export profiles, which can be used directly in a GIS. The GeoTiff stores information about 
the height - this means: in GIS your Y-value becomes your Z-value. 

# How to use this code

You can run the script in Metashape like any other python scripts. But this script also uses matplotlib, which means you have to install
it beforehand. Just import pip and install matplotlib with this code. Sometimes it can take a few minutes.

```
import pip
pip.main(['install', 'matplotlib'])
```

If you are interested in using this script on start - please read [here](https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start).

# Functions

| Functions| Description |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Analysis: Error Report									| generates an error report in the python console                                                          |
| Analysis: Histogram Reprojection Error					| plots a histogram for reprojection error                                                               |
| Analysis: Histogram Reprojection Error [px]				| plots a histogram for reprojection error in pixel                                                      |
| Analysis: Histogram Key Point Size (Projection accuracy)	| plots a histogram for projection accuracy                                                              |
| Analysis: Histogram Image count							| plots a histogram for image count                                                                      |
| Analysis: Export Projections as CSV						| export all valid projections to a CSV file for own analysis                                             |
| Profile:  Build and Export Marker-Based Profile			| build and export Ortho-image based on defined markers 												  |
| Profile:  Export projected Marker from Ortho				| export markers with corresponding projection based on defined markers									  |
