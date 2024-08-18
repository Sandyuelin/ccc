# Chinese Character Classifier (to distinguish simplified and complex version)


### Purpose
- This is to classify the dataset where folders of images for each character include both complex and simplified version, so the final version can be for example:
```
folder of complex characters 于... 
folder of simplified chracters 於...
with the images corresponding the actual character
```

### Procedure
- preprocess the image
- create the image data using PIL
- extract the latents of both created data and existing data
- organize the folder by calculating the cosine similarity

