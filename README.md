chinese_character_classifier/
├── data/
│   ├── raw/
│   │   └── example_image.jpg
│   ├── processed/
│   └── fonts/
│       └── standard_font.ttf
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── render.py
│   ├── encode.py
│   ├── cluster.py
│   ├── classify.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_render.py
│   ├── test_encode.py
│   └── test_classify.py
├── requirements.txt
├── README.md
└── .gitignore

data/:

raw/: Contains raw input images.
processed/: Contains preprocessed images.
fonts/: Contains standard fonts used for rendering characters.
src/:

preprocess.py: Contains functions to preprocess images (convert to grayscale, thresholding, etc.).
render.py: Contains functions to render standard characters using PIL.
encode.py: Contains functions to encode images using different encoders (CLIP, Inception-v3, VGG-16).
cluster.py: Contains functions for clustering encoded features using t-SNE.
classify.py: Contains functions to classify characters based on cosine similarity.
utils.py: Contains utility functions that are used across the project.
notebooks/:

exploration.ipynb: Jupyter notebook for data exploration and testing ideas interactively.
tests/:

test_preprocess.py: Contains unit tests for the preprocess module.
test_render.py: Contains unit tests for the render module.
test_encode.py: Contains unit tests for the encode module.
test_classify.py: Contains unit tests for the classify module.
requirements.txt: Lists all dependencies required for the project.

README.md: Provides an overview of the project, how to set it up, and how to use it.

.gitignore: Specifies which files and directories to ignore in version control (e.g., __pycache__/, *.pyc).