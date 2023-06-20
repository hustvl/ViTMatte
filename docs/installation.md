## Installation

* Create a conda virtual env and activate it.

  ```
  conda create -n ViTMatte python==3.8.8
  conda activate ViTMatte
  ```
* Install packages.

  ```
  cd path/to/ViTMatte
  pip install -r requriments.txt
  ```
* Install [detectron2](https://github.com/facebookresearch/detectron2) , follow its [documentation](https://detectron2.readthedocs.io/en/latest/).
  For ViTMatte, we recommend to build it from latest source code.
  ```
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```