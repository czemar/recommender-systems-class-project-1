# Getting started

  ## Installation

  This package is written in Python 3.9.12. Packages are forked from [this GitHub repository](https://github.com/PiotrZiolo/recommender-systems-class) from university classes. This project was focused on making content based recommender for hotel, to recommend best items to user.

  ### Install dependencies

  All dependencies with corresponding versions are listed in `requirements.txt`. To install them all at once you can use the following command:

  ```bash
  pip install -r requirements.txt
  ```

  ### Run project in jupyter notebook

  Project is run in a Jupyter notebook. To run it you can use the following command:

  ```bash
  jupyter notebook
  ```

  You can also run the project in Visual Studio Code using jupyter extension. This extension is available in the [VSCode marketplace](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

  ### Generating `.html` files from jupyter notebook

  To generate `.html` files from jupyter notebook you can use the following commands:
  ```bash
   jupyter nbconvert --to html project_1_data_preparation.ipynb
   jupyter nbconvert --to html project_1_recommender_and_evaluation.ipynb
  ```

  Project is separated into two notebooks. First one is used to prepare data for recommender. Second one is used to train and evaluate recommender. So it is needed to run both notebooks to get the results.

# Results

  ## Linear Regression model
  Using linear regression model I was able to achieve score:

  <img src="docs/linear-regression-best-result.png" width="100%">

  Using tuned parameters:
  ```yml
  n_neg_per_pos: 9
  ```

  ## Random Forest model
  Using random forest model I was able to achieve score:

  <img src="docs/random-forest-best-result.png" width="100%">

  Using tuned parameters:
  ```yml
  n_neg_per_pos: 9
  n_estimators: 255
  max_depth: 8
  min_samples_split: 18
  ```
