library(tidyverse)
library(tidymodels)
library(vroom)

data <- vroom("train.csv")
test <- vroom("test.csv")

# Make a recipe
data_recipe <- recipe(type ~ ., data = data) |>
  step_mutate_at(color, fn = factor)


# Create an SVM model specification
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |>
  set_mode("classification") |>
  set_engine("kernlab")

# Create a workflow with the recipe and model
svm_workflow <- workflow() |>
  add_recipe(data_recipe) |>
  add_model(svm_spec)

# Set up cross-validation
set.seed(123)  # For reproducibility
folds <- vfold_cv(data, v = 5)

# Set up a tuning grid
svm_grid <- grid_regular(cost(), rbf_sigma(), levels = 5)

# Tune the model using cross-validation
svm_tune_results <- tune_grid(
  svm_workflow,
  resamples = folds,
  grid = svm_grid,
  metrics = metric_set(accuracy)
)

# Select the best model based on accuracy

best_params <- select_best(svm_tune_results, metric = "accuracy")


# Finalize the workflow with the best hyperparameters
final_svm_workflow <- finalize_workflow(svm_workflow, best_params)

# Fit the final model to the entire training data
final_svm_fit <- fit(final_svm_workflow, data = data)

# Make predictions on the test data
predictions <- predict(final_svm_fit, new_data = test, type = "class")

kaggle <- predictions|>
  bind_cols(test) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

vroom_write(x= kaggle, file = "./SVM.csv", delim = ",")