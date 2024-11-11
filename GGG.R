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

nn_recipe <- recipe(type ~ ., data = data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate(color = factor(color)) %>%  
  step_dummy(color) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1)  


nn_model <- mlp(hidden_units = tune(), epochs = 50) %>%
  set_engine("keras", verbose = 0) %>%
  set_mode("classification")


nn_workflow <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)


set.seed(123) 
folds <- vfold_cv(data, v = 5)


maxHiddenUnits <- 10  
nn_tuneGrid <- grid_regular(hidden_units(range = c(1, maxHiddenUnits)), levels = 5)


tuned_nn <- nn_workflow %>%
  tune_grid(
    resamples = folds,
    grid = nn_tuneGrid,
    metrics = metric_set(accuracy)
  )


tuned_nn %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  labs(title = "Cross-Validation Results for Hidden Units vs. Accuracy",
       x = "Number of Hidden Units",
       y = "Mean Accuracy")


best_params <- select_best(tuned_nn, metric = "accuracy")


final_nn_workflow <- finalize_workflow(nn_workflow, best_params)

# Fit the final model to the entire training data
final_nn_fit <- fit(final_nn_workflow, data = data)

# Make predictions on the test data
predictions <- predict(final_nn_fit, new_data = test, type = "class")

# Prepare Kaggle submission
kaggle <- predictions %>%
  bind_cols(test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x = kaggle, file = "./MLP.csv", delim = ",")
