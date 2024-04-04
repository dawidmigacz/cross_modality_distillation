import optuna
from train_all import main
import argparse

def objective(trial):
    # Define hyperparameters using trial object
    dropout = round(trial.suggest_uniform('dropout', 0.0, 0.0), 3)
    dropblock_prob = round(trial.suggest_uniform('dropblock_prob', 0.0, 0.0), 3)
    dropblock_size = trial.suggest_int('dropblock_size', 5, 5, step=2)
    temp = trial.suggest_categorical('temp', [1])
    distillation_weight = round(trial.suggest_uniform('distillation_weight_value', 0.0, 0.0), 3)
    model = trial.suggest_categorical('model', ['trn18'])

    # Create argparse Namespace object
    args = argparse.Namespace(
        dropout=dropout,
        dropblock_prob=dropblock_prob,
        dropblock_size=dropblock_size,
        temp=temp,
        distillation_weight=distillation_weight,
        model=model,
        resol=224  # This can also be a hyperparameter if you want
    )

    # Call your main function with these hyperparameters
    # Replace this with the function that trains your model
    acc_val, acc_epoch = main(args)

    # Return the metric you want to optimize
    return acc_val

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=33)

# Print the result
best_params = study.best_params
best_value = study.best_value
print(f'Best parameters: {best_params}\nBest accuracy: {best_value}')