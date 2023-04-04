from hyperopt import  hp
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 300)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'learning_rate': hp.loguniform('learning_rate', -4, 0),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}