# Tune max_depth and min_child_weight on a wide list first

objective = "reg:linear"
seed = 100
n_estimators = 100
learning_rate = 0.1
gamma = 0.05
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.5
reg_lambda = 0.5
silent = False

cv_params = {'max_depth': [2,4,6,8], 'min_child_weight': [1,3,5,7]}

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
gbm.cv_results_, gbm.best_params_, gbm.best_score_

# Refine tuning of max_depth and min_child_weight 

cv_params = {'max_depth': [7,8,9], 'min_child_weight': [0.5,1,2]}

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
gbm.grid_scores_, gbm.best_params_, gbm.best_score_

# Tune gamma next

max_depth = 9
min_child_weight = 4

cv_params = {'gamma': [i/10.0 for i in range(1,8)]}

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
gbm.grid_scores_, gbm.best_params_, gbm.best_score_

# Try a fit

gamma = 0.6

xgb1 = xgb.XGBRegressor(
    objective = objective,
    seed = seed,
    n_estimators = 1000,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    learning_rate = learning_rate,
    gamma = gamma,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    silent = False
)

xgb1.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric = 'rmse', verbose = True)
