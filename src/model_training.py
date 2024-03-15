from starter import *
from convolutionmodel import * 
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from time import time, ctime
from tabulate import tabulate
import subprocess
del tqdm
from tqdm.auto import tqdm

# subprocess.run(["clear"]) 

def get_timestamp():
    return int(time())

def get_ctimestamp():
    return ctime(time())

def model_training(params, df, target, timestamp, 
                   horizon=0,
                   horizon_x4=6,
                   n_splits=5, 
                   num_epochs=5):
    
    kf = KFold(n_splits=n_splits)
    horizon = horizon_x4 if horizon<horizon_x4 else horizon 
    
    
    tmp = df.iloc[horizon:].reset_index(drop=True) # used to get the train and test index splits
    results = []

    # Loop over the dataset to create separate folds
    fold = 0
    
    for train_index, test_index in tqdm(kf.split(tmp), total=n_splits, desc='Folds'):
        fold +=1

        #################### getting the dataset ##############
             
        _, _, _, _, _, _, X4_train, X4_test, \
         _, _, \
        y_train, y_test = df_to_data(df,
                                                        target = target, 
                                                        horizon = horizon, 
                                                        horizon_x4 = horizon_x4,
                                                        train_index = train_index,
                                                        test_index = test_index)
        
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

        ###################### model training #################

        # callbacks
        earlystopping = callbacks.EarlyStopping(
            patience=30, 
            monitor='val_accuracy',
            restore_best_weights=True,
            verbose=1)

        # build and compile               
        model = ConvolutionModel(
                                    num_convolution_blocks = params['num_convolution_blocks'],
                                    num_dense_layers = params['num_dense_layers'],
                                    kernel = params['kernel'],
                                    pooling = params['pooling'])
        model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer='adamax',
                      metrics=['accuracy'])

        # fitting
        model.fit(X4_train, y_train, 
                  shuffle=True,
                  batch_size=32,
                  epochs=num_epochs, 
                  verbose=1, 
                  validation_data=[X4_test, y_test], 
                  validation_batch_size=128,
                  callbacks = [earlystopping]
                  )

    #     ###################### model evaluation #################

        result = model.evaluate(X4_test, y_test, verbose=0)
        results.append([fold, *result])

        break
    return results

def custom_grid_search(param_grid, 
                       df, target, timestamp,
                       horizon=0, 
                       horizon_x4=6, 
                       n_splits=5,
                       num_epochs=5):
    global glob_results 
    glob_results = []
    timestamp = get_timestamp()
    
    
    # Perform grid search
    for dict_ in tqdm(param_grid, position=0, leave=True, desc='Grid Search'):
        
        kernel, num_convolution_blocks, num_dense_layers, pooling = dict_.values()
        params = {
            'num_convolution_blocks': num_convolution_blocks,
            'num_dense_layers': num_dense_layers,
            'kernel': kernel,
            'pooling': pooling}
        
        # Call model training with the current parameters
        accuracy_results = model_training(params, 
                                          df, 
                                          target,  
                                          timestamp,
                                          horizon, 
                                          horizon_x4, 
                                          n_splits,
                                          num_epochs)

        # Save the results
        for result in accuracy_results:
            fold, accuracy, loss = result

            result_entry = {
                'num_convolution_blocks': num_convolution_blocks,
                'num_dense_layers': num_dense_layers,
                'kernel': kernel,
                'pooling': pooling,
                'fold': fold,
                'accuracy': accuracy,
                'loss': loss
            }
            glob_results.append(result_entry)              

    return glob_results

def main():
    param_grid = ParameterGrid(
                            {
                            'num_convolution_blocks': [1],
                            'num_dense_layers': [1, 3], 
                            'kernel': [(3, 3), (6, 6)], 
                            'pooling': [True]
                            })
    n_splits = 5  
    horizon = 0
    horizon_x4 = 10  
    target = 'WB 1' 
    timestamp = get_timestamp()
    
    print(get_ctimestamp(),":\tStarted Grid Search")
    grid_search_results = custom_grid_search(
        param_grid, df, target, timestamp, 
        horizon, horizon_x4, n_splits, num_epochs=1)
    print(get_ctimestamp(),":\tEnded Grid Search")
    
    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(grid_search_results)
    tabulate(results_df, headers='keys', tablefmt='psql')
    results_df.to_csv(f'output_data_tmp/grid_search_results_{get_timestamp()}.csv', index=False)


if __name__ == "__main__":
    
    try:
        main()
    except KeyboardInterrupt:
        try:
            print (get_ctimestamp(),':\tKeyboard Interruption')
            results_df = pd.DataFrame(glob_results)
            tabulate(results_df, headers='keys', tablefmt='psql')
            results_df.to_csv(f'output_data_tmp/grid_search_results_interrupted_{get_timestamp()}.csv', index=False)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(0)
        

