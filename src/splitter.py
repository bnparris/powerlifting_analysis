from sklearn.model_selection import BaseCrossValidator
import numpy as np
class YearBasedTimeSeriesSplit(BaseCrossValidator):
    #see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py
    # inheriting from BaseCrossValidator to ensure compatibility with GridSearchCV
    
    def __init__(self, year_column='Year', exclude_val_years = None):
        self.year_column = year_column
        self.exclude_val_years = exclude_val_years or ()

    def split(self, X, y=None, groups=None):
        #needed to overwrite split method in parent class to make sure splits are time aware
        #otherwise the training rows are just all rows that are not validation rows which would create leakage in hyperparam tuning
        
        years = sorted(X[self.year_column].unique())
        
        for i in range(1, len(years)):
            val_year = years[i]
            if val_year in self.exclude_val_years:
                continue
            train_years = years[:i]
            train_idx = np.where(X[self.year_column].isin(train_years))[0]
            val_idx = np.where(X[self.year_column] == val_year)[0]

            #parent class uses yield in implementation of split() so will use here
            yield train_idx, val_idx
        
    def _iter_test_indices(self, X=None, y=None, groups=None):
        #BaseCrossValidator expects _iter_test_indices or _iter_test_masks
        #but since split had to be overwritten, logic for _iter_test_indices is same as for split().
        #(usually split calls _iter_test_masks which calls _iter_test_indices but here it makes more sense to call split for _iter_test_indices)
        for _, val_idx in self.split(X):
            yield val_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        #returns the number of folds the splitter will generate. 
        years = X[self.year_column].unique()
        #checks which years in excluded_years are actually present in the dataset
        excl_length = len([y for y in years if y in self.exclude_val_years])
        return len(years) - 1 - excl_length