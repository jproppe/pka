import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.linear_model  import LinearRegression
from   sklearn.linear_model  import BayesianRidge
from   sklearn.preprocessing import PolynomialFeatures

class pKa:
    
    def __init__(self, path_to_data='./training_data.xlsx'):
        
        '''
        Load data from path (default: './training_data.xlsx') and specify variables that are being
        used throughout the procedure. Display the underlying database.
        '''
        
        self.data = pd.read_excel('training_data.xlsx')
        
        display(self.data)
        
        self.x = np.array(self.data['pKa_theo']).flatten() # independent variable
        self.y = np.array(self.data['pKa_exp']).flatten()  # dependent variable
        
        if len(self.x) != len(self.y):
            raise Exception("Number of instances in pKa_exp and pKa_theo is required to be identical.")
            
        self.N  = len(self.x) # number of data points
        self.X  = PolynomialFeatures(1).fit_transform(self.x.reshape(-1, 1)) # add intercept term
        self.X2 = PolynomialFeatures(2).fit_transform(self.x.reshape(-1, 1)) # ... and quadratic term
        
        # define <x_grid> and <X_grid> (equivalent to <x> and <X>) for plotting purposes
        delta       = np.max(self.x) - np.min(self.x)
        self.x_grid = np.linspace(np.min(self.x) - .05 * delta, np.max(self.x) + .05 * delta, 250)
        self.X_grid = PolynomialFeatures(1).fit_transform(self.x_grid.reshape(-1, 1))

        self.rng = np.random.RandomState() # random number generator (local to the class)
        
    #------------------------------------------------------------------------------------------------------#
            
    def get_coefs(self, seed=None):
        
        '''
        Return coefficients (<model.coef_>) of a weighted linear regression model (<model>).
        Weights (<weight>) are obtained on the basis of Bayesian bootstrapping.
        If <x>-dependent variance (<var>) is available (from heteroscedastic regression), adjust weights.
        '''
        
        if not hasattr(self, 'var'):
            self.var = 1.
            
        if not seed is None: # sample-specific seed
            self.rng.seed(seed)
            
        weight = np.diff(np.concatenate(([0.], np.sort(self.rng.uniform(0., 1., self.N-1)), [1.])))
        model  = LinearRegression(fit_intercept=False, normalize=True).fit(self.X, self.y, weight / self.var)
            
        return model.coef_
    
    #------------------------------------------------------------------------------------------------------#
    
    def bootstrap(self):
        
        '''
        Draw 1000 bootstrap samples and perform weighted linear regression.
        Collect regression coefficients (<coefs>) and determine the ensemble mean (<coefs_mean>)
        and covariance (<coefs_cov>).
        Approximate <y> on the basis of the ensemble of regression models (predictions <f>).
        '''
        
        self.coefs = []
        
        for b in range(1000):
            self.coefs.append(self.get_coefs(seed=b))
            
        self.coefs      = np.array(self.coefs)
        self.coefs_mean = np.mean(self.coefs, axis=0).reshape(-1, 1)
        self.coefs_cov  = np.cov(self.coefs.T)
        
        self.f = self.X.dot(self.coefs_mean).flatten()
        
        # necessary if heteroscedastic regression has not yet been performed
        if not hasattr(self, 'subcoefs'): 
            self.subcoefs = np.array([self.N / (self.N - 2) * np.mean((self.y - self.f)**2), 0., 0.])
    
    #------------------------------------------------------------------------------------------------------#
    
    def predict(self, x_query):
        
        '''
        Make a prediction (<f>) including uncertainty (<u>, 95% confidence interval (CI)) 
        based on <x_query>.
        '''
        
        x_query  = np.array(x_query)
        X_query  = PolynomialFeatures(1).fit_transform(x_query.reshape(-1, 1))
        X_query2 = PolynomialFeatures(2).fit_transform(x_query.reshape(-1, 1))
        
        f = X_query.dot(self.coefs_mean).flatten()
        u = 1.96 * np.sqrt(X_query2.dot(self.subcoefs).flatten() + np.diag(X_query.dot(self.coefs_cov.dot(X_query.T))))
        
        return f, u
    
    #------------------------------------------------------------------------------------------------------#
    
    def plot_bootstrap_results(self, show_ensemble=True):
        
        '''
        Plot the results of the bootstrapping procedure. If <show_ensemble> is True, all regression lines
        will be plotted.
        '''
    
        if show_ensemble is True:
            for b in range(1000):
                if b == 0:
                    label_ = 'result for $b$th sample'
                else:
                    label_ = None
                plt.plot(self.x_grid, 
                         self.X_grid.dot(self.coefs[b,:].reshape(-1, 1)), 
                         color='#75bbfd', 
                         linewidth=.5, 
                         label=label_
                        )
                    
        f, u = self.predict(self.x_grid)
    
        plt.plot(self.x_grid, f, 'k-', label='regression line')
        plt.plot(self.x, self.y, 'k.', label='training data')
        plt.fill_between(self.x_grid, (f + u), (f - u), facecolor='red', alpha=0.2, label='uncertainty (95% CI)')
        plt.xlabel(r'p$K_a$ (theo)', fontsize=12)
        plt.ylabel(r'p$K_a$ (exp)', fontsize=12)
        plt.legend()
        
    #------------------------------------------------------------------------------------------------------#
        
    def query(self, x_query):
        
        '''
        Make a prediction (<f>) including uncertainty (<u>, 95% confidence interval (CI)) for a
        user-specific query (<x_query>). Print statistics.
        '''
        
        x_query = np.array(x_query).flatten()
        
        if len(x_query) != 1:
            raise Exception("Multiple queries were made, but only one at a time is possible at the moment.")
            
        self.plot_bootstrap_results(show_ensemble=False)
        
        f, u = self.predict(x_query)
        
        plt.errorbar(x_query, f, u, color='red', mfc='black', capsize=3, marker='o', label='queried prediction')
        
        print('Prediction           = ' + str(format(f.item(), '.3f')))
        print('Uncertainty (95% CI) = ' + str(format(u.item(), '.3f')))
        
        plt.legend()
        
    #------------------------------------------------------------------------------------------------------#
    
    def fit_variance(self):
        
        '''
        Heteroscedastic regression. Determine <var> as the <x>-dependent variance and <subcoefs> as
        the coefficients of this additional regression model.
        '''
     
        model    = BayesianRidge(fit_intercept=False, normalize=True).fit(self.X2, (self.y - self.f)**2)
        self.var = model.predict(self.X2).flatten()
        
        self.subcoefs = model.coef_
        
    #------------------------------------------------------------------------------------------------------#
        
    def check_query(self, query):
        
        '''
        Check whether a user-specify query is a valid number or not.
        '''
        
        try:
            float(query)
            return True
        except ValueError:
            return False
        
    #------------------------------------------------------------------------------------------------------#
        
    def run(self):
        
        '''
        The key method of the pKa class. Perform one run of homoscedastic regression (boostrapped)
        followed by three runs of heteroscedastic regression (bootstrapped). Print statistics.
        Allow users to make individual queries.
        '''
        
        if hasattr(self, 'var'):
            del self.var
        if hasattr(self, 'subcoefs'):
            del self.subcoefs
        
        self.bootstrap()
        
        for i in range(3):
            self.fit_variance()
            self.bootstrap()
    
        self.plot_bootstrap_results()
        plt.show()
    
        print('===============================================')
        print('SUMMARY OF HETEROSCEDASTIC BOOTSTRAP REGRESSION')
        print('intercept = ' + str(format(np.mean(self.coefs[:,0]), '.3f')) \
                             + ' +/- ' + str(format(1.96 * np.std(self.coefs[:,0]), '.3f')) + ' (95% confidence)')
        print('slope     = ' + str(format(np.mean(self.coefs[:,1]), '.3f')) \
                             + ' +/- ' + str(format(1.96 * np.std(self.coefs[:,1]), '.3f')) + ' (95% confidence)')
        print('===============================================\n')
        
        querying = True
        
        while querying:
            print('Enter any non-digit character to stop the procedure.')
            query    = input('Enter pKa value: ')
            querying = self.check_query(query)
            if querying:
                self.query(float(query))
                plt.show()
