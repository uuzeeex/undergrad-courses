import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
import matplotlib.pyplot as plt

def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 1,
                    'weight_regularization': 0.1,
                    'num_iterations': 50
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    #weights = np.random.normal(0, 1, (M + 1, 1))
    run_time = 10
    lmds = [0.001, 0.01, 0.1, 1.0]
    CE_train_lmd = []
    CE_valid_lmd = []
    err_train_lmd = []
    err_valid_lmd = []
    for lmd in lmds:
        
        hyperparameters['weight_regularization'] = lmd
        #run_check_grad(hyperparameters)
        
        cross_entropies_train = []
        cross_entropies_valid = []
        accs_correct_train = []
        accs_correct_valid = []
        for run in range(run_time):
            weights = np.random.rand(M + 1, 1)

            # Verify that your logistic function produces the right gradient.
            # diff should be very close to 0.
        
            # Begin learning with gradient descent
            for t in range(hyperparameters['num_iterations']):

                # TODO: you may need to modify this loop to create plots, etc.

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
                # Evaluate the prediction.
                cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

                if np.isnan(f) or np.isinf(f):
                    raise ValueError("nan/inf error")

                # update parameters
                weights = weights - hyperparameters['learning_rate'] * df / N

                # Make a prediction on the valid_inputs.
                predictions_valid = logistic_predict(weights, valid_inputs)

                # Evaluate the prediction.
                cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
                # print some stats
                #stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
                #stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
                #print(stat_msg.format(t+1,
                #                      float(f / N),
                #                      float(cross_entropy_train),
                #                      float(frac_correct_train*100),
                #                      float(cross_entropy_valid),
                #                      float(frac_correct_valid*100)))
                if t == hyperparameters['num_iterations'] - 1:
                    cross_entropies_train.append(cross_entropy_train)
                    cross_entropies_valid.append(cross_entropy_valid)
                    accs_correct_train.append(frac_correct_train)
                    accs_correct_valid.append(frac_correct_valid)
        CE_train_lmd.append(np.mean(cross_entropies_train))
        CE_valid_lmd.append(np.mean(cross_entropies_valid))
        err_train_lmd.append(1. - np.mean(accs_correct_train))
        err_valid_lmd.append(1. - np.mean(accs_correct_valid))
    
    plt.plot(np.log(lmds), CE_train_lmd, label="train CE")
    plt.plot(np.log(lmds), CE_valid_lmd, label="validation CE")
    plt.legend(loc='center left')
    plt.xlabel('log lambda')
    plt.ylabel('CE')
    plt.savefig('plot_small_ce_pen.eps', format='eps', dpi=1000)
    plt.show()
    plt.close()
    
    plt.plot(np.log(lmds), err_train_lmd, label="train error")
    plt.plot(np.log(lmds), err_valid_lmd, label="validation error")
    plt.axis([-7, 0, -0.01, 0.4])
    plt.legend(loc='center left')
    plt.xlabel('log lambda')
    plt.ylabel('error')
    plt.savefig('plot_small_err_pen.eps', format='eps', dpi=1000)
    plt.show()
    
    #predictions_test = logistic_predict(weights, test_inputs)
    #cross_entropy_test, frac_correct_test= evaluate(test_targets, predictions_test)
    #print("Test CE:", cross_entropy_test)
    #print("Test Accuracy:", frac_correct_test)

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic_pen,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)

if __name__ == '__main__':
    run_logistic_regression()
