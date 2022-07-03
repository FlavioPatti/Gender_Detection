"""
   This module contains the implementation of the GMM algorithm for classification 
"""
import numpy as np
import scipy.special as sps

def covariance_matrix2(D):
    """ Computes and returns the covariance matrix given the dataset D
        this is a more efficient implementation
    """
    # compute the dataset mean mu
    mu = D.mean(1)

    # mu is a 1-D array, we need to reshape it to a column vector
    mu = vcol(mu)

    # remove the mean from all the points
    DC = D - mu

    # DC is the matrix of centered data
    C = np.dot(DC, DC.T)
    C = C / float(D.shape[1])

    return C


def logpdf_GAU_ND(x, mu, C):
    """ Computes the Multivariate Gaussian log density for the dataset x
        C represents the covariance matrix sigma
    """
    # M is the number of rows of x, n of attributes for each sample
    M = x.shape[0]
    first = -(M/2) * np.log(2*np.pi)
    second = -0.5 * np.linalg.slogdet(C)[1]
    third = -0.5 * np.dot(
        np.dot((x-mu).T, np.linalg.inv(C)), (x - mu))

    return np.diag(first+second+third)


def vcol(x):
    """ Reshape the vector x into a column vector """
    return x.reshape(x.shape[0], 1)


def vrow(x):
    """ Reshape the vector x into a row vector """
    return x.reshape(1, x.shape[0])


def logpdf_GMM(X, gmm):
    """ Computes the log-density of a gmm for a set of samples contained in 
        matrix X
        X is a matrix of samples of shape (D, N),
        where D is the size of a sample 
        and D is the number of samples in X
        gmm is a list of component parameters representing the GMM:
        gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
        the result will be an array of shape (N,), whose components will
        contain the log-density for sample xi
    """
    M = len(gmm)
    N = X.shape[1]

    # Each row of S contains the (sub-)class conditional densities given
    # component Gi = g for all samples xi
    S = np.empty([M, N])

    for g in range(len(gmm)):
        for j, sample in enumerate(X.T):
            sample = vcol(sample)
            # gmm[g][1] = mu_g, gmm[g][2] = C_g
            S[g, j] = logpdf_GAU_ND(sample, vcol(gmm[g][1]), gmm[g][2])

    # Add to each row of S the logarithm of the prior of the corresponding
    # component log w_g
    for g in range(len(gmm)):
        # gmm[g][0] = w_g
        S[g, :] += np.log(gmm[g][0])

    # Compute the log-marginal log f_Xi(xi) for all samples xi
    logdens = sps.logsumexp(S, axis=0)

    return logdens


def EM_algorithm(X, initial_gmm, psi=0.01, printDetails=False, version="full"):
    """ Implementation of the GMM EM estimation procedure:
        the EM algorithm is useful to estimate the parameters of a GMM
        that maximize the likelihood for traning set X
        The initial estimate of the GMM is passed as parameter
        psi is used to constrain the eigenvalues of covariance matrices 
        in order to avoid degenerate solutions
        printDetails is a boolean, to print iterations of the algorithm or no
        type can be "full", "diagonal", "tied" to specify which version to use
        the default is Full covariance
    """

    gmm = initial_gmm
    M = len(gmm)
    F = X.shape[0]  # number of features of each sample
    N = X.shape[1]
    stop = False

    # calculate the average loglikelihood using the initial gmm
    previous_avg_ll = sum(logpdf_GMM(X, initial_gmm)) / N

    if(printDetails):
        print("-"*40)
        print("\nEM algorithm starting\n")
        print("INITIAL       avg ll: ", previous_avg_ll)

    # continue the algorithm untill the stopping criterion is met
    counter = 1
    while(stop == False):

        # E-step

        # Each row of S contains the (sub-)class conditional densities given
        # component Gi = g for all samples xi
        S = np.empty([M, N])

        for g in range(M):  # for g in range(len(gmm)):
            for j, sample in enumerate(X.T):
                sample = vcol(sample)
                # gmm[g][1] = mu_g, gmm[g][2] = C_g
                S[g, j] = logpdf_GAU_ND(sample, vcol(gmm[g][1]), gmm[g][2])

        # Add to each row of S the logarithm of the prior of the corresponding
        # component log w_g
        for g in range(M):  # for g in range(len(gmm)):
            S[g, :] += np.log(gmm[g][0])

        # S is now the matrix of joint densities f_Xi,Gi(xi,g)

        # Compute the log-marginal log f_Xi(xi) for all samples xi
        logdens = sps.logsumexp(S, axis=0)

        # Remove from each row of the joint densities matrix S the row vector
        # containing the N marginal densities logdens
        log_responsabilities = S - logdens

        # Compute the MxN matrix of class posterior probabilities, responsabilities
        responsabilities = np.exp(log_responsabilities)

        # M-step

        # Compute statistics
        Zg_list = []
        Fg_list = []
        Sg_list = []
        for g in range(M):
            Zg_list.append(vrow(responsabilities[g]).sum(axis=1))
            Fg_list.append((vrow(responsabilities[g]) * X).sum(axis=1))
            tmp = np.zeros((F, F))
            for i in range(N):
                tmp += responsabilities[g][i] * \
                    np.dot(vcol(X.T[i]), vrow(X.T[i]))
            Sg_list.append(tmp)

        sum_covariances = np.zeros((F, F))  # used for tied covariance version
        # Obtain the new paramters
        for g in range(M):
            w_new = (Zg_list[g] / sum(Zg_list))[0]  # extract the float
            mu_new = vcol(Fg_list[g] / Zg_list[g])
            sigma_new = (Sg_list[g] / Zg_list[g]) - \
                np.dot(vcol(mu_new), vrow(mu_new))

            # diagonal version
            if(version == "diagonal"):
                sigma_new = sigma_new * np.eye(sigma_new.shape[0])

            # tied version
            if(version == "tied"):
                sum_covariances += Zg_list[g] * sigma_new

            # Constraining the eigenvalues of the covariance matrices to be
            # larger or equal to psi
            U, s, _ = np.linalg.svd(sigma_new)
            s[s < psi] = psi
            sigma_new = np.dot(U, vcol(s) * U.T)

            gmm[g] = (w_new, mu_new, sigma_new)

        for g in gmm:
            if(version == "tied"):
                g = (g[0], g[1], sum_covariances / N)

        # Check stopping criterion
        threshold = 1e-6
        this_avg_ll = sum(logpdf_GMM(X, gmm)) / N
        if printDetails:
            print("ITERATION ", counter, " avg ll: ", this_avg_ll)
        if (this_avg_ll - previous_avg_ll < threshold):
            stop = True
            if printDetails:
                print("\nSTOPPING CRITERION MET")
                print("\nEM algorithm finished\n")
                print("-"*40)
        else:
            previous_avg_ll = this_avg_ll
            counter += 1

    return gmm


def LBG_algorithm(X, gmm=None, goal_components=None, alpha=0.1, psi=0.01, printDetails=False, version="full"):
    """ Implementation of the LBG algorithm:
        starting from a gmm passed as parameter (or GMM_1  if nothing is passed)
        incrementally constructs a GMM with 2G components from a GMM with G 
        components, we stop when we reach goal components
        type can be "full", "diagonal", "tied" to specify which version to use
        the default is Full covariance
    """

    if (gmm == None):
        # GMM_1 = [(w, mu, C)] = [(1.0, mu, C)] gaussian density
        gmm = [(1.0, vcol(X.mean(1)), covariance_matrix2(X))]

    components = len(gmm)

    # Constraining the eigenvalues of the covariance matrices
    # g[2] is the covariance matrix
    for g in gmm:
        U, s, _ = np.linalg.svd(g[2])
        s[s < psi] = psi
        g = (g[0], g[1], np.dot(U, vcol(s)*U.T))

    if (goal_components == None):
        goal_components = components * 2
    if (printDetails):
        print("-"*40)
        print("\nLBG algorithm starting\n")
    counter = 1
    while components < goal_components:
        if (printDetails):
            print("\nITERATION ", counter)
            print("to obtain n_components = ", components*2, "\n")
        new_gmm = []
        for g in gmm:

            # Constraining the eigenvalues of the covariance matrices
            # g[2] is the covariance matrix
            U, s, _ = np.linalg.svd(g[2])
            s[s < psi] = psi
            g = (g[0], g[1], np.dot(U, vcol(s)*U.T))

            U, s, Vh = np.linalg.svd(g[2])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_gmm.append((g[0] / 2, g[1] + d, g[2]))
            new_gmm.append((g[0] / 2, g[1] - d, g[2]))

        # The 2G components gmm can be used as initial gmm for the EM algorithm
        gmm = EM_algorithm(X, new_gmm, psi, printDetails, version)
        counter += 1
        components *= 2

    if(printDetails):
        print("\nLBG algorithm finished\n")
        print("-"*40)

    return gmm


def GMM_classifier(DTR, LTR, DTE, LTE,  M, psi, version="full"):
    """ Implementation of the GMM classifier for binary classification """

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    gmm0 = LBG_algorithm(DTR0, goal_components=M, psi=psi,
                         version=version, printDetails=False)
    gmm1 = LBG_algorithm(DTR1, goal_components=M, psi=psi,
                         version=version, printDetails=False)

    S0 = logpdf_GMM(DTE, gmm0)
    S1 = logpdf_GMM(DTE, gmm1)

    return S1 - S0
