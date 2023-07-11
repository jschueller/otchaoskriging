import openturns as ot


class PCKriging:
    """
    Base class for chaos/kriging algorithms.
    """
    def __init__(self, X, Y, distribution, covarianceModel, basis=None, basisSize=None):
        if len(X) != len(Y):
            raise ValueError('input size must match output size')
        if distribution.getDimension() != len(X[0]):
            raise ValueError('distribution dimension must match input data dimension')
        if covarianceModel.getInputDimension() != len(X[0]):
            raise ValueError('covariance model input dimension must match input data dimension')
        if covarianceModel.getOutputDimension() != len(Y[0]):
            raise ValueError('covariance model output dimension must match output data dimension')
        self.X_ = X
        self.Y_ = Y
        self.distribution_ = distribution
        self.covarianceModel_ = covarianceModel
        dimension = distribution.getDimension()
        if basis is None:
            polynomials = [None] * dimension
            for i in range(dimension):
                marg_i = distribution.getMarginal(i)
                polynomials[i] = ot.StandardDistributionPolynomialFactory(marg_i)
            enumerateF = ot.LinearEnumerateFunction(dimension)
            basis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerateF)
        self.basis_ = basis
        if basisSize is None:
            maximumTotalDegree = ot.ResourceMap.GetAsUnsignedInteger("FunctionalChaosAlgorithm-MaximumTotalDegree")
            basisSize = basis.getEnumerateFunction().getBasisSizeFromTotalDegree(maximumTotalDegree)
        self.basisSize_ = basisSize
        self.result_ = None

    def getResult(self):
        return self.result_


class SPCKriging(PCKriging):
    """
    SPC-Kriging algorithm.

    Parameters
    ----------
    X, Y : 2d-sequence of sample
        Input/Output data
    distribution : openturns.Distribution
        Input distribution
    covarianceModel : openturns.CovarianceModel
        Covariance model
    """

    def __init__(self, X, Y, distribution, covarianceModel, basis=None, basisSize=None):
        super().__init__(X, Y, distribution, covarianceModel, basis, basisSize)

    def run(self):
        adaptive = ot.FixedStrategy(self.basis_, self.basisSize_)
        approx = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        proj = ot.LeastSquaresStrategy(approx)
        chaos = ot.FunctionalChaosAlgorithm(self.X_, self.Y_, self.distribution_, adaptive, proj)
        chaos.run()
        pc_res = chaos.getResult()
        psi_k = pc_res.getReducedBasis()
        kriging = ot.KrigingAlgorithm(self.X_, self.Y_, self.covarianceModel_, ot.Basis(psi_k))
        kriging.run()
        self.result_ = kriging.getResult()


class OPCKriging(PCKriging):

    def __init__(X, Y, distribution, basis=None, basisSize=None):
        pass

    def run():
        pass
