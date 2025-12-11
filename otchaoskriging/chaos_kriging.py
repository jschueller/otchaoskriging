import openturns as ot
try:
    from openturns import GaussianProcessFitter, GaussianProcessRegression
except ImportError:
    from openturns.experimental import GaussianProcessFitter, GaussianProcessRegression

class PCGPR:
    """
    Base class for chaos/gpr algorithms.
    """
    def __init__(self, X, Y, distribution, covarianceModel, basis=None, basisSize=None):
        if len(X) != len(Y):
            raise ValueError('input size must match output size')
        if distribution.getDimension() != len(X[0]):
            raise ValueError('distribution dimension must match input data dimension')
        if covarianceModel.getInputDimension() != len(X[0]):
            raise ValueError('covariance model input dimension must match input data dimension')
        if covarianceModel.getOutputDimension() != 1:
            raise ValueError('covariance model output dimension must be 1')
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


class SPCGPR(PCGPR):
    """
    SPC-GPR algorithm.

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
        projection = ot.LeastSquaresStrategy(approx)
        gpr_results = [None] * self.Y_.getDimension()
        for j in range(self.Y_.getDimension()):
            Yj = self.Y_.getMarginal(j)
            chaos = ot.FunctionalChaosAlgorithm(self.X_, Yj, self.distribution_, adaptive, projection)
            chaos.run()
            pc_res = chaos.getResult()
            psi_k = ot.Basis(pc_res.getReducedBasis())
            fitter = GaussianProcessFitter(self.X_, Yj, self.covarianceModel_, psi_k)
            fitter.run()
            gpr = GaussianProcessRegression(fitter.getResult())
            gpr.run()
            gpr_results[j] = gpr.getResult()
        metamodel = ot.AggregatedFunction([result.getMetaModel() for result in gpr_results])
        self.result_ = ot.MetaModelResult(self.X_, self.Y_, metamodel)


class OPCGPR(PCGPR):

    def __init__(X, Y, distribution, basis=None, basisSize=None):
        pass

    def run():
        pass
