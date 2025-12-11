import otchaoskriging
import openturns as ot
from openturns.usecases import ishigami_function


def test_ishigami1():
    im = ishigami_function.IshigamiModel()
    size = 1000
    X = im.distribution.getSample(size)
    Y = im.model(X)
    covmodel = ot.SquaredExponential([0.1] * im.dim, [1.0])
    algo = otchaoskriging.SPCGPR(X, Y, im.distribution, covmodel)
    algo.run()
    result = algo.getResult()

    # validation
    surrogate = result.getMetaModel()
    validation = ot.MetaModelValidation(Y, surrogate(X))
    mse = validation.computeMeanSquaredError()
    print(f"MSE={mse}")
    r2 = validation.computeR2Score()[0]
    print(f"R2={r2}")
    assert r2 >= 1-1e-5


if __name__ == "__main__":
    test_ishigami()
