import otchaoskriging
import openturns as ot
from openturns.usecases import ishigami_function


def test_ishigami1():
    im = ishigami_function.IshigamiModel()
    distribution = im.distributionX
    size = 1000
    X = im.distributionX.getSample(size)
    Y = im.model(X)
    covmodel = ot.SquaredExponential([0.1] * im.dim, [1.0])
    algo = otchaoskriging.SPCKriging(X, Y, im.distributionX, covmodel)
    algo.run()
    result = algo.getResult()
    print("res=", result.getResiduals())
    print("err=", result.getRelativeErrors())

    assert abs(result.getRelativeErrors()[0]) < 1e-5
    mm = result.getMetaModel()


if __name__ == "__main__":
    test_ishigami()
