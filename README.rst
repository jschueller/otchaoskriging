.. image:: https://github.com/jschueller/otchaoskriging/actions/workflows/build.yml/badge.svg?branch=master
    :target: https://github.com/jschueller/otchaoskriging/actions/workflows/build.yml

otchaoskriging
==============
Allows to build metamodels thanks to the chaos-kriging method (https://arxiv.org/pdf/1502.03939.pdf).

Example::

    import otchaoskriging 
    import openturns as ot
    from openturns.usecases import ishigami_function

    im = ishigami_function.IshigamiModel()
    distribution = im.distributionX
    size = 1000
    X = im.distributionX.getSample(size)
    Y = im.model(X)
    covmodel = ot.SquaredExponential([0.1] * im.dim, [1.0])
    algo = otchaoskriging.SPCKriging(X, Y, im.distributionX, covmodel)
    algo.run()
    result = algo.getResult()
    mm = result.getMetaModel()
