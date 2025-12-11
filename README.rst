.. image:: https://github.com/jschueller/otchaoskriging/actions/workflows/build.yml/badge.svg?branch=master
    :target: https://github.com/jschueller/otchaoskriging/actions/workflows/build.yml

otchaosgpr
==========
Experimental module to build metamodels thanks to the chaos-kriging method (https://arxiv.org/pdf/1502.03939.pdf).

Example::

    import otchaosgpr
    import openturns as ot
    from openturns.usecases import ishigami_function

    im = ishigami_function.IshigamiModel()
    size = 1000
    X = im.distribution.getSample(size)
    Y = im.model(X)
    covmodel = ot.SquaredExponential([0.1] * im.dim, [1.0])
    algo = otchaosgpr.SPCGPR(X, Y, im.distribution, covmodel)
    algo.run()
    result = algo.getResult()
    mm = result.getMetaModel()
