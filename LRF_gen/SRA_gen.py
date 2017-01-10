import numpy
import numpy.fft
import math
import scipy.stats
import warnings


def cube_make_SRA(res, sigma, H):

    warnings.warn("deprecated", DeprecationWarning)

    N = 2**res
    X = numpy.zeros([N+1, N+1, N+1])

    size = N
    halfsize = N/2

    delta = sigma

    X[0:N+1:N, 0:N+1:N, 0:N+1:N] = scipy.stats.norm.rvs(scale=delta,
                                                        size=[2, 2, 2])

    N1 = N
    N2 = N1/2

    delta1 = delta*pow(3./4., H)*math.sqrt(1-0.25*pow(4./3., H)) / pow(2, -H)
    delta2 = delta*pow(2., -2*H)*math.sqrt(1-0.25*pow(3./2., H)) / pow(2, -H)
    delta3 = delta*pow(2., -H) * math.sqrt(1-0.25*pow(2., H)) / pow(2, -H)

    for stage in range(1, res+1):

        delta1 *= pow(2., -H)
        delta2 *= pow(2., -H)
        delta3 *= pow(2., -H)

    # Type 1 analogue (Saupe) cube - Jilesen a
    # cube centre points

        X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += (
             (X[2*N2::N1, 2*N2::N1, 2*N2::N1]
              + X[2*N2::N1, 2*N2::N1, :-2*N2:N1]
              + X[2*N2::N1, :-2*N2:N1, 2*N2::N1]
              + X[2*N2::N1, :-2*N2:N1, :-2*N2:N1]
              + X[:-2*N2:N1, 2*N2::N1, 2*N2::N1]
              + X[:-2*N2:N1, 2*N2::N1, :-2*N2:N1]
              + X[:-2*N2:N1, :-2*N2:N1, 2*N2::N1]
              + X[:-2*N2:N1, :-2*N2:N1, :-2*N2:N1]
              )/8. + scipy.stats.norm.rvs(
                        scale=delta1,
                        size=X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape))

        # Random addition
        X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(
                                      scale=delta1,
                                      size=X[0::N1, 0::N1, 0::N1].shape)

    # Type 2a analogue - square bipyramid - Jilesen b
    # face mid points

        # outer-side points
        X[N2:-N2:N1, N2:-N2:N1, 0] = ((X[2*N2::N1, 2*N2::N1, 0]
                                       + X[2*N2::N1, :-2*N2:N1, 0]
                                       + X[:-2*N2:N1, 2*N2::N1, 0]
                                       + X[:-2*N2:N1, :-2*N2:N1, 0]
                                       + X[N2:-N2:N1, N2:-N2:N1, N2]
                                       )/5.
                                      + scipy.stats.norm.rvs(
                                             scale=delta2,
                                             size=X[N2:-N2:N1, N2:-N2:N1, 0].
                                             shape))

        X[N2:-N2:N1, N2:-N2:N1, -1] = ((X[2*N2::N1, 2*N2::N1, -1]
                                        + X[2*N2::N1, :-2*N2:N1, -1]
                                        + X[:-2*N2:N1, 2*N2::N1, -1]
                                        + X[:-2*N2:N1, :-2*N2:N1, -1]
                                        + X[N2:-N2:N1, N2:-N2:N1, -N2-1]
                                        )/5.
                                       + scipy.stats.norm.rvs(
                                             scale=delta2,
                                             size=X[N2:-N2:N1, N2:-N2:N1, 0].
                                             shape))

        X[N2:-N2:N1, 0, N2:-N2:N1] = ((X[2*N2::N1, 0, 2*N2::N1]
                                       + X[2*N2::N1, 0, :-2*N2:N1]
                                       + X[:-2*N2:N1, 0, 2*N2::N1]
                                       + X[:-2*N2:N1, 0, :-2*N2:N1]
                                       + X[N2:-N2:N1, N2, N2:-N2:N1]
                                       )/5.
                                      + scipy.stats.norm.rvs(
                                            scale=delta2,
                                            size=X[N2:-N2:N1, N2:-N2:N1, 0]
                                            .shape))

        X[N2:-N2:N1, -1, N2:-N2:N1] = ((X[2*N2::N1, -1, 2*N2::N1]
                                        + X[2*N2::N1, -1, :-2*N2:N1]
                                        + X[:-2*N2:N1, -1, 2*N2::N1]
                                        + X[:-2*N2:N1, -1, :-2*N2:N1]
                                        + X[N2:-N2:N1, -N2-1, N2:-N2:N1]
                                        )/5.
                                       + scipy.stats.norm.rvs(
                                             scale=delta2,
                                             size=X[N2:-N2:N1, N2:-N2:N1, 0]
                                             .shape))

        X[0, N2:-N2:N1, N2:-N2:N1] = ((X[0, 2*N2::N1, 2*N2::N1]
                                       + X[0, 2*N2::N1, :-2*N2:N1]
                                       + X[0, :-2*N2:N1, 2*N2::N1]
                                       + X[0, :-2*N2:N1, :-2*N2:N1]
                                       + X[N2, N2:-N2:N1, N2:-N2:N1]
                                       )/5.
                                      + scipy.stats.norm.rvs(
                                            scale=delta2,
                                            size=X[N2:-N2:N1, N2:-N2:N1, 0]
                                            .shape))

        X[-1, N2:-N2:N1, N2:-N2:N1] = ((X[-1, 2*N2::N1, 2*N2::N1]
                                        + X[-1, 2*N2::N1, :-2*N2:N1]
                                        + X[-1, :-2*N2:N1, 2*N2::N1]
                                        + X[-1, :-2*N2:N1, :-2*N2:N1]
                                        + X[N2:-N2:N1, N2:-N2:N1, -N2-1]
                                        )/5.
                                       + scipy.stats.norm.rvs(
                                             scale=delta2,
                                             size=X[N2:-N2:N1, N2:-N2:N1, 0]
                                             .shape))

        # other points
        if stage != 1:
            X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1] = (
                (X[2*N2::N1, 2*N2::N1, N1:-N2:N1]
                 + X[2*N2::N1, :-2*N2:N1, N1:-N2:N1]
                 + X[:-2*N2:N1, 2*N2::N1, N1:-N2:N1]
                 + X[:-2*N2:N1, :-2*N2:N1, N1:-N2:N1]
                 + X[N2:-N2:N1, N2:-N2:N1, N1+N2::N1]
                 + X[N2:-N2:N1, N2:-N2:N1, N1-N2:-2*N2:N1])/6.
                + scipy.stats.norm.rvs(scale=delta2,
                                       size=X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1]
                                       .shape))

            X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1] = (
                (X[2*N2::N1, N1:-N2:N1, 2*N2::N1]
                 + X[2*N2::N1, N1:-N2:N1, :-2*N2:N1]
                 + X[:-2*N2:N1, N1:-N2:N1, 2*N2::N1]
                 + X[:-2*N2:N1, N1:-N2:N1, :-2*N2:N1]
                 + X[N2:-N2:N1, N1+N2::N1, N2:-N2:N1]
                 + X[N2:-N2:N1, N1-N2:-2*N2:N1, N2:-N2:N1])/6.
                + scipy.stats.norm.rvs(scale=delta2,
                                       size=X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1]
                                       .shape))

            X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1] = (
                (X[N1:-N2:N1, 2*N2::N1, 2*N2::N1]
                 + X[N1:-N2:N1, 2*N2::N1, :-2*N2:N1]
                 + X[N1:-N2:N1, :-2*N2:N1, 2*N2::N1]
                 + X[N1:-N2:N1, :-2*N2:N1, :-2*N2:N1]
                 + X[N1+N2::N1, N2:-N2:N1, N2:-N2:N1]
                 + X[N1-N2:-2*N2:N1, N2:-N2:N1, N2:-N2:N1])/6.
                + scipy.stats.norm.rvs(scale=delta2,
                                       size=X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1]
                                       .shape))

        # Random addition
        X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(
                                      scale=delta2,
                                      size=X[0::N1, 0::N1, 0::N1].shape)
        X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                                 scale=delta2,
                                 size=X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)

    # Type 2b analogue - octohedron - Jilesen c
    # edge middle points
    #
    # Maybe an error here in Lu et al.'s technique that I have attempted
    # to correct

        # outer edges x12!
        # 1-4

        X[N2:-N2:N1, 0, 0] = ((X[2*N2::N1, 0, 0] + X[:-2*N2:N1, 0, 0]
                               + X[N2:-N2:N1, 0, N2] + X[N2:-N2:N1, N2, 0])/4.
                              + scipy.stats.norm.rvs(
                                    scale=delta3,
                                    size=X[N2:-N2:N1, 0, 0].shape))

        X[N2:-N2:N1, 0, -1] = ((X[2*N2::N1, 0, -1] + X[:-2*N2:N1, 0, -1]
                                + X[N2:-N2:N1, 0, -N2-1] + X[N2:-N2:N1, N2, -1]
                                )/4. + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N2:-N2:N1, 0, 0].shape))

        X[N2:-N2:N1, -1, 0] = ((X[2*N2::N1, -1, 0] + X[:-2*N2:N1, -1, 0]
                                + X[N2:-N2:N1, -1, N2] + X[N2:-N2:N1, -N2-1, 0]
                                )/4. + scipy.stats.norm.rvs(
                                           scale=delta3,
                                           size=X[N2:-N2:N1, 0, 0].shape))

        X[N2:-N2:N1, -1, -1] = ((X[2*N2::N1, -1, -1] + X[:-2*N2:N1, -1, -1]
                                 + X[N2:-N2:N1, -1, -N2-1]
                                 + X[N2:-N2:N1, -N2-1, -1])/4.
                                + scipy.stats.norm.rvs(
                                      scale=delta3,
                                      size=X[N2:-N2:N1, 0, 0].shape))

        # 5-8

        X[0, N2:-N2:N1, 0] = ((X[0, 2*N2::N1, 0] + X[0, :-2*N2:N1, 0]
                               + X[0, N2:-N2:N1, N2] + X[N2, N2:-N2:N1, 0])/4.
                              + scipy.stats.norm.rvs(
                                    scale=delta3,
                                    size=X[0, N2:-N2:N1, 0].shape))

        X[0, N2:-N2:N1, -1] = ((X[0, 2*N2::N1, -1] + X[0, :-2*N2:N1, -1]
                                + X[0, N2:-N2:N1, -N2-1] + X[N2, N2:-N2:N1, -1]
                                )/4. + scipy.stats.norm.rvs(
                                           scale=delta3,
                                           size=X[0, N2:-N2:N1, 0].shape))

        X[-1, N2:-N2:N1, 0] = ((X[-1, 2*N2::N1, 0] + X[-1, :-2*N2:N1, 0]
                                + X[-1, N2:-N2:N1, N2] + X[-N2-1, N2:-N2:N1, 0]
                                )/4. + scipy.stats.norm.rvs(
                                           scale=delta3,
                                           size=X[0, N2:-N2:N1, 0].shape))

        X[-1, N2:-N2:N1, -1] = ((X[-1, 2*N2::N1, -1] + X[-1, :-2*N2:N1, -1]
                                 + X[-1, N2:-N2:N1, -N2-1]
                                 + X[-N2-1, N2:-N2:N1, -1])/4.
                                + scipy.stats.norm.rvs(
                                      scale=delta3,
                                      size=X[0, N2:-N2:N1, 0].shape))

        # 9-12

        X[0, 0, N2:-N2:N1] = ((X[0, 0, 2*N2::N1] + X[0, 0, :-2*N2:N1]
                               + X[0, N2, N2:-N2:N1] + X[N2, 0, N2:-N2:N1])/4.
                              + scipy.stats.norm.rvs(
                                    scale=delta3,
                                    size=X[0, 0, N2:-N2:N1].shape))

        X[0, -1, N2:-N2:N1] = ((X[0, -1, 2*N2::N1] + X[0, -1, :-2*N2:N1]
                                + X[0, -N2-1, N2:-N2:N1] + X[N2, -1, N2:-N2:N1]
                                )/4. + scipy.stats.norm.rvs(
                                           scale=delta3,
                                           size=X[0, 0, N2:-N2:N1].shape))

        X[-1, 0, N2:-N2:N1] = ((X[-1, 0, 2*N2::N1] + X[-1, 0, :-2*N2:N1]
                                + X[-1, N2, N2:-N2:N1] + X[-N2-1, 0, N2:-N2:N1]
                                )/4. + scipy.stats.norm.rvs(
                                           scale=delta3,
                                           size=X[0, 0, N2:-N2:N1].shape))

        X[-1, -1, N2:-N2:N1] = ((X[-1, -1, 2*N2::N1] + X[-1, -1, :-2*N2:N1]
                                 + X[-1, -N2-1, N2:-N2:N1]
                                 + X[-N2-1, -1, N2:-N2:N1])/4.
                                + scipy.stats.norm.rvs(
                                      scale=delta3,
                                      size=X[0, 0, N2:-N2:N1].shape))

        # other points

        if stage > 1:
            # 1-4

            X[N2:-N2:N1, N1:-N2:N1, 0] = ((X[N2:-N2:N1, N1:-N2:N1, N2]
                                           + X[N2:-N2:N1, N1+N2::N1, 0]
                                           + X[N2:-N2:N1, N1-N2:-2*N2:N1, 0]
                                           + X[2*N2::N1, N1:-N2:N1, 0]
                                           + X[:-2*N2:N1, N1:-N2:N1, 0]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N2:-N2:N1, N1:-N2:N1, 0]
                                                .shape))

            X[N2:-N2:N1, N1:-N2:N1, -1] = ((X[N2:-N2:N1, N1:-N2:N1, -N2-1]
                                            + X[N2:-N2:N1, N1+N2::N1, -1]
                                            + X[N2:-N2:N1, N1-N2:-2*N2:N1, -1]
                                            + X[2*N2::N1, N1:-N2:N1, -1]
                                            + X[:-2*N2:N1, N1:-N2:N1, -1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N2:-N2:N1, N1:-N2:N1, 0]
                                                .shape))

            X[N1:-N2:N1, N2:-N2:N1, 0] = ((X[N1:-N2:N1, N2:-N2:N1, N2]
                                           + X[N1:-N2:N1, 2*N2::N1, 0]
                                           + X[N1:-N2:N1, :-2*N2:N1, 0]
                                           + X[N1+N2::N1, N2:-N2:N1, 0]
                                           + X[N1-N2:-2*N2:N1, N2:-N2:N1, 0]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N1:-N2:N1, N2:-N2:N1, 0]
                                                .shape))

            X[N1:-N2:N1, N2:-N2:N1, -1] = ((X[N1:-N2:N1, N2:-N2:N1, -N2-1]
                                            + X[N1:-N2:N1, 2*N2::N1, -1]
                                            + X[N1:-N2:N1, :-2*N2:N1, -1]
                                            + X[N1+N2::N1, N2:-N2:N1, -1]
                                            + X[N1-N2:-2*N2:N1, N2:-N2:N1, -1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N1:-N2:N1, N2:-N2:N1, 0]
                                                .shape))

            # 5-8

            X[N2:-N2:N1, 0, N1:-N2:N1] = ((X[N2:-N2:N1, N2, N1:-N2:N1]
                                           + X[N2:-N2:N1, 0, N1+N2::N1]
                                           + X[N2:-N2:N1, 0, N1-N2:-2*N2:N1]
                                           + X[2*N2::N1, 0, N1:-N2:N1]
                                           + X[:-2*N2:N1, 0, N1:-N2:N1]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N2:-N2:N1, 0, N1:-N2:N1]
                                                .shape))

            X[N2:-N2:N1, -1, N1:-N2:N1] = ((X[N2:-N2:N1, -N2-1, N1:-N2:N1]
                                            + X[N2:-N2:N1, -1, N1+N2::N1]
                                            + X[N2:-N2:N1, -1, N1-N2:-2*N2:N1]
                                            + X[2*N2::N1, -1, N1:-N2:N1]
                                            + X[:-2*N2:N1, -1, N1:-N2:N1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                               scale=delta3,
                                               size=X[N2:-N2:N1, -1, N1:-N2:N1]
                                               .shape))

            X[N1:-N2:N1, 0, N2:-N2:N1] = ((X[N1:-N2:N1, N2, N2:-N2:N1]
                                           + X[N1:-N2:N1, 0, 2*N2::N1]
                                           + X[N1:-N2:N1, 0, :-2*N2:N1]
                                           + X[N1+N2::N1, 0, N2:-N2:N1]
                                           + X[N1-N2:-2*N2:N1, 0, N2:-N2:N1]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N1:-N2:N1, 0, N2:-N2:N1]
                                                .shape))

            X[N1:-N2:N1, -1, N2:-N2:N1] = ((X[N1:-N2:N1, -N2-1, N2:-N2:N1]
                                            + X[N1:-N2:N1, -1, 2*N2::N1]
                                            + X[N1:-N2:N1, -1, :-2*N2:N1]
                                            + X[N1+N2::N1, -1, N2:-N2:N1]
                                            + X[N1-N2:-2*N2:N1, -1, N2:-N2:N1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[N1:-N2:N1, 0, N2:-N2:N1]
                                                .shape))

            # 9-12

            X[0, N2:-N2:N1, N1:-N2:N1] = ((X[N2, N2:-N2:N1, N1:-N2:N1]
                                           + X[0, N2:-N2:N1, N1+N2::N1]
                                           + X[0, N2:-N2:N1, N1-N2:-2*N2:N1]
                                           + X[0, 2*N2::N1, N1:-N2:N1]
                                           + X[0, :-2*N2:N1, N1:-N2:N1]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[0, N2:-N2:N1, N1:-N2:N1]
                                                .shape))

            X[-1, N2:-N2:N1, N1:-N2:N1] = ((X[-N2-1, N2:-N2:N1, N1:-N2:N1]
                                            + X[-1, N2:-N2:N1, N1+N2::N1]
                                            + X[-1, N2:-N2:N1, N1-N2:-2*N2:N1]
                                            + X[-1, 2*N2::N1, N1:-N2:N1]
                                            + X[-1, :-2*N2:N1, N1:-N2:N1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[0, N2:-N2:N1, N1:-N2:N1]
                                                .shape))

            X[0, N1:-N2:N1, N2:-N2:N1] = ((X[N2, N1:-N2:N1, N2:-N2:N1]
                                           + X[0, N1:-N2:N1, 2*N2::N1]
                                           + X[0, N1:-N2:N1, :-2*N2:N1]
                                           + X[0, N1+N2::N1, N2:-N2:N1]
                                           + X[0, N1-N2:-2*N2:N1, N2:-N2:N1]
                                           )/5.
                                          + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[0, N1:-N2:N1, N2:-N2:N1]
                                                .shape))

            X[-1, N1:-N2:N1, N2:-N2:N1] = ((X[-N2-1, N1:-N2:N1, N2:-N2:N1]
                                            + X[-1, N1:-N2:N1, 2*N2::N1]
                                            + X[-1, N1:-N2:N1, :-2*N2:N1]
                                            + X[-1, N1+N2::N1, N2:-N2:N1]
                                            + X[-1, N1-N2:-2*N2:N1, N2:-N2:N1]
                                            )/5.
                                           + scipy.stats.norm.rvs(
                                                scale=delta3,
                                                size=X[0, N1:-N2:N1, N2:-N2:N1]
                                                .shape))

            # 13-15

            X[N2:-N2:N1, N1:-N1:N1, N1:-N1:N1] = (
                (X[N2:-N2:N1, N1:-N1:N1, N1+N2:-N1+N2:N1]
                 + X[N2:-N2:N1, N1:-N1:N1, N1-N2:-N1-N2:N1]
                 + X[N2:-N2:N1, N1+N2:-N1+N2:N1, N1:-N1:N1]
                 + X[N2:-N2:N1, N1-N2:-N1-N2:N1, N1:-N1:N1]
                 + X[2*N2::N1, N1:-N1:N1, N1:-N1:N1]
                 + X[:-2*N2:N1, N1:-N1:N1, N1:-N1:N1]
                 )/6. + scipy.stats.norm.rvs(
                            scale=delta3,
                            size=X[N2:-N2:N1, N1:-N1:N1, N1:-N1:N1].shape))

            X[N1:-N1:N1, N2:-N2:N1, N1:-N1:N1] = (
                (X[N1:-N1:N1, N2:-N2:N1, N1+N2:-N1+N2:N1]
                 + X[N1:-N1:N1, N2:-N2:N1, N1-N2:-N1-N2:N1]
                 + X[N1+N2:-N1+N2:N1, N2:-N2:N1, N1:-N1:N1]
                 + X[N1-N2:-N1-N2:N1, N2:-N2:N1, N1:-N1:N1]
                 + X[N1:-N1:N1, 2*N2::N1, N1:-N1:N1]
                 + X[N1:-N1:N1, :-2*N2:N1, N1:-N1:N1]
                 )/6. + scipy.stats.norm.rvs(
                            scale=delta3,
                            size=X[N1:-N1:N1, N2:-N2:N1, N1:-N1:N1].shape))

            X[N1:-N1:N1, N1:-N1:N1, N2:-N2:N1] = (
                (X[N1:-N1:N1, N1+N2:-N1+N2:N1, N2:-N2:N1]
                 + X[N1:-N1:N1, N1-N2:-N1-N2:N1, N2:-N2:N1]
                 + X[N1+N2:-N1+N2:N1, N1:-N1:N1, N2:-N2:N1]
                 + X[N1-N2:-N1-N2:N1, N1:-N1:N1, N2:-N2:N1]
                 + X[N1:-N1:N1, N1:-N1:N1, 2*N2::N1]
                 + X[N1:-N1:N1, N1:-N1:N1, :-2*N2:N1]
                 )/6. + scipy.stats.norm.rvs(
                            scale=delta3,
                            size=X[N1:-N1:N1, N1:-N1:N1, N2:-N2:N1].shape))

        # random addition

        X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(
                                      scale=delta3,
                                      size=X[0::N1, 0::N1, 0::N1].shape)
        X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                                scale=delta3,
                                size=X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)

        X[N2:-N2:N1, N2:-N2:N1, 0] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[N2:-N2:N1, N2:-N2:N1, 0].shape)
        X[N2:-N2:N1, N2:-N2:N1, -1] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[N2:-N2:N1, N2:-N2:N1, -1].shape)
        X[N2:-N2:N1, 0, N2:-N2:N1] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[N2:-N2:N1, 0, N2:-N2:N1].shape)
        X[N2:-N2:N1, -1, N2:-N2:N1] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[N2:-N2:N1, -1, N2:-N2:N1].shape)
        X[0, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[0, N2:-N2:N1, N2:-N2:N1].shape)
        X[-1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                        scale=delta3, size=X[-1, N2:-N2:N1, N2:-N2:N1].shape)

        if stage != 1:
            X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1] += scipy.stats.norm.rvs(
                scale=delta3, size=X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1].shape)
            X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                scale=delta3, size=X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1].shape)
            X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(
                scale=delta3, size=X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)

        N1 /= 2
        N2 /= 2

    return numpy.exp(X)
