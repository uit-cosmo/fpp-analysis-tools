import numpy as np
import scipy.stats as sps
from scipy.integrate import simpson


def cdf(Data, ccdf=True):
    """ This function calculates only the cdf (or ccdf) of the data using the method described belwo in 'distirbution'. It does not interpolate. """
    DS = np.sort(Data)
    ECDF = np.linspace(0.0, 1.0, len(DS))
    if ccdf == True:
        ECDF = 1 - ECDF
    return DS, ECDF


def get_hist(Data, N):
    """
    This function calculates the histogram of Data.
    N is the number of bins to separate the data into.
    returns:
        x: array of bin centers
        hist: histogram
    """
    hist, edges = np.histogram(Data, N, density=True)
    # We are interested in the middle points inside the bins, not the edges of the bins:
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return bin_centers, hist


def distribution(Data, N, kernel=False, ccdf=True):
    """ This function calculates the pdf and ccdf of Data, either by histogram or by gaussian kernels.
    N:
    If histogram is used, N is the number of bins to separate the data into.
    If kernel is used, N gives the number of data points.
    ccdf: if true, returns the complementary cdf
    """
    if kernel == False:
        # Calculate PDF
        pdf, edges = np.histogram(Data, N, density=True)
        # We are interested in the middle points inside the bins, not the edges of the bins:
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Finding the CDF:
        # This sorts the data (with M datapoints) and, for each data point the cdf increases by 1/M from 0 to (M-1)/M
        # This is an unbiased estimator for the CDF
        # https://en.wikipedia.org/wiki/Empirical_distribution_function
        DS = np.sort(Data)
        ECDF = np.arange(len(DS)) / float(len(DS))

        # We wish to use the bin_centers as data points, and interpolate:
        cdf = np.interp(bin_centers, DS, ECDF)

        if ccdf == True:
            cdf = (
                1.0 - cdf
            )  # We want the complementary cummulative distribution function

        return pdf, cdf, bin_centers
    elif kernel == True:
        X = np.linspace(min(Data), max(Data), N)

        pdf_func = sps.gaussian_kde(Data)
        pdf = pdf_func(X)

        cdf_func = lambda ary: np.array(
            [pdf_func.integrate_box_1d(-np.inf, x) for x in ary]
        )
        cdf = 1 - cdf_func(X)

        return pdf, cdf, X


def joint_pdf(X, Y, N=64, pdfs=False):
    """ This function creates the joint PDF of the datasets X and Y. A square is created with N data points on each side.
    pdfs: if True, also returns the marginal PDFs from the joint PDF.
    """

    H, xedges, yedges = np.histogram2d(X, Y, N, normed=True)
    # Use midpoints, not edges
    x = 0.5 * (xedges[1:] + xedges[:-1])
    y = 0.5 * (yedges[1:] + yedges[:-1])

    if pdfs == False:
        return H, x, y

    Xpdf = simpson(H, y, axis=1)
    Xpdf = Xpdf / simpson(Xpdf, x)
    Ypdf = simpson(H, x, axis=0)
    Ypdf = Ypdf / simpson(Ypdf, y)
    
    return H, Xpdf, Ypdf, x, y
