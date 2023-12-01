#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power spectrum of random filamentary and Gaussian RM maps.
"""
__author__ = "Andrea Bracco"
from typing import NamedTuple, Tuple

import numpy as np
import pylab as plt
import pywavan


def powspec2D(im: np.ndarray, im1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes 2D angular power spectra between the input maps im and im1.

    Args:
        im (np.ndarray): Map 1
        im1 (np.ndarray): Map 2

    Returns:
        Tuple[np.ndarray, np.ndarray]: k and power
    """

    nx = np.shape(im)[0]
    nz = np.shape(im)[1]

    x = np.linspace(-1, 1, nx)
    z = np.linspace(-1, 1, nz)
    dk_x = 1.0 / (x[-1] - x[0])
    dk_z = 1.0 / (z[-1] - z[0])
    k_x = np.linspace(-nx / 2, nx / 2, nx)
    k_z = np.linspace(-nz / 2, nz / 2, nz)
    kk_x, kk_z = np.meshgrid(k_x, k_z, indexing="ij")

    kk = np.sqrt(kk_x**2 + kk_z**2)
    dk = np.min([dk_x, dk_z])
    k_shells = np.arange(0, np.sqrt(np.max(kk_x**2 + kk_z**2)) + dk, dk)
    power = np.zeros(len(k_shells))

    im_k = np.fft.fftshift(np.fft.fft2(im))
    im1_k = np.fft.fftshift(np.fft.fft2(im1))

    for j, k in enumerate(k_shells):
        mask0 = abs(kk - k - dk) <= dk
        power[j] = np.mean((im_k[mask0]) * np.conj(im1_k[mask0])) / (nx * nz)

    power[np.isnan(power)] = 0

    return k_shells, power


def compism(spec: float, nx: int, ny: int) -> np.ndarray:
    """Generate a Gaussian random field with power-law power spectrum k^(-spec).

    Args:
        spec (float): Spectral index
        nx (int): Number of pixels in x
        ny (int): Number of pixels in y

    Returns:
        np.ndarray: Random field image
    """

    # Define the box size.
    x = np.linspace(0, nx - 1, nx)

    # Generate a random afield (white noise).
    phase = np.random.random([nx, ny]) * 2 * np.pi

    dk = 1.0 / (x[-1] - x[0])
    k = np.linspace(-1, 1, len(x)) * dk * len(x) / 2.0
    kk_x, kk_y = np.meshgrid(k, k, indexing="ij")
    kk = np.sqrt(kk_x**2 + kk_y**2)

    # Apply a fourier filter.
    a_k = kk ** (-spec / 2.0)

    # Apply teh inverse transform.
    m_k = a_k * np.cos(phase) + 1j * a_k * np.sin(phase)
    mout = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m_k))))

    return mout


class RandFilRM(NamedTuple):
    """
    Random filamentary and Gaussian RM maps with power-law power
    spectrum k^(-spec) and standard deviation sigmaRM.
    """

    mRM_r: np.ndarray
    """
    Gaussian RM map.
    """
    mRM_f: np.ndarray
    """
    Filamentary RM map.
    """
    k0: np.ndarray
    """
    Wavenumbers of Gaussian RM map.
    """
    p0: np.ndarray
    """
    Power spectrum of Gaussian RM map.
    """
    k1: np.ndarray
    """
    Wavenumbers of filamentary RM map.
    """
    p1: np.ndarray
    """
    Power spectrum of filamentary RM map.
    """
    res0: np.ndarray
    """
    Slope of Gaussian RM map.
    """
    res1: np.ndarray
    """
    Slope of filamentary RM map.
    """


def ranfil_ab(
    nax: int = 256,
    spec: float = 3.0,
    ndiri: int = 13,
    plot: bool = False,
    cmap="coolwarm",
) -> RandFilRM:
    """Generate a filamentary random field from input Gaussian field with
    power-law power spectrum k^(-spec).

    Args:
        nax (int, optional): Box size. Defaults to 256.
        spec (float, optional): Spectral index. Defaults to 3.0.
        ndiri (int, optional): Number of directions to sample. Defaults to 13.
        plot (bool, optional): Show plots. Defaults to False.
        cmap (str, optional): Colormap. Defaults to "coolwarm".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Random field, filamentary field, power spectrum of random field, power spectrum of filamentary field, slope of random field, slope of filamentary field
    """
    mran = compism(spec, nax, nax)
    wt, s11a, wavk, s1a, q = pywavan.fan_trans(
        mran, reso=1, q=0, qdyn=False, Ndir=ndiri, angular=True
    )
    prod = 1
    beta = (-spec + 1.6) / 2.0
    for i in range(np.shape(wt)[0]):
        prod = prod * np.exp(
            wt[i, :, :, :] / np.std(wt[i, :, :, :]) * wavk[i] ** (beta)
        )
    sum0 = np.sum(prod, axis=0)
    mfil = np.log10(np.abs(sum0) / np.shape(wt)[1])
    k1, p1 = powspec2D((mfil), (mfil), 1)
    k0, p0 = powspec2D((mran), (mran), 1)
    pmod = (k0 + 1e-5) ** (-spec)
    if plot:
        plt.figure(1)
        plt.loglog(k1, p1 / np.percentile(p1, 99), label="filaments")
        plt.loglog(k0, p0 / np.percentile(p0, 99), label="random")
        plt.loglog(k0, pmod / np.percentile(pmod, 99), label="input")
        plt.ylim(0, 1e4)
        plt.legend()
        plt.figure(2)
        plt.subplot(121)
        plt.imshow(mran, cmap=cmap)
        plt.subplot(122)
        plt.imshow(mfil, cmap=cmap)
        plt.tight_layout()
    res0 = np.polyfit(np.log10(k0[1 : len(k0) - 1]), np.log10(p0[1 : len(k0) - 1]), 1)
    res1 = np.polyfit(np.log10(k1[1 : len(k1) - 1]), np.log10(p1[1 : len(k1) - 1]), 1)
    return RandFilRM(
        mRM_r=mran,
        mRM_f=mfil,
        k0=k0[1 : len(k0) - 1],
        p0=p0[1 : len(k0) - 1],
        k1=k1[1 : len(k1) - 1],
        p1=p1[1 : len(k1) - 1],
        res0=res0[0],
        res1=res1[0],
    )


def randfil_RM(nax=256, spec=1.6, ndiri=13, sigmaRM=10, plot=False):
    """
    Generate random filamentary and Gaussian rotation measure maps with
    standard deviations defined by sigmaRM and power spectrum defined
    by the spectral index spec.
    """

    mRM_tmp = (
        ranfil_ab(spec=spec, ndiri=ndiri, nax=nax).mRM_f
        - ranfil_ab(spec=spec, ndiri=ndiri, nax=nax).mRM_f
    )
    mRM_f = mRM_tmp / np.std(mRM_tmp) * sigmaRM  # filamentary RM
    mRM_tmp_r = (
        ranfil_ab(spec=spec, ndiri=ndiri, nax=nax).mRM_r
        - ranfil_ab(spec=spec, ndiri=ndiri, nax=nax).mRM_r
    )
    mRM_r = mRM_tmp_r / np.std(mRM_tmp_r) * sigmaRM  # Gaussian RM

    if plot:
        plt.figure(figsize=[8, 8])
        plt.subplot(221)
        plt.imshow(mRM_r, cmap="seismic", origin="lower")
        plt.colorbar()
        plt.title(r"Gaussian RM [rad m$^{-2}$]")
        plt.subplot(223)
        plt.hist(mRM_r.flatten(), bins=100, alpha=0.5, label="Gaussian RM")
        plt.hist(mRM_f.flatten(), bins=100, alpha=0.5, label="Filamentary RM")
        plt.xlabel(r"RM [rad m$^{-2}$]")
        plt.ylabel("histogram")
        plt.legend(fontsize=7)
        plt.subplot(222)
        plt.imshow(mRM_f, cmap="seismic", origin="lower")
        plt.colorbar()
        plt.title(r"Filamentary RM [rad m$^{-2}$]")
        k3, p3 = powspec2D((mRM_f), (mRM_f), 1)
        k2, p2 = powspec2D((mRM_r), (mRM_r), 1)
        pmod = 100 * (k3[1:]) ** (-spec)
        plt.subplot(224)
        plt.loglog(
            2 * np.pi * k3 / (nax), p3 / np.percentile(p3, 99), label="Filamentary RM"
        )
        plt.loglog(
            2 * np.pi * k2 / (nax), p2 / np.percentile(p2, 99), label="Gaussian RM"
        )
        plt.loglog(
            2 * np.pi * k3[1:] / (nax),
            pmod / np.percentile(pmod, 99),
            label="input slope",
        )
        plt.ylabel("Normalized power")
        plt.xlabel(r"$k$ [px$^{-1}$]")
        plt.legend()
        plt.tight_layout()

    return mRM_r, mRM_f
