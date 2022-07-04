#!/usr/bin/env python3

import argparse
import json
import sys
import warnings
from importlib.resources import files
from typing import Callable

import awkward as ak
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import scipy
import scipy.stats as st
import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from iminuit.util import make_func_code
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import ArrayLike
from sweights import Cow, SWeight

# import rcdb


def load_rcdb(path: str) -> dict:
    """Load RCDB polarization information.

    Parameters
    ----------
    path: str
        Path to a tab-separated file with two columns, run number and polarization angle.

    Returns
    -------
    dict
        Dictionary containing run numbers as keys and their polarization angle as values.
    """
    df = np.loadtxt(path).astype(int)
    di = {key: value for key, value in df}
    return di


def validate_flattree(tree) -> None:
    """Verify that the required branches exist in the input `tree`.

    Parameters
    ----------
    tree
        Input tree read by uproot.

    Raises
    ------
    AssertionError
        If any required branches are missing.

    Warns
    -----
    UserWarning
        If any suggested branches are missing.
    """
    required_branches = {
        "Weight": "float",
        "E_Beam": "float",
        "Px_Beam": "float",
        "Py_Beam": "float",
        "Pz_Beam": "float",
        "NumFinalState": "int32_t",
        "E_FinalState": "float[]",
        "Px_FinalState": "float[]",
        "Py_FinalState": "float[]",
        "Pz_FinalState": "float[]",
    }
    semirequired_branches = {
        "RunNumber": "uint32_t",
        "EventNumber": "uint64_t",
        "ComboNumber": "uint32_t",
    }
    for key, value in required_branches.items():
        assert key in tree.keys(), f"Required branch not found: {key}!"
        assert (
            tree.get(key).typename == value
        ), f"Branch {key} has the wrong data type, found {tree.get(key).typename} but should have found {value}!"
    for key, value in semirequired_branches.items():
        if not key in tree.keys():
            warnings.warn(
                f"Branch {key} is not required but recommended to include RCDB polarization!"
            )
        else:
            assert (
                tree.get(key).typename == value
            ), f"Branch {key} has the wrong data type, found {tree.get(key).typename} but should have found {value}!"


def fast_fit(data: ArrayLike, config: dict, make_plot=True) -> tuple[Minuit, Callable]:
    """Fit using `numexpr` library (fast, but only uses `numexpr` functions).

    Configuration files are JSON formatted and must contain the following fields:
    1. `x`: a string which specifies which branch to use for the fitted variable
    2. `range`: A list with two values specifying the range of `x` to use in the fit
    3. `signal`: Python code to represent the signal function. This string can only
        include `numexpr` functions over `x`, as well as `xmin`, `xmax`, and `pi`.
        This function must be a normalized probability distribution function (PDF).
    4. `background`: Python code formatted like `signal` but containing a PDF for
        the background.
    5. `parameters`: A dictionary whose keys are parameters used in the fit functions.
        Each parameter is a dictionary containing the following fields:
        1. `initial`: The starting value for the parameter.
        2. (optional) `limits`: List with minimum and maximum allowed values.
        3. (optional) `fixed`: If set to "True", this parameter will be fixed in the fit.

    Parameters
    ----------
    data: ArrayLike
        Variable to fit.
    config: dict
        Dictionary from JSON configuration file describing the fit parameters.
    make_plot: bool
        Flag specifying whether or not to output a plot of the fit as fit.png.

    Returns
    -------
    mi: Minuit
        A fit object from iMinuit
    pdf: Callable
        A function of the form pdf(x, *args, comps=["sig", "bkg"]) which contains
        both the signal and background PDFs and takes in *mi.values for its *args.
    """
    signal = config.get("signal")
    background = config.get("background")
    assert signal, "No signal function provided in configuration!"
    assert background, "No background function provided in configuration!"
    x_range = config.get("range")
    if not x_range:
        x_range = [-np.inf, np.inf]
    params = ["Ns", "Nb"] + list(config.get("params").keys())

    def pdf(x, *args, comps=["sig", "bkg"]):
        loc = {param: arg for param, arg in zip(params, args)}
        loc.update({"pi": np.pi, "xmin": x_range[0], "xmax": x_range[1], "x": x})
        sig = int("sig" in comps)
        bkg = int("bkg" in comps)
        sig_eval = sig * loc["Ns"] * np.array(ne.evaluate(signal, local_dict=loc))
        bkg_eval = bkg * loc["Nb"] * np.array(ne.evaluate(background, local_dict=loc))
        return sig_eval + bkg_eval

    def model(x, Ns, Nb, *args):
        return (Ns + Nb, pdf(x, Ns, Nb, *args))

    f = lambda x, *args: model(x, *args)
    f.func_code = make_func_code(tuple(["x"] + [params]))

    cost = ExtendedUnbinnedNLL(data, f)
    inits = [len(data) / 2, len(data) / 2] + [
        param.get("initial") for param in config.get("params").values()
    ]
    mi = Minuit(cost, *inits, name=tuple(params))
    mi.limits["Ns"] = (0, len(data))
    mi.limits["Nb"] = (0, len(data))
    for key, value in config.get("params").items():
        plimit = value.get("limits")
        if plimit:
            mi.limits[key] = plimit
        pfixed = value.get("fixed")
        if pfixed:
            mi.fixed[key] = pfixed
    mi.migrad()
    mi.hesse()
    mi.minos()
    print(mi)
    if make_plot:
        fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw=dict(hspace=0, height_ratios=[4, 1]),
            figsize=[20, 10],
        )
        H, xe, _ = ax[0].hist(data, range=x_range, bins=100)
        ce = (xe[1:] + xe[:-1]) / 2
        de = np.diff(xe)[0]
        ax[0].plot(
            ce, pdf(ce, *mi.values, comps=["sig"]) * de, label="Signal", color="blue"
        )
        ax[0].plot(
            ce,
            pdf(ce, *mi.values, comps=["bkg"]) * de,
            label="Background",
            color="red",
            ls=":",
        )
        ax[0].plot(ce, pdf(ce, *mi.values) * de, label="Total", color="black")
        ax[0].legend()
        ax[1].scatter(ce, H - (pdf(ce, *mi.values) * de), color="black")
        fig.savefig("fit.png", dpi=80, bbox_inches="tight")
    return mi, pdf


def fit(data: ArrayLike, config: dict, make_plot=True) -> tuple[Minuit, Callable]:
    """Fit using standard Python `eval` (slow, but can call any NumPy or SciPy methods).

    Configuration files are JSON formatted and must contain the following fields:
    1. `x`: a string which specifies which branch to use for the fitted variable
    2. `range`: A list with two values specifying the range of `x` to use in the fit
    3. `signal`: Python code to represent the signal function. This string can include
        any native Python functions, as well as NumPy functions (using `np.<method>`),
        SciPy functions (using `scipy.<method>`), and SciPy statistical functions
        (using `st.<method>` in place of `scipy.stats.<method>` as shorthand). It
        additionally knows about the range of `x` through the built-in `xmin` and `xmax`
        as well as `pi` as shorthand for `np.pi`.
        This function must be a normalized probability distribution function (PDF).
    4. `background`: Python code formatted like `signal` but containing a PDF for
        the background.
    5. `parameters`: A dictionary whose keys are parameters used in the fit functions.
        Each parameter is a dictionary containing the following fields:
        1. `initial`: The starting value for the parameter.
        2. (optional) `limits`: List with minimum and maximum allowed values.
        3. (optional) `fixed`: If set to "True", this parameter will be fixed in the fit.

    Parameters
    ----------
    data: ArrayLike
        Variable to fit.
    config: dict
        Dictionary from JSON configuration file describing the fit parameters.
    make_plot: bool
        Flag specifying whether or not to output a plot of the fit as fit.png.

    Returns
    -------
    mi: Minuit
        A fit object from iMinuit
    pdf: Callable
        A function of the form pdf(x, *args, comps=["sig", "bkg"]) which contains
        both the signal and background PDFs and takes in *mi.values for its *args.
    """
    signal = config.get("signal")
    background = config.get("background")
    assert signal, "No signal function provided in configuration!"
    assert background, "No background function provided in configuration!"
    x_range = config.get("range")
    if not x_range:
        x_range = [-np.inf, np.inf]
    params = ["Ns", "Nb"] + list(config.get("params").keys())

    def pdf(x, *args, comps=["sig", "bkg"]):
        loc = {param: arg for param, arg in zip(params, args)}
        loc.update({"pi": np.pi, "xmin": x_range[0], "xmax": x_range[1], "x": x})
        loc.update({"np": np, "st": st, "scipy": scipy})
        sig = int("sig" in comps)
        bkg = int("bkg" in comps)
        sig_eval = sig * loc["Ns"] * np.array(eval(signal, loc))
        bkg_eval = bkg * loc["Nb"] * np.array(eval(background, loc))
        return sig_eval + bkg_eval

    def model(x, Ns, Nb, *args):
        return (Ns + Nb, pdf(x, Ns, Nb, *args))

    f = lambda x, *args: model(x, *args)
    f.func_code = make_func_code(tuple(["x"] + [params]))

    cost = ExtendedUnbinnedNLL(data, f)
    inits = [len(data) / 2, len(data) / 2] + [
        param.get("initial") for param in config.get("params").values()
    ]
    mi = Minuit(cost, *inits, name=tuple(params))
    mi.limits["Ns"] = (0, len(data))
    mi.limits["Nb"] = (0, len(data))
    for key, value in config.get("params").items():
        plimit = value.get("limits")
        if plimit:
            mi.limits[key] = plimit
        pfixed = value.get("fixed")
        if pfixed:
            mi.fixed[key] = pfixed
    mi.migrad()
    mi.hesse()
    mi.minos()
    print(mi)
    if make_plot:
        fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw=dict(hspace=0, height_ratios=[4, 1]),
            figsize=[20, 10],
        )
        H, xe, _ = ax[0].hist(data, range=x_range, bins=100)
        ce = (xe[1:] + xe[:-1]) / 2
        de = np.diff(xe)[0]
        ax[0].plot(
            ce, pdf(ce, *mi.values, comps=["sig"]) * de, label="Signal", color="blue"
        )
        ax[0].plot(
            ce,
            pdf(ce, *mi.values, comps=["bkg"]) * de,
            label="Background",
            color="red",
            ls=":",
        )
        ax[0].plot(ce, pdf(ce, *mi.values) * de, label="Total", color="black")
        ax[0].legend()
        ax[1].scatter(ce, H - (pdf(ce, *mi.values) * de), color="black")
        fig.savefig("fit.png", dpi=80, bbox_inches="tight")
    return mi, pdf


def splot(tree, cut_list: list[str], config: dict) -> tuple[str, ArrayLike]:
    """Generate sPlot weights.

    See `fit` for configuration file formatting.

    Parameters
    ----------
    tree
        `uproot` TTree to use.
    cut_list: list[str]
        List of strings describing cuts, formatted as `varname <comparitor> value`.
    config: dict
        Dictionary from JSON configuration file describing the fit parameters.

    Returns
    -------
    cuts: str
        String containing cuts formatted as `(cut #1) & (cut #2) & ...` from `cut_list`.
        This method also adds a cut on the range of the fitted variable from the
        configuration file.
    ArrayLike
        A list of weights corresponding to each event based on the sPlot weighting method.
    """

    varname = config.get("x")
    assert (
        varname in tree.keys()
    ), f"The variable {varname} is not a valid branch in the TTree provided!"
    x_range = config.get("range")
    if not x_range:
        x_range = [-np.inf, np.inf]
        cuts = " & ".join([f"({c})" for c in cut_list])
    else:
        cuts = " & ".join(
            [
                f"({c})"
                for c in cut_list
                + [f"{varname} > {x_range[0]}", f"{varname} < {x_range[1]}"]
            ]
        )
    data = tree.arrays([varname], cuts, library="np").get(varname)
    mi, pdf = fit(data, config)
    spdf = lambda x: pdf(x, *mi.values, comps=["sig"])
    bpdf = lambda x: pdf(x, *mi.values, comps=["bkg"])
    weighter = SWeight(
        data,
        [spdf, bpdf],
        [mi.values["Ns"], mi.values["Nb"]],
        (x_range,),
        method="summation",
        compnames=("sig", "bkg"),
        verbose=True,
        checks=True,
    )
    return cuts, weighter.get_weight(0, data)


def cows(tree, cut_list: list[str], config: dict) -> tuple[str, ArrayLike]:
    """Generate COWs weights.

    See `fit` for configuration file formatting.

    Parameters
    ----------
    tree
        `uproot` TTree to use.
    cut_list: list[str]
        List of strings describing cuts, formatted as `varname <comparitor> value`.
    config: dict
        Dictionary from JSON configuration file describing the fit parameters.

    Returns
    -------
    cuts: str
        String containing cuts formatted as `(cut #1) & (cut #2) & ...` from `cut_list`.
        This method also adds a cut on the range of the fitted variable from the
        configuration file.
    ArrayLike
        A list of weights corresponding to each event based on the COWs weighting method.
    """
    Im = 1  # need to figure this out
    varname = config.get("x")
    assert (
        varname in tree.keys()
    ), f"The variable {varname} is not a valid branch in the TTree provided!"
    x_range = config.get("range")
    if not x_range:
        x_range = [-np.inf, np.inf]
        cuts = " & ".join([f"({c})" for c in cut_list])
    else:
        cuts = " & ".join(
            [
                f"({c})"
                for c in cut_list
                + [f"{varname} > {x_range[0]}", f"{varname} < {x_range[1]}"]
            ]
        )
    data = tree.arrays([varname], cuts, library="np").get(varname)
    mi, pdf = fit(data, config)
    spdf = lambda x: pdf(x, *mi.values, comps=["sig"])
    bpdf = lambda x: pdf(x, *mi.values, comps=["bkg"])
    weighter = Cow(x_range, spdf, bpdf, Im, verbose=True)
    return cuts, weighter.get_weight(0, data)


def write_tree(
    tree, cuts: str, outpath: str, weights=None, rcdb=True, min_pol=0.1, output_branches=None
) -> None:
    """Writes results to a new FlatTree

    Parameters
    ----------
    tree
        Input FlatTree
    cuts: str
        String containing cuts formatted as `(cut #1) & (cut #2) & ...`.
    outpath: str
        Path to which to save the resulting FlatTree.
    weights: ArrayLike
        Event-by-event weights to apply to the `Weight` branch in the output TTree.
    rcdb: bool
        Flag to add energy-dependent polarization information into the Beam 3-momentum.
    min_pol: float
        Minimum polarization fraction to use.
    output_branches: list[str]
        Names of additional branches to output (can be renamed by formatting the string
        as <old branch name>:<new branch name>).
    """
    branches = [
        "Weight",
        "E_Beam",
        "Px_Beam",
        "Py_Beam",
        "Pz_Beam",
        "NumFinalState",
        "E_FinalState",
        "Px_FinalState",
        "Py_FinalState",
        "Pz_FinalState",
    ]
    if output_branches:
        added_branches = {branch.split(":")[0]: branch.split(":")[1] for branch in output_branches}
        branches += list(added_branches.keys())
    if rcdb:
        branches += ["RunNumber"]
    df = tree.arrays(branches, cuts, library="ak")
    finalstate = ak.zip(
        {
            "E": df["E_FinalState"],
            "Px": df["Px_FinalState"],
            "Py": df["Py_FinalState"],
            "Pz": df["Pz_FinalState"],
        }
    ) 
    datatypes = {
        "Weight": "float32",
        "E_Beam": "float32",
        "Px_Beam": "float32",
        "Py_Beam": "float32",
        "Pz_Beam": "float32",
        "FinalState": finalstate.type,
    }
    datatypes.update({value: df[key].type for key, value in added_branches.items()})
    outfile = uproot.recreate(outpath)
    # AmpTools convention on names
    outfile.mktree(
        "kin",
        datatypes,
        counter_name=lambda s: f"Num{s}",
        field_name=lambda outer, inner: inner if outer == "" else f"{inner}_{outer}",
    )
    weight = df["Weight"].to_numpy()
    if not weights is None:
        sign = [
            -1 if signed else 1
            for signed in np.logical_or(np.sign(weight) < 0, np.sign(weights) < 0)
        ]
        weight = np.abs(weight) * np.abs(weights) * sign
    print("Writing...")
    valid_runs = np.ones_like(weight, dtype=bool)
    if rcdb:
        pols = [0, 45, 90, 135]
        pol_keys = [f"hPol{i}" for i in pols]
        is_S17 = np.array([int(30000 <= n < 40000) for n in df["RunNumber"]])
        is_S18 = np.array([int(40000 <= n < 50000) for n in df["RunNumber"]])
        is_F18 = np.array([int(50000 <= n < 60000) for n in df["RunNumber"]])
        di = load_rcdb(files("splot").joinpath("rcdb"))
        polarization = np.array([di.get(n, -1) for n in df["RunNumber"]])
        angle0 = 1.8 * is_S17 + 4.1 * is_S18 + 3.3 * is_F18
        angle45 = 47.9 * is_S17 + 48.5 * is_S18 + 48.3 * is_F18
        angle90 = 94.5 * is_S17 + 94.2 * is_S18 + 92.9 * is_F18
        angle135 = -41.6 * is_S17 + -42.4 * is_S18 + -42.1 * is_F18
        angle = (
            angle0 * (polarization == 0)
            + angle45 * (polarization == 45)
            + angle90 * (polarization == 90)
            + angle135 * (polarization == 135)
        )
        valid_runs = np.bitwise_and(angle != 0, polarization >= min_pol)
        with uproot.open(files("splot").joinpath("S17.root")) as S17, uproot.open(
            files("splot").joinpath("S18.root")
        ) as S18, uproot.open(files("splot").joinpath("F18.root")) as F18:
            PARA_0 = np.sum(
                [
                    is_run
                    * run["hPol0"].to_numpy()[0][
                        np.searchsorted(run["hPol0"].to_numpy()[1], df["E_Beam"]) - 1
                    ]
                    for is_run, run in zip([is_S17, is_S18, is_F18], [S17, S18, F18])
                ],
                axis=0,
            )
            PERP_45 = np.sum(
                [
                    is_run
                    * run["hPol45"].to_numpy()[0][
                        np.searchsorted(run["hPol45"].to_numpy()[1], df["E_Beam"]) - 1
                    ]
                    for is_run, run in zip([is_S17, is_S18, is_F18], [S17, S18, F18])
                ],
                axis=0,
            )
            PERP_90 = np.sum(
                [
                    is_run
                    * run["hPol90"].to_numpy()[0][
                        np.searchsorted(run["hPol90"].to_numpy()[1], df["E_Beam"]) - 1
                    ]
                    for is_run, run in zip([is_S17, is_S18, is_F18], [S17, S18, F18])
                ],
                axis=0,
            )
            PARA_135 = np.sum(
                [
                    is_run
                    * run["hPol135"].to_numpy()[0][
                        np.searchsorted(run["hPol135"].to_numpy()[1], df["E_Beam"]) - 1
                    ]
                    for is_run, run in zip([is_S17, is_S18, is_F18], [S17, S18, F18])
                ],
                axis=0,
            )
        polfrac = (
            PARA_0 * (polarization == 0)
            + PERP_45 * (polarization == 45)
            + PERP_90 * (polarization == 90)
            + PARA_135 * (polarization == 135)
        )
        df["Px_Beam"] = polfrac * np.cos(angle * np.pi / 180)
        df["Py_Beam"] = polfrac * np.sin(angle * np.pi / 180)
        df["Pz_Beam"] = np.zeros_like(df["Pz_Beam"])
    extension = {
        "Weight": weight[valid_runs],
        "E_Beam": df["E_Beam"][valid_runs],
        "Px_Beam": df["Px_Beam"][valid_runs],
        "Py_Beam": df["Py_Beam"][valid_runs],
        "Pz_Beam": df["Pz_Beam"][valid_runs],
        "FinalState": finalstate[valid_runs]
        }
    if output_branches:
        extension.update({value: df[key][valid_runs] for key, value in added_branches.items()})
    outfile["kin"].extend(extension)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input flat tree")
    parser.add_argument("--splot", help="path to config file")
    parser.add_argument("--cows", help="path to config file")
    parser.add_argument(
        "-l", "--list", action="store_true", help="list branch names in input tree"
    )
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument(
        "-c",
        "--cuts",
        type=str,
        help="cuts to apply (string enclosed in quotes with commas separating each selection)",
    )
    parser.add_argument(
        "--norcdb",
        action="store_true",
        help="don't use RCDB information to include polarization",
    )
    parser.add_argument(
        "--minpol", default=0.1, help="minimum polarization fraction to use"
    )
    parser.add_argument("--output-branches", nargs="+", type=str, help="branches to include in output tree, can be renamed by inputting them as <old_name>:<new_name>")
    args = parser.parse_args()
    with uproot.open(args.input) as f:
        t = f.get(f.keys()[0])
        if args.list:
            t.show()
            sys.exit(0)
        validate_flattree(t)
        print("TTree has passed validation!")
        cut_list = ["Weight != 0"]
        if args.cuts:
            cut_list.extend([c.strip() for c in args.cuts.split(",")])
        cuts = " & ".join([f"({c})" for c in cut_list])
        print("Applying the following selection criteria:")
        print(cuts)
        weights = None
        if args.splot or args.cows:
            fname = args.splot if args.splot else args.cows
            try:
                with open(fname) as config_file:
                    config = json.load(config_file)
            except FileNotFoundError:
                print(f"File {fname} not found!")
                sys.exit(1)
            except OSError:
                print(f"OS error occured while opening {fname}!")
                sys.exit(1)
            except Exception as err:
                print(f"Unexpected error occurred while reading {fname}:", repr(err))
                sys.exit(1)
            if args.splot:
                print("Computing sPlot Weights")
                cuts, weights = splot(t, cut_list, config)
            else:
                print("Computing COWs Weights")
                cuts, weights = cows(t, cut_list, config)
        if args.output:
            output_branches = None
            if args.output_branches:
                output_branches = []
                for branch in args.output_branches:
                    if ":" in branch:
                        output_branches.append(branch)
                    else:
                        output_branches.append(f"{branch}:{branch}")
            write_tree(
                t,
                cuts,
                args.output,
                weights=weights,
                rcdb=(not args.norcdb),
                min_pol=args.minpol,
                output_branches=output_branches
            )
            print(f"Output tree written to {args.output} !")


if __name__ == "__main__":
    main()
