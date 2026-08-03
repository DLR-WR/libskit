/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <unordered_map>

/**
 * @file constants.h
 * @brief Physical constants and element definitions for atomic spectroscopy calculations.
 *
 * This file contains fundamental physical constants in SI units and defines
 * enumerations and conversion utilities for chemical elements used throughout
 * the libskit library.
 */

namespace libskit {

/**
 * @defgroup PhysicalConstants Physical Constants
 * @brief Fundamental physical constants in SI units.
 *
 * These constants are used throughout the library for atomic physics calculations,
 * plasma modeling, and spectroscopy simulations. All values are taken from
 * NIST CODATA.
 * @{
 */

/// @brief Boltzmann constant in J/K
constexpr double BOLTZMANN_IN_J_K = 1.380649e-23;
/// @brief Boltzmann constant in eV/K
constexpr double BOLTZMANN_IN_EV_K = 8.617333262e-5;
/// @brief Reduced Planck constant (ℏ) in J⋅s
constexpr double HBAR = 1.054571817e-34;
/// @brief Planck constant (h) in J⋅s
constexpr double PLANCK_CONSTANT = 6.62607015e-34;
/// @brief Speed of light in vacuum in m/s
constexpr double SPEED_OF_LIGHT = 299792458;
/// @brief Electron mass in kg
constexpr double ELECTRON_MASS = 9.1093837015e-31;
/// @brief Elementary charge in C
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;
/// @brief Atomic mass unit in kg
constexpr double ATOMIC_MASS = 1.66053906892e-27;
/// @brief Fine structure constant (dimensionless)
constexpr double FINE_STRUCTURE_CONSTANT = 7.2973525694e-3;
/// @brief Bohr radius in m
constexpr double BOHR_RADIUS = 5.29177210544e-11;
/// @brief Hydrogen ionization energy (Rydberg energy) in Joules
constexpr double RYDBERG_ENERGY_J = 2.1798723611030e-18;
/// @brief Hydrogen ionization energy (Rydberg energy) in eV
constexpr double RYDBERG_ENERGY_EV = 13.605693122990;

/** @} */ // end PhysicalConstants group

/**
 * @brief Enumeration of chemical elements.
 *
 * This enumeration provides integer codes for chemical elements. The
 * enumeration follows the standard periodic table ordering by atomic number.
 */
enum Element {
  H,   ///< Hydrogen (Z=1)
  He,  ///< Helium (Z=2)
  Li,  ///< Lithium (Z=3)
  Be,  ///< Beryllium (Z=4)
  B,   ///< Boron (Z=5)
  C,   ///< Carbon (Z=6)
  N,   ///< Nitrogen (Z=7)
  O,   ///< Oxygen (Z=8)
  F,   ///< Fluorine (Z=9)
  Ne,  ///< Neon (Z=10)
  Na,  ///< Sodium (Z=11)
  Mg,  ///< Magnesium (Z=12)
  Al,  ///< Aluminum (Z=13)
  Si,  ///< Silicon (Z=14)
  P,   ///< Phosphorus (Z=15)
  S,   ///< Sulfur (Z=16)
  Cl,  ///< Chlorine (Z=17)
  Ar,  ///< Argon (Z=18)
  K,   ///< Potassium (Z=19)
  Ca,  ///< Calcium (Z=20)
  Sc,
  Ti,
  V,
  Cr,
  Mn,
  Fe,
  Co,
  Ni,
  Cu,
  Zn,
  Ga,
  Ge,
  As,
  Se,
  Br,
  Kr,
  Rb,
  Sr,
  Y,
  Zr,
  Nb,
  Mo,
  Tc,
  Ru,
  Rh,
  Pd,
  Ag,
  Cd,
  In,
  Sn,
  Sb,
  Te,
  I,
  Xe,
  Cs,
  Ba,
  La,
  Ce,
  Pr,
  Nd,
  Pm,
  Sm,
  Eu,
  Gd,
  Tb,
  Dy,
  Ho,
  Er,
  Tm,
  Yb,
  Lu,
  Hf,
  Ta,
  W,
  Re,
  Os,
  Ir,
  Pt,
  Au,
  Hg,
  Tl,
  Pb,
  Bi,
  Po,
  At,
  Rn,
  Fr,
  Ra,
  Ac,
  Th,
  Pa,
  U,
  Np,
  Pu,
  Am,
  Cm,
  Bk,
  Cf,
  Es,
  Fm,
  Md,
  No,
  Lr,
  Rf,
  Db,
  Sg,
  Bh,
  Hs,
  Mt,
  Ds,
  Rg,
  Cn,
  Nh,
  Fl,
  Mc,
  Lv,
  Ts,
  Og,

};

/**
 * @brief Conversion map from Element enumeration to string symbol.
 *
 * This unordered map provides lookup for converting Element
 * enumeration values to their corresponding chemical symbols (e.g., Fe -> "Fe").
 */
const std::unordered_map<Element, std::string> element_to_string = {
    {H, "H"},   {He, "He"}, {Li, "Li"}, {Be, "Be"}, {B, "B"},   {C, "C"},
    {N, "N"},   {O, "O"},   {F, "F"},   {Ne, "Ne"}, {Na, "Na"}, {Mg, "Mg"},
    {Al, "Al"}, {Si, "Si"}, {P, "P"},   {S, "S"},   {Cl, "Cl"}, {Ar, "Ar"},
    {K, "K"},   {Ca, "Ca"}, {Sc, "Sc"}, {Ti, "Ti"}, {V, "V"},   {Cr, "Cr"},
    {Mn, "Mn"}, {Fe, "Fe"}, {Co, "Co"}, {Ni, "Ni"}, {Cu, "Cu"}, {Zn, "Zn"},
    {Ga, "Ga"}, {Ge, "Ge"}, {As, "As"}, {Se, "Se"}, {Br, "Br"}, {Kr, "Kr"},
    {Rb, "Rb"}, {Sr, "Sr"}, {Y, "Y"},   {Zr, "Zr"}, {Nb, "Nb"}, {Mo, "Mo"},
    {Tc, "Tc"}, {Ru, "Ru"}, {Rh, "Rh"}, {Pd, "Pd"}, {Ag, "Ag"}, {Cd, "Cd"},
    {In, "In"}, {Sn, "Sn"}, {Sb, "Sb"}, {Te, "Te"}, {I, "I"},   {Xe, "Xe"},
    {Cs, "Cs"}, {Ba, "Ba"}, {La, "La"}, {Ce, "Ce"}, {Pr, "Pr"}, {Nd, "Nd"},
    {Pm, "Pm"}, {Sm, "Sm"}, {Eu, "Eu"}, {Gd, "Gd"}, {Tb, "Tb"}, {Dy, "Dy"},
    {Ho, "Ho"}, {Er, "Er"}, {Tm, "Tm"}, {Yb, "Yb"}, {Lu, "Lu"}, {Hf, "Hf"},
    {Ta, "Ta"}, {W, "W"},   {Re, "Re"}, {Os, "Os"}, {Ir, "Ir"}, {Pt, "Pt"},
    {Au, "Au"}, {Hg, "Hg"}, {Tl, "Tl"}, {Pb, "Pb"}, {Bi, "Bi"}, {Po, "Po"},
    {At, "At"}, {Rn, "Rn"}, {Fr, "Fr"}, {Ra, "Ra"}, {Ac, "Ac"}, {Th, "Th"},
    {Pa, "Pa"}, {U, "U"},   {Np, "Np"}, {Pu, "Pu"}, {Am, "Am"}, {Cm, "Cm"},
    {Bk, "Bk"}, {Cf, "Cf"}, {Es, "Es"}, {Fm, "Fm"}, {Md, "Md"}, {No, "No"},
    {Lr, "Lr"}, {Rf, "Rf"}, {Db, "Db"}, {Sg, "Sg"}, {Bh, "Bh"}, {Hs, "Hs"},
    {Mt, "Mt"}, {Ds, "Ds"}, {Rg, "Rg"}, {Cn, "Cn"}, {Nh, "Nh"}, {Fl, "Fl"},
    {Mc, "Mc"}, {Lv, "Lv"}, {Ts, "Ts"}, {Og, "Og"}

};

/**
 * @brief Conversion map from string symbol to Element enumeration.
 *
 * This unordered map provides lookup for converting chemical
 * symbols to their corresponding Element enumeration values (e.g., "Fe" -> Fe).
 */
const std::unordered_map<std::string, Element> string_to_element = {
    {"H", H},   {"He", He}, {"Li", Li}, {"Be", Be}, {"B", B},   {"C", C},
    {"N", N},   {"O", O},   {"F", F},   {"Ne", Ne}, {"Na", Na}, {"Mg", Mg},
    {"Al", Al}, {"Si", Si}, {"P", P},   {"S", S},   {"Cl", Cl}, {"Ar", Ar},
    {"K", K},   {"Ca", Ca}, {"Sc", Sc}, {"Ti", Ti}, {"V", V},   {"Cr", Cr},
    {"Mn", Mn}, {"Fe", Fe}, {"Co", Co}, {"Ni", Ni}, {"Cu", Cu}, {"Zn", Zn},
    {"Ga", Ga}, {"Ge", Ge}, {"As", As}, {"Se", Se}, {"Br", Br}, {"Kr", Kr},
    {"Rb", Rb}, {"Sr", Sr}, {"Y", Y},   {"Zr", Zr}, {"Nb", Nb}, {"Mo", Mo},
    {"Tc", Tc}, {"Ru", Ru}, {"Rh", Rh}, {"Pd", Pd}, {"Ag", Ag}, {"Cd", Cd},
    {"In", In}, {"Sn", Sn}, {"Sb", Sb}, {"Te", Te}, {"I", I},   {"Xe", Xe},
    {"Cs", Cs}, {"Ba", Ba}, {"La", La}, {"Ce", Ce}, {"Pr", Pr}, {"Nd", Nd},
    {"Pm", Pm}, {"Sm", Sm}, {"Eu", Eu}, {"Gd", Gd}, {"Tb", Tb}, {"Dy", Dy},
    {"Ho", Ho}, {"Er", Er}, {"Tm", Tm}, {"Yb", Yb}, {"Lu", Lu}, {"Hf", Hf},
    {"Ta", Ta}, {"W", W},   {"Re", Re}, {"Os", Os}, {"Ir", Ir}, {"Pt", Pt},
    {"Au", Au}, {"Hg", Hg}, {"Tl", Tl}, {"Pb", Pb}, {"Bi", Bi}, {"Po", Po},
    {"At", At}, {"Rn", Rn}, {"Fr", Fr}, {"Ra", Ra}, {"Ac", Ac}, {"Th", Th},
    {"Pa", Pa}, {"U", U},   {"Np", Np}, {"Pu", Pu}, {"Am", Am}, {"Cm", Cm},
    {"Bk", Bk}, {"Cf", Cf}, {"Es", Es}, {"Fm", Fm}, {"Md", Md}, {"No", No},
    {"Lr", Lr}, {"Rf", Rf}, {"Db", Db}, {"Sg", Sg}, {"Bh", Bh}, {"Hs", Hs},
    {"Mt", Mt}, {"Ds", Ds}, {"Rg", Rg}, {"Cn", Cn}, {"Nh", Nh}, {"Fl", Fl},
    {"Mc", Mc}, {"Lv", Lv}, {"Ts", Ts}, {"Og", Og}

};

} // namespace libskit

#endif /* CONSTANTS_H */
