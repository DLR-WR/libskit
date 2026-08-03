/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef ATOM_INFO_H
#define ATOM_INFO_H

#include <string>
#include <unordered_map>

namespace libskit {

struct AtomInfo {
  unsigned int element_number;
  double atomic_weight;
};

const std::unordered_map<std::string, AtomInfo> ATOM_INFO = {
    {"H", AtomInfo(1, 1.007)},     {"He", AtomInfo(2, 4.002)},
    {"Li", AtomInfo(3, 6.941)},    {"Be", AtomInfo(4, 9.012)},
    {"B", AtomInfo(5, 10.811)},    {"C", AtomInfo(6, 12.011)},
    {"N", AtomInfo(7, 14.007)},    {"O", AtomInfo(8, 15.999)},
    {"F", AtomInfo(9, 18.998)},    {"Ne", AtomInfo(10, 20.18)},
    {"Na", AtomInfo(11, 22.99)},   {"Mg", AtomInfo(12, 24.305)},
    {"Al", AtomInfo(13, 26.982)},  {"Si", AtomInfo(14, 28.086)},
    {"P", AtomInfo(15, 30.974)},   {"S", AtomInfo(16, 32.065)},
    {"Cl", AtomInfo(17, 35.453)},  {"Ar", AtomInfo(18, 39.948)},
    {"K", AtomInfo(19, 39.098)},   {"Ca", AtomInfo(20, 40.078)},
    {"Sc", AtomInfo(21, 44.956)},  {"Ti", AtomInfo(22, 47.867)},
    {"V", AtomInfo(23, 50.942)},   {"Cr", AtomInfo(24, 51.996)},
    {"Mn", AtomInfo(25, 54.938)},  {"Fe", AtomInfo(26, 55.845)},
    {"Co", AtomInfo(27, 58.933)},  {"Ni", AtomInfo(28, 58.693)},
    {"Cu", AtomInfo(29, 63.546)},  {"Zn", AtomInfo(30, 65.38)},
    {"Ga", AtomInfo(31, 69.723)},  {"Ge", AtomInfo(32, 72.64)},
    {"As", AtomInfo(33, 74.922)},  {"Se", AtomInfo(34, 78.96)},
    {"Br", AtomInfo(35, 79.904)},  {"Kr", AtomInfo(36, 83.798)},
    {"Rb", AtomInfo(37, 85.468)},  {"Sr", AtomInfo(38, 87.62)},
    {"Y", AtomInfo(39, 88.906)},   {"Zr", AtomInfo(40, 91.224)},
    {"Nb", AtomInfo(41, 92.906)},  {"Mo", AtomInfo(42, 95.96)},
    {"Tc", AtomInfo(43, 98)},      {"Ru", AtomInfo(44, 101.07)},
    {"Rh", AtomInfo(45, 102.906)}, {"Pd", AtomInfo(46, 106.42)},
    {"Ag", AtomInfo(47, 107.868)}, {"Cd", AtomInfo(48, 112.411)},
    {"In", AtomInfo(49, 114.818)}, {"Sn", AtomInfo(50, 118.71)},
    {"Sb", AtomInfo(51, 121.76)},  {"Te", AtomInfo(52, 127.6)},
    {"I", AtomInfo(53, 126.904)},  {"Xe", AtomInfo(54, 131.293)},
    {"Cs", AtomInfo(55, 132.905)}, {"Ba", AtomInfo(56, 137.327)},
    {"La", AtomInfo(57, 138.905)}, {"Ce", AtomInfo(58, 140.116)},
    {"Pr", AtomInfo(59, 140.908)}, {"Nd", AtomInfo(60, 144.242)},
    {"Pm", AtomInfo(61, 145)},     {"Sm", AtomInfo(62, 150.36)},
    {"Eu", AtomInfo(63, 151.964)}, {"Gd", AtomInfo(64, 157.25)},
    {"Tb", AtomInfo(65, 158.925)}, {"Dy", AtomInfo(66, 162.5)},
    {"Ho", AtomInfo(67, 164.93)},  {"Er", AtomInfo(68, 167.259)},
    {"Tm", AtomInfo(69, 168.934)}, {"Yb", AtomInfo(70, 173.054)},
    {"Lu", AtomInfo(71, 174.967)}, {"Hf", AtomInfo(72, 178.49)},
    {"Ta", AtomInfo(73, 180.948)}, {"W", AtomInfo(74, 183.84)},
    {"Re", AtomInfo(75, 186.207)}, {"Os", AtomInfo(76, 190.23)},
    {"Ir", AtomInfo(77, 192.217)}, {"Pt", AtomInfo(78, 195.084)},
    {"Au", AtomInfo(79, 196.967)}, {"Hg", AtomInfo(80, 200.59)},
    {"Tl", AtomInfo(81, 204.383)}, {"Pb", AtomInfo(82, 207.2)},
    {"Bi", AtomInfo(83, 208.98)},  {"Po", AtomInfo(84, 210)},
    {"At", AtomInfo(85, 210)},     {"Rn", AtomInfo(86, 222)},
    {"Fr", AtomInfo(87, 223)},     {"Ra", AtomInfo(88, 226)},
    {"Ac", AtomInfo(89, 227)},     {"Th", AtomInfo(90, 232.038)},
    {"Pa", AtomInfo(91, 231.036)}, {"U", AtomInfo(92, 238.029)},
    {"Np", AtomInfo(93, 237)},     {"Pu", AtomInfo(94, 244)},
    {"Am", AtomInfo(95, 243)},     {"Cm", AtomInfo(96, 247)},
    {"Bk", AtomInfo(97, 247)},     {"Cf", AtomInfo(98, 251)},
    {"Es", AtomInfo(99, 252)},     {"Fm", AtomInfo(100, 257)},
    {"Md", AtomInfo(101, 258)},    {"No", AtomInfo(102, 259)},
    {"Lr", AtomInfo(103, 262)},    {"Rf", AtomInfo(104, 261)},
    {"Db", AtomInfo(105, 262)},    {"Sg", AtomInfo(106, 266)},
    {"Bh", AtomInfo(107, 264)},    {"Hs", AtomInfo(108, 267)},
    {"Mt", AtomInfo(109, 268)},    {"Ds", AtomInfo(110, 271)},
    {"Rg", AtomInfo(111, 272)},    {"Cn", AtomInfo(112, 285)},
    {"Nh", AtomInfo(113, 284)},    {"Fl", AtomInfo(114, 289)},
    {"Mc", AtomInfo(115, 288)},    {"Lv", AtomInfo(116, 292)},
    {"Ts", AtomInfo(117, 295)},    {"Og", AtomInfo(118, 294)}};
} // namespace libskit
#endif /* ATOM_INFO_H */
