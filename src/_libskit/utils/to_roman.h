/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef TO_ROMAN_H
#define TO_ROMAN_H

#include <string>

namespace libskit {
/**
 * @brief Convert integer to roman numeral.
 * Blatantly stolen from
 * https://rosettacode.org/wiki/Roman_numerals/Encode#C.2B.2B.
 * @param value integer
 * @return std::string roman numeral
 */
std::string to_roman(unsigned int value);
} // namespace libskit

#endif /* TO_ROMAN_H */
