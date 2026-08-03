
Details
=========

C++ does not allow virtual templates.
Therefore I have implemented a default conversion function that shall never be called directly.
`template <typename U> Instrument<U> convert() const {}`
