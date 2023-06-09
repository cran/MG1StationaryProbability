% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/script.R
\name{stationaryProbabilities}
\alias{stationaryProbabilities}
\title{Stationary probability function}
\usage{
stationaryProbabilities(m = c(0.2, 0.3), lambda = c(1, 2), tmax = 12, nmax = 5)
}
\arguments{
\item{m}{distribution parameters vector of sojourn times
in alternating environment states}

\item{lambda}{Poisson flow intensity vector}

\item{tmax}{upper integration limit}

\item{nmax}{limitation for number of arriving claims}
}
\value{
MC stationary probability vector
}
\description{
Stationary probability function
}
