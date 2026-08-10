# Optional reference-validation script against CRAN QCA.
# This is not used by the Python runtime.
#
# Usage:
#   Rscript validation/r/parity.R validation/parity_input.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Expected one CSV path")
if (!requireNamespace("QCA", quietly = TRUE)) stop("Install CRAN package QCA")

dat <- read.csv(args[[1]])
cat("R QCA version:", as.character(utils::packageVersion("QCA")), "\n")

# Canonical reference checks should be extended as features reach parity.
print(QCA::truthTable(dat, outcome = "Y", conditions = c("A", "B"), incl.cut = 0.8))
