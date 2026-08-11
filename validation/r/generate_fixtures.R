# Generate golden parity fixtures from the reference R QCA implementation.
#
# The fixtures produced here are committed to the repository so that the Python
# parity tests run everywhere without requiring R. R is needed only to
# regenerate them, which should be done deliberately and reviewed as a diff.
#
# Usage:
#   Rscript validation/r/generate_fixtures.R validation/fixtures/r_qca.json

suppressMessages(library(QCA))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[[1]] else "validation/fixtures/r_qca.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# R codes remainders "?" and contradictions "C"; setqca uses "R" and "C".
recode_out <- function(x) {
  x <- as.character(x)
  ifelse(x == "?", "R", x)
}

# A truth-table row number in R is the minterm index plus one, because both
# enumerate the property space with the last condition varying fastest.
truth_table_records <- function(tt, conditions) {
  frame <- tt$tt
  lapply(seq_len(nrow(frame)), function(i) {
    row <- frame[i, ]
    incl <- suppressWarnings(as.numeric(as.character(row[["incl"]])))
    pri <- suppressWarnings(as.numeric(as.character(row[["PRI"]])))
    list(
      minterm = i - 1L,
      configuration = as.integer(unlist(row[conditions])),
      n = as.integer(row[["n"]]),
      consistency = if (is.na(incl)) NULL else incl,
      pri = if (is.na(pri)) NULL else pri,
      out = recode_out(row[["OUT"]])
    )
  })
}

# Return each solution as a list of implicants, so the Python side can compare
# canonical literal sets rather than formatted strings.
solution_records <- function(model) {
  if (is.null(model$solution)) {
    return(list())
  }
  lapply(model$solution, function(solution) as.character(solution))
}

safe_minimize <- function(tt, include) {
  result <- tryCatch(
    minimize(tt, include = include, details = TRUE),
    error = function(e) NULL
  )
  if (is.null(result)) list() else solution_records(result)
}

# R names the fit columns by relation: inclS/PRI/covS for sufficiency and
# inclN/RoN/covN for necessity.
pof_record <- function(expression, outcome, data, relation) {
  value <- tryCatch(
    pof(expression, outcome, data, relation = relation),
    error = function(e) NULL
  )
  if (is.null(value)) {
    return(NULL)
  }
  frame <- value$incl.cov
  columns <- colnames(frame)
  # A disjunctive expression yields one row per disjunct plus an aggregate row
  # named "expression". The aggregate is the value for the whole expression.
  row <- if ("expression" %in% rownames(frame)) frame["expression", ] else frame[1, ]
  pick <- function(name) {
    if (name %in% columns) as.numeric(row[[name]]) else NULL
  }

  out <- list(
    expression = expression,
    outcome = outcome,
    relation = relation
  )
  if (relation == "sufficiency") {
    out$consistency <- pick("inclS")
    out$coverage <- pick("covS")
    out$pri <- pick("PRI")
  } else {
    out$consistency <- pick("inclN")
    out$coverage <- pick("covN")
    out$ron <- pick("RoN")
  }
  out
}

# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

build_analysis <- function(id, dataset_name, outcome, conditions, incl_cut, n_cut) {
  data(list = dataset_name, package = "QCA", envir = environment())
  frame <- get(dataset_name, envir = environment())
  frame <- frame[, c(conditions, outcome), drop = FALSE]

  tt <- truthTable(
    frame,
    outcome = outcome,
    conditions = conditions,
    incl.cut = incl_cut,
    n.cut = n_cut,
    show.cases = TRUE
  )

  list(
    id = id,
    dataset = dataset_name,
    outcome = outcome,
    conditions = conditions,
    incl_cut = incl_cut,
    n_cut = n_cut,
    data = as.list(frame),
    case_ids = rownames(frame),
    truth_table = truth_table_records(tt, conditions),
    conservative = safe_minimize(tt, include = ""),
    parsimonious = safe_minimize(tt, include = "?")
  )
}

analyses <- list(
  build_analysis("LF-SURV-incl80", "LF", "SURV",
                 c("DEV", "URB", "LIT", "IND", "STB"), 0.8, 1),
  build_analysis("LF-SURV-incl90", "LF", "SURV",
                 c("DEV", "URB", "LIT", "IND", "STB"), 0.9, 1),
  build_analysis("LC-SURV-crisp", "LC", "SURV",
                 c("DEV", "URB", "LIT", "IND", "STB"), 1.0, 1),
  build_analysis("LF-SURV-three", "LF", "SURV",
                 c("DEV", "LIT", "STB"), 0.8, 1)
)

# ---------------------------------------------------------------------------
# Intermediate solutions
# ---------------------------------------------------------------------------

# R keeps the intermediate solution in $i.sol, not $solution: $solution stays
# the parsimonious result even when dir.exp is supplied.
minterms_of <- function(frame, conditions) {
  if (is.null(frame) || nrow(frame) == 0) {
    return(integer(0))
  }
  sort(apply(frame[, conditions, drop = FALSE], 1, function(row) {
    sum(as.integer(row) * 2^rev(seq_along(row) - 1))
  }))
}

build_intermediate <- function(id, dataset_name, outcome, conditions, incl_cut, expectations) {
  data(list = dataset_name, package = "QCA", envir = environment())
  frame <- get(dataset_name, envir = environment())
  frame <- frame[, c(conditions, outcome), drop = FALSE]
  tt <- truthTable(frame, outcome = outcome, conditions = conditions, incl.cut = incl_cut)

  model <- minimize(tt, include = "?", dir.exp = expectations, details = TRUE)
  branch <- model$i.sol[[1]]

  list(
    id = id,
    dataset = dataset_name,
    outcome = outcome,
    conditions = conditions,
    incl_cut = incl_cut,
    expectations = as.list(expectations),
    data = as.list(frame),
    case_ids = rownames(frame),
    intermediate = lapply(branch$solution, as.character),
    easy = as.integer(minterms_of(branch$EC, conditions)),
    difficult = as.integer(minterms_of(branch$DC, conditions)),
    simplifying_assumptions = as.integer(minterms_of(model$SA[[1]], conditions))
  )
}

intermediate_cases <- list(
  build_intermediate(
    "LF-SURV-all-present", "LF", "SURV",
    c("DEV", "URB", "LIT", "IND", "STB"), 0.8,
    c(DEV = 1, URB = 1, LIT = 1, IND = 1, STB = 1)
  ),
  build_intermediate(
    "LF-SURV-ind-absent", "LF", "SURV",
    c("DEV", "URB", "LIT", "IND", "STB"), 0.8,
    c(DEV = 1, URB = 1, LIT = 1, IND = 0, STB = 1)
  ),
  build_intermediate(
    "LF-SURV-three", "LF", "SURV",
    c("DEV", "LIT", "STB"), 0.8,
    c(DEV = 1, LIT = 1, STB = 1)
  )
)

# ---------------------------------------------------------------------------
# Parameters of fit
# ---------------------------------------------------------------------------

data(LF, envir = environment())
pof_cases <- Filter(Negate(is.null), list(
  pof_record("DEV*URB", "SURV", LF, "sufficiency"),
  pof_record("DEV*~URB", "SURV", LF, "sufficiency"),
  pof_record("DEV", "SURV", LF, "sufficiency"),
  pof_record("LIT", "SURV", LF, "necessity"),
  pof_record("DEV", "SURV", LF, "necessity"),
  pof_record("DEV+URB", "SURV", LF, "necessity")
))

# ---------------------------------------------------------------------------
# Systematic necessity screening
# ---------------------------------------------------------------------------

# Every condition and its negation, screened one at a time, plus a couple of
# unions to exercise the SUIN case.
build_necessity_screen <- function(id, dataset_name, outcome, conditions, extra) {
  data(list = dataset_name, package = "QCA", envir = environment())
  frame <- get(dataset_name, envir = environment())
  frame <- frame[, c(conditions, outcome), drop = FALSE]

  expressions <- c(conditions, paste0("~", conditions), extra)
  records <- lapply(expressions, function(expression) {
    value <- pof(expression, outcome, frame, relation = "necessity")
    row <- if ("expression" %in% rownames(value$incl.cov)) {
      value$incl.cov["expression", ]
    } else {
      value$incl.cov[1, ]
    }
    list(
      expression = expression,
      consistency = as.numeric(row[["inclN"]]),
      coverage = as.numeric(row[["covN"]]),
      ron = as.numeric(row[["RoN"]])
    )
  })

  list(
    id = id,
    dataset = dataset_name,
    outcome = outcome,
    conditions = conditions,
    data = as.list(frame),
    case_ids = rownames(frame),
    candidates = records
  )
}

necessity_screens <- list(
  build_necessity_screen(
    "LF-SURV-screen", "LF", "SURV",
    c("DEV", "URB", "LIT", "IND", "STB"),
    c("DEV+URB", "LIT+STB", "~DEV+~URB")
  )
)

# ---------------------------------------------------------------------------
# Direct calibration
# ---------------------------------------------------------------------------

# The last two values sit far outside the anchors on purpose. R's calibrate()
# ends with `fs[fs < 1e-04] <- 0; fs[fs > 0.9999] <- 1`, so those entries come
# back exactly 0 and 1 rather than as the true logistic value. Keeping them in
# the fixture pins that documented divergence instead of hiding it.
raw_values <- c(0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, -50, 150)

# Mirrors R's snapping rule, recorded alongside the values so the Python side
# can assert the divergence precisely rather than loosening its tolerance.
R_SNAP_LOW <- 1e-04
R_SNAP_HIGH <- 0.9999

calibration_cases <- list(
  list(
    id = "increasing-logistic-idm95",
    values = raw_values,
    thresholds = c(20, 50, 80),
    idm = 0.95,
    logistic = TRUE,
    expected = as.numeric(calibrate(raw_values, type = "fuzzy",
                                    thresholds = c(20, 50, 80), idm = 0.95))
  ),
  list(
    id = "decreasing-logistic-idm95",
    values = raw_values,
    thresholds = c(80, 50, 20),
    idm = 0.95,
    logistic = TRUE,
    expected = as.numeric(calibrate(raw_values, type = "fuzzy",
                                    thresholds = c(80, 50, 20), idm = 0.95))
  ),
  list(
    id = "increasing-logistic-idm99",
    values = raw_values,
    thresholds = c(20, 50, 80),
    idm = 0.99,
    logistic = TRUE,
    expected = as.numeric(calibrate(raw_values, type = "fuzzy",
                                    thresholds = c(20, 50, 80), idm = 0.99))
  ),
  list(
    id = "increasing-linear",
    values = raw_values,
    thresholds = c(20, 50, 80),
    idm = 0.95,
    logistic = FALSE,
    expected = as.numeric(calibrate(raw_values, type = "fuzzy",
                                    thresholds = c(20, 50, 80), logistic = FALSE))
  )
)

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

fixture <- list(
  generated_with = list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    QCA = as.character(packageVersion("QCA"))
  ),
  r_snapping = list(
    low = R_SNAP_LOW,
    high = R_SNAP_HIGH,
    source = "QCA::calibrate ends with fs[fs < 1e-04] <- 0; fs[fs > 0.9999] <- 1"
  ),
  note = paste(
    "Golden values from the reference CRAN QCA implementation.",
    "Regenerate with: Rscript validation/r/generate_fixtures.R"
  ),
  calibration = calibration_cases,
  pof = pof_cases,
  analyses = analyses,
  intermediate = intermediate_cases,
  necessity_screens = necessity_screens
)

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write(toJSON(fixture, auto_unbox = TRUE, digits = 15, null = "null", pretty = 2), out_path)
cat("wrote", out_path, "\n")
cat("  calibration cases:", length(calibration_cases), "\n")
cat("  pof cases:        ", length(pof_cases), "\n")
cat("  analyses:         ", length(analyses), "\n")
cat("  intermediate:     ", length(intermediate_cases), "\n")
cat("  necessity screens:", length(necessity_screens), "\n")
