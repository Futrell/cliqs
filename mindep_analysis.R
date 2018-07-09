library(tidyverse)
library(lme4)
library(broom)
library(stringr)
library(optimx)

DATA_FILENAME = "mindep_20180309_lin_melted.csv"
BASELINE = "real"
#COMPARISONS = c("free random", "fixed random per language", "free head-consistent random", "fixed head-consistent random", "nonprojective free random", "nonprojective free head-consistent random")
COMPARISONS = c("free random", "rand_proj_lin_r_lic", "rand_proj_lin_perplex", "rand_proj_lin_meaningsame")
OPTIMIZER = "Nelder_Mead"

set.seed(1)

args <- commandArgs(TRUE)
the_lang = args[1]
na_returner = function(err) NA
null_to_na = function(x) ifelse(is.null(x), NA, x)


fit_by_lang = function(dm, baseline, comparison) {
    ## Make sure the real sentences are the baseline
    ## Do two regressions
    ## Dependency length should be predicted by (squared) length for the different factors;
    ## what would a main effect of these factors on dependency length even mean?
    dm %>%
        filter(real %in% c(baseline, comparison)) %>%
        mutate(real=factor(real, levels=c(baseline, comparison))) %>%
        do(
            lang=first(.$lang),
            model = tryCatch(lmer(value ~ length * real + (1+real|start_line), data=., REML=F), error=na_returner),
            model_noint = tryCatch(lmer(value ~ length + real + (1+real|start_line), data=., REML=F), error=na_returner)
        )
}

summarise_model = function(dm) {
    dm %>%
        summarise(
            coef = tryCatch(null_to_na(tidy(model)[4,]$estimate), error=na_returner),
            p = tryCatch(null_to_na(tidy(anova(model, model_noint))$p.value[2]), error=no_returner),
            lang = lang
    )
}


run_comparison = function(comparison) {    
    print(str_c("Running comparison to ", comparison))
    d %>%
        group_by(lang) %>%
          fit_by_lang(BASELINE, comparison) %>%
          summarise_model() %>%
          ungroup() %>%
        mutate(comparison=comparison)
}

d = read_csv(DATA_FILENAME) %>%
    #select(-X1) %>%
    filter(lang == the_lang) %>%
    filter(real != "Unnamed: ") %>%
    mutate(start_line = as.factor(start_line))

result = COMPARISONS %>%
    map(run_comparison) %>%
    reduce(bind_rows, tibble())

outfilename = str_c(the_lang, "_model_coefficients_20180308.csv")
print(outfilename)
write.csv(result, file=outfilename)
