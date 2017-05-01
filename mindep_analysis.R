library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(readr)
library(broom)
library(stringr)

DATA_FILENAME = "melted_20170426_fh.csv"
BASELINE = "real"
COMPARISONS = c("free random")#, "fixed random") #, "free head-fixed random", "free head-consistent random")

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
            model2 = lmer(value ~ length2 * real + (1+real|start_line), data=., REML=F),
            model2_noint = lmer(value ~ length2 + real + (1+real|start_line), data=., REML=F)
        )
}

summarise_model = function(dm) {
    dm %>% summarise(
        coef2 = tidy(model2)[4,]$estimate,
        p = tidy(anova(model2, model2_noint))$p.value[2],
        lang = lang
    )
}

d = read_csv(DATA_FILENAME) %>%
    filter(real != "Unnamed: ") %>%
    select(-X1) %>%
    mutate(length2 = length^2,
           start_line = as.factor(start_line))

result = data.frame()

for (comparison in COMPARISONS) {
    print(str_c("Running comparison to ", comparison))
    subresult = d %>%
        group_by(lang) %>%
          fit_by_lang(BASELINE, comparison) %>%
          summarise_model() %>%
          ungroup() %>%
        mutate(comparison=comparison)
    result = rbind(result, subresult)
}

write.csv(result, file="model_coefficients.csv")

# The quadratic model is worse than the linear one in 4/34 languages
# (by AIC/BIC using anova). So let's just use the quadratic model going
# forward. (The double-quadratic model is worse in all languages.)
