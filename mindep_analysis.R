library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)

DATA_FILENAME = "result.csv"

fit_by_lang = function(dm) {
    # Make sure the real sentences are the baseline
    the_levels = levels(dm$real)
    the_levels_without_real = the_levels[the_levels != "real"]
    the_levels_reordered = c("real", the_levels_without_real)
    dm$real = factor(dm$real, levels=the_levels_reordered)

    # Do two regressions
    # Dependency length should be predicted by (squared) length for the different factors;
    # what would a main effect of these factors on dependency length even mean?
    dm %>% group_by(lang) %>% mutate(start_line=as.factor(start_line)) %>% do(
        model2 = lmer(value ~ length2 * real + (1+real|start_line), data=., REML=F),
        model2_noint = lmer(value ~ length2 + real + (1+real|start_line), data=., REML=F)
        #model2_without = lmer(value ~ length2 + (1+real|start_line), data=., REML=F)
    )
}

summarise_model_coefs = function(dm) {
    dm %>% summarise( # crazy indexing garbage... otherwise it breaks
        intercept2 = summary(model2[[1]])[[10]][1],
        coef2 = summary(model2[[1]])[[10]][2]
    )
}

summarise_model_significance = function(dm) {
    dm %>% do( # some crazy indexing garbage here...
        aov = anova(.$model2[[1]], .$model2_without[[1]]),
        aov.int = anova(.$model2[[1]], .$model2_noint[[1]])
    ) %>% summarise(p.value = aov[[1]]$`Pr(>Chisq)`[[2]],
                    p.value.int = aov.int[[1]]$`Pr(>Chisq)`[[2]]
    )
}

analyze_by_lang = function(d) {
    print("Fitting models...")
    fits = fit_by_lang(dm)

    print(fits)

    print("Summarising coefficients...")
    coefs = summarise_model_coefs(fits)

    print("Getting model significance...")
    sigs = summarise_model_significance(fits)

    print(list(fits, coefs, sigs))
}

analyze_all_langs = function(dm, name) {
    model.filename = paste("m2", name, ".rda")
    print("Fitting quadratic model.")
    m2 = lmer(value ~ length2.cent * real + (1+real|start_line:lang:family) + (1+real|lang:family) + (1+real|family), data=dm, REML=F)
    save(m2, file=model.filename)
    remove(m2)
}

d = read.csv(DATA_FILENAME) 
d %>% filter(real != "Unnamed: ") %>%
  select(-X) %>%
  mutate(length2 = length^2) %>%
  fit_by_lang() %>%
  save(file="monster_models.rda")

# The quadratic model is worse than the linear one in 4/34 languages
# (by AIC/BIC using anova). So let's just use the quadratic model going
# forward. (The double-quadratic model is worse in all languages.)
