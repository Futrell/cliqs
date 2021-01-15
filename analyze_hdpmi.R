setwd("~/code/cliqs") 
rm(list = ls())

library(tidyverse)
library(broom)
library(stringr)
library(lme4)

 MAXLEN = 20
 MAX_SENTENCES = 10000 # max sentences per lang, to keep R from choking on big data
 DATA_FILENAME = "survey2.csv"
 
 BAD_DEP = c(
   "X",
   "INTJ",
   "SYM",
   "NUM"
)
 
 BAD_HEAD = c(BAD_DEP, c(
   "PROPN"
 ))
 
 BAD_REL = c(
   "flat",
   "flat:foreign",
   "discourse",
   "goeswith",
   "fixed",
   "reparandum"
 )
 
 collapse_deptype = function(deptype) {
   str_split(deptype, ":") %>% 
     map(1) %>% 
     unlist()
 }
  
 pmi = read_csv("hdpmi_pos2.csv") %>% rename(hpos=h, dpos=d)
 langs = unique(pmi$lang)
 
 odd_langs = langs[unique(ceiling(1:length(langs)/2)*2)-1]
 
entropy = function(x) {
  freq = table(x)
  sum(1/freq*log(freq))
}
 
read_data = function(the_langs) {
  read_csv(DATA_FILENAME) %>%
    filter(lang %in% the_langs,
           length <= MAXLEN,
           sentence_no < MAX_SENTENCES) %>% # will result in <=MAX_SENTENCES per corpus
    inner_join(pmi) %>%
    filter(!(hpos %in% BAD_HEAD), !(dpos %in% BAD_DEP), !(deptype %in% BAD_REL)) %>%
    gather(lin, deplen, -lang, -sentence_no, -result_no, -length, -deptype, -dpos, -hpos, -pmi, -dword, -hword) %>%
    separate(lin, into=c("lin", "delete_me", "lin_no")) %>%
    select(-delete_me) %>%
    mutate(length2 = length^2,
           length2.c = length2 - mean(length2),
           pmi.c = pmi - mean(pmi),
           deplen.c = deplen - mean(deplen),
           sentence_no_lang = paste(lang, sentence_no))
}

process_lang_with_baseline = function(the_lang) {
  print(the_lang)
  d = read_data(the_lang) 
  
  m = lmer(deplen.c ~  
             length2.c*pmi.c*lin
           + (1 + pmi.c*lin | sentence_no_lang),
           #+ (1 + pmi.c*lin | lang),
           data=d, REML=F)
  m0 = lmer(deplen.c ~  
              length2.c + lin + pmi.c + length2.c:pmi.c 
            + (1 + pmi.c + lin | sentence_no_lang),
            data=d, REML=F)
  
  s_ = tidy(m) %>% 
    filter(group == "fixed") %>%
    select(-group) %>%
    mutate(lang=the_lang)
  
  anova(m, m0) %>%
    tidy() %>%
    filter(term == "m") %>%
    select(p.value) %>%
    mutate(lang=the_lang) %>%
    bind_rows(s_)
}

 
process_lang_without_baseline = function(the_lang) {
  print(the_lang)
 d = read_data(the_lang) %>% 
   filter(lin == "real")
          
 m = lmer(deplen.c ~  
            pmi.c
          + (1 + pmi.c | sentence_no_lang),
        data=d, REML=F)
 m0 = lmer(deplen.c ~  
          + (1 + pmi.c | sentence_no_lang),
        data=d, REML=F)
 
 s_ = tidy(m) %>% 
   filter(group == "fixed") %>%
   select(-group) %>%
   mutate(lang=the_lang)
 
 anova(m, m0) %>%
   tidy() %>%
   filter(term == "m") %>%
   select(p.value) %>%
   mutate(lang=the_lang) %>%
   bind_rows(s_)
}

s_with_baseline = langs %>% 
  map(process_lang_with_baseline) %>%
  reduce(bind_rows, init=tibble())

s_without_baseline = langs %>% 
  map(process_lang_without_baseline) %>%
  reduce(bind_rows, init=tibble())


# maybe just plot without the baseline?

d = read_data(odd_langs)


cut2 = function(xs, breaker) {
  m = breaker(xs)
  classify = function(x) {ifelse(x < m, 1, 2)}
  map(xs, classify)
}

d2 = d %>% 
  group_by(lang) %>%
    mutate(pmi_cat=cut2(pmi, mean),
           pmi_cat=ifelse(pmi_cat == 1, "low", "high")) %>%
    ungroup() %>%
  group_by(lang, sentence_no, pmi_cat, lin, lin_no) %>%
    summarise(deplen=sum(deplen),
              length=unique(length)) %>%
    ungroup() %>%
  mutate(lin=ifelse(lin == "real", "real", "random projective"),
         lin=factor(lin, levels=c("real", "random projective"))) # make rand the dotted one

# in how many langs is deplen for high pmi lower than deplen for low pmi?
h = d2 %>% 
  filter(lin == "real") %>%
  group_by(lang, pmi_cat) %>%
    summarise(m=mean(deplen)) %>%
    ungroup() %>%
  spread(pmi_cat, m) %>%
  mutate(ilocal=high<low)

# plot at length 10, 15, 20
plot_histograms = function(d, n) {
  
dm_means = d %>% 
  filter(lin == "real",
         length == n) %>%
  group_by(lang, pmi_cat) %>% 
    mutate(m=mean(deplen)) %>%
    ungroup()

MIN_FREQ = 10

dm_spread = d %>%
  filter(lin == "real", length == n) %>%
  spread(pmi_cat, deplen) %>%
    group_by(lang) %>%
      mutate(n=n()) %>%
      ungroup() %>%
    filter(n>=MIN_FREQ) %>% 
    select(-n) 

dm_tests = dm_spread %>%
    group_by(lang) %>%
      summarise(t=t.test(low, high)$statistic, 
                p=t.test(low, high)$p.value) %>%
      ungroup()

dm = inner_join(dm_means, dm_tests)

dm %>%
  ggplot(aes(x=deplen, y=..density.., fill=pmi_cat)) + 
    geom_histogram(alpha=.5) + 
    facet_wrap(~lang) +
    geom_vline(aes(xintercept=m, color=pmi_cat))

}

d %>% 
  filter(lin == "real") %>%
  group_by(lang) %>% 
    mutate(pmi_cat=ntile(pmi, 2),
           pmi_cat=ifelse(pmi_cat == 1, "low", "high")) %>%
    ungroup() %>% 
  plot_histograms(15) +
    ylim(0, 8) +
    xlab("") +
    ylab("Dependency length") +
    scale_fill_discrete(name="MI category") +
    scale_color_discrete(guide=FALSE)

ggsave("high_low_pmi_15.pdf")

d2 %>%  
  filter(lin == "real") %>%
  ggplot(aes(x=length, y=deplen, color=pmi_cat)) +
  stat_smooth(method="lm") +
  facet_wrap(~lang) +
  ylim(0, 30) 
  xlab("Sentence length") +
  ylab("Dependency length") +
  scale_fill_discrete(name = "pmi") 

# including the projective baseline
d2 %>%  
  ggplot(aes(x=length, y=deplen, color=pmi_cat, linetype=lin)) +
  stat_smooth() +
  facet_wrap(~lang) +
  ylim(0, 30) +
  xlab("Sentence length") +
  ylab("Dependency length") +
  scale_fill_discrete(name = "pmi") +
  scale_linetype_discrete(name = "")

  
deptype_mi = d %>% 
  group_by(lang, deptype) %>% 
    summarise(m_pmi=mean(pmi)) %>% 
    ungroup() %>% 
  spread(lang, m_pmi)

## New analysis

d_effect = d %>% 
  select(-deplen.c) %>%
  group_by_at(vars(-lin_no, -deplen)) %>% 
    summarise(deplen=mean(deplen)) %>% 
    ungroup() %>% 
  spread(lin, deplen) %>%
  mutate(effect1=real - rand, # which way of doing this gives values that look more normal?
         effect2=real/rand) # a third way would be mean(real/rand)

d_by_deptype = d %>% 
  select(-deplen.c) %>%
  group_by_at(vars(-lin_no, -deplen)) %>% 
    summarise(deplen=mean(deplen)) %>% 
    ungroup() %>% 
  spread(lin, deplen) %>%
  mutate(effect=real/rand) %>%
  mutate(deptype=collapse_deptype(deptype)) %>%
  group_by(lang, deptype)%>%
    mutate(real_deptype=mean(real),
            rand_deptype=mean(rand),
            effect_deptype=mean(effect),
            pmi_deptype=mean(pmi)) %>%
    ungroup() 

d_by_deptype %>%
  select(lang, real_deptype, rand_deptype, effect_deptype, pmi_deptype, deptype) %>%
  distinct() %>%
  ggplot(aes(x=-effect_deptype, y=pmi_deptype, label=deptype)) +
    geom_text() +
    stat_smooth(method="lm") +
    facet_wrap(~lang, scales="free") +
    xlab("Dependency attraction") +
    ylab("Average pmi")

corrs = function(d, x, y) {
  d %>%
    group_by(lang) %>%
      summarise(r=pearson_corr_r(!!x, !!y),
                r_p=pearson_corr_p(!!x, !!y),
                rho=spearman_corr_rho(!!x, !!y),
                rho_p=spearman_corr_p(!!x, !!y)) %>%
      ungroup()
}

residualize = function(y, x) {
  residuals(lm(y~x))
}

pearson_corr_r = function(x, y) {
  cor.test(x, y) %>% tidy() %>% pull(estimate)
}

pearson_corr_p = function(x, y) {
  cor.test(x, y) %>% tidy() %>% pull(p.value)
}

spearman_corr_rho = function(x, y) {
  cor.test(x, y, method="spearman") %>% tidy() %>% pull(estimate)
}

spearman_corr_p = function(x, y) {
  cor.test(x, y, method="spearman") %>% tidy() %>% pull(p.value)
}

# Calculate MI (not pmi) within deptypes (calculation will be inaccurate due to sentence cutoff)
d_mi = d %>%
  mutate(deptype=collapse_deptype(deptype)) %>%
  group_by(lang, deptype, dpos, hpos) %>%
    summarize(n=n()) %>%
    ungroup() %>%
  group_by(lang, deptype, dpos) %>%
    mutate(n_dpos=sum(n)) %>%
    ungroup() %>%
  group_by(lang, deptype, hpos) %>%
    mutate(n_hpos=sum(n)) %>%
    ungroup() %>%
  group_by(lang, deptype) %>%
    mutate(Z=sum(n)) %>%
    mutate(p=n/Z, 
           p_hpos=n_hpos/Z, 
           p_dpos=n_dpos/Z, 
           p_marginal=p_hpos*p_dpos, 
           mi=mean(p*log(p/p_marginal))) %>%
    ungroup() 
  
d_sum = d_mi %>%  
  inner_join(select(d, lang, deptype, hpos, dpos, sentence_no, length, deplen)) %>%
  group_by(lang, deptype) %>%
    summarise(mi=unique(mi),
              dl=mean(deplen)) %>%
    ungroup() 





              
