library(ggplot2)
library(dplyr)
library(tidyr)
library(plotrix)

LENGTH = 12
DATA_FILENAME = "result.csv"

d = read.csv(DATA_FILENAME) %>%
  select(-X) %>%
  filter(length >= LENGTH)

typology = read.csv("typology3.csv")

do_plot = function(d) {
  d %>%
    filter(real %in% c("real", "free random")) %>%
    inner_join(typology) %>%
    group_by(lang_name, real) %>%
      summarise(m=mean(value),
                s=std.error(value),
  	        u=m+1.96*s,
  	        l=m-1.96*s) %>%
      ungroup() %>%
    ggplot(aes(x=real, y=m, color=real, fill=real)) +
      geom_bar(stat='identity') +
      geom_errorbar(aes(ymin=l, ymax=u), color='black') +
      facet_wrap(~lang_name) +
      theme_bw() +
      ylab("dependency length") +
      xlab("") +
      theme(axis.text.x=element_blank()) +
      scale_fill_discrete(name="Condition") +
      scale_color_discrete(guide=FALSE)
}
    
d %>% filter(length == LENGTH) %>% do_plot()
ggsave("langs_deplen_bars_12.pdf", width=9.16, height=6.5)

