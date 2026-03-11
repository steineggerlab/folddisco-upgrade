g1 <- scan("folddisco-upgrade/folddisco-validation-result/result/group_lowe_querylen.txt")
g2 <- scan("folddisco-upgrade/folddisco-validation-result/result/group_highe_querylen.txt")
t.test(g1, g2, var.equal = FALSE, alternative = "greater")
wilcox.test(g1, g2)   # Mann–Whitney