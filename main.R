# Gerekli kütüphaneleri yükle
library(MASS)
library(dplyr)

rm(list=ls())

source("funs-gh.R")

# Veri setini yükle

# # Source: StatLib Datasets Archive (http://lib.stat.cmu.edu/datasets/bodyfat)


bodyfat <- read.csv("bodyfat.csv", sep=";")
X <- model.matrix(Y_Density~., bodyfat)[,-1]
y <- bodyfat$Y_Density; y <- as.matrix(y)

# Veriyi normalleştir
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
X <- apply(X, 2, normalize)
y <- normalize(y)

# Deney parametreleri
n_hidden_values <- c(50, 100)
n_trials <- 100
methods <- list(
  classic_elm = list(func = classic_elm, lambdas = c("none")),
  ridge_elm = list(func = ridge_elm, lambdas = c("khm")),
  liu_elm = list(func = liu_elm, lambdas = c("dopt")),
  aur_elm = list(func = aur_elm, lambdas = c("khm")),
  kl_elm = list(func = kl_elm, lambdas = c("hkm", "kmin")),
  mkl_elm = list(func = mkl_elm, lambdas = c("hkm", "kmin"))
)

# Sonuçları saklamak için veri çerçevesi
results <- data.frame(
  Method = character(),
  n_hidden = integer(),
  Lambda_Method = character(),
  Lambda_Value_Mean = numeric(),
  Train_RMSE_Mean = numeric(),
  Train_RMSE_SD = numeric(),
  Test_RMSE_Mean = numeric(),
  Test_RMSE_SD = numeric(),
  Test_RMSE_Reduce_Rate = numeric(),
  Comp_Time_Mean = numeric(),
  Comp_Time_SD = numeric(),
  stringsAsFactors = FALSE
)

# Bireysel test RMSE değerlerini saklamak için liste
all_trial_rmse <- list()

# Deneyleri çalıştır
for (method_name in names(methods)) {
  method <- methods[[method_name]]
  func <- method$func
  lambdas <- method$lambdas
  
  for (n_hidden in n_hidden_values) {
    for (lambda in lambdas) {
      trial_results <- list(
        lambda_values = numeric(n_trials),
        train_rmse = numeric(n_trials),
        test_rmse = numeric(n_trials),
        comp_time = numeric(n_trials)
      )
      
      for (trial in 1:n_trials) {
        # Rastgele eğitim/test bölünmesi
        set.seed(123 + trial)
        train_idx <- sample(1:nrow(X), floor(nrow(X)*0.7))
        test_idx <- setdiff(1:nrow(X), train_idx)
        
        X_train <- X[train_idx, ]
        y_train <- y[train_idx, , drop = FALSE]
        X_test <- X[test_idx, ]
        y_test <- y[test_idx, , drop = FALSE]
        
        # Hesaplama süresini ölç
        start_time <- system.time({
          if (method_name == "classic_elm" && lambda == "none") {
            result <- func(
              X_train = X_train,
              y_train = y_train,
              X_test = X_test,
              y_test = y_test,
              n_hidden = n_hidden,
              activation = "tanh",
              weight_init = "uniform",
              bias = FALSE,
              seed = 123 + trial,
              verbose = FALSE
            )
            lambda_val <- NA
          } else {
            result <- func(
              X_train = X_train,
              y_train = y_train,
              X_test = X_test,
              y_test = y_test,
              n_hidden = n_hidden,
              lambda = lambda,
              activation = "tanh",
              weight_init = "uniform",
              bias = FALSE,
              seed = 123 + trial,
              verbose = FALSE
            )
            lambda_val <- result$model_info$lambda
          }
        })[3]
        
        # Sonuçları kaydet
        trial_results$lambda_values[trial] <- lambda_val
        trial_results$train_rmse[trial] <- result$train_metrics$rmse
        trial_results$test_rmse[trial] <- result$test_metrics$rmse
        trial_results$comp_time[trial] <- start_time
      }
      
      # Ortalama ve standart sapmaları hesapla
      lambda_mean <- mean(trial_results$lambda_values, na.rm = TRUE)
      train_rmse_mean <- mean(trial_results$train_rmse)
      train_rmse_sd <- sd(trial_results$train_rmse)
      test_rmse_mean <- mean(trial_results$test_rmse)
      test_rmse_sd <- sd(trial_results$test_rmse)
      comp_time_mean <- mean(trial_results$comp_time)
      comp_time_sd <- sd(trial_results$comp_time)
      
      # Reduce Rate hesaplama
      if (method_name == "classic_elm") {
        test_rmse_reduce_rate <- NA
      } else {
        elm_result <- results[results$Method == "classic_elm" & results$n_hidden == n_hidden, ]
        if (nrow(elm_result) > 0) {
          elm_test_rmse <- elm_result$Test_RMSE_Mean[1]
          test_rmse_reduce_rate <- ((elm_test_rmse - test_rmse_mean) / elm_test_rmse) * 100
        } else {
          test_rmse_reduce_rate <- NA
        }
      }
      
      # Sonuçları kaydet
      results <- rbind(results, data.frame(
        Method = method_name,
        n_hidden = n_hidden,
        Lambda_Method = lambda,
        Lambda_Value_Mean = lambda_mean,
        Train_RMSE_Mean = train_rmse_mean,
        Train_RMSE_SD = train_rmse_sd,
        Test_RMSE_Mean = test_rmse_mean,
        Test_RMSE_SD = test_rmse_sd,
        Test_RMSE_Reduce_Rate = test_rmse_reduce_rate,
        Comp_Time_Mean = comp_time_mean,
        Comp_Time_SD = comp_time_sd
      ))
      
      # Bireysel test RMSE'leri kaydet
      group_key <- paste(method_name, n_hidden, lambda, sep = "_")
      all_trial_rmse[[group_key]] <- trial_results$test_rmse
    }
  }
}

# Bireysel verileri dataframe'e dönüştür ve CSV'ye yaz
individual_df <- do.call(rbind, lapply(names(all_trial_rmse), function(g) {
  data.frame(group = g, rmse = all_trial_rmse[[g]])
}))
write.csv(individual_df, "individual_test_rmse.csv", row.names = FALSE)

# Orijinal sonuçları yaz
print(results, digits = 4)
results[is.na(results)] <- 0
write.csv(results, "elm_results.csv", row.names = FALSE)
xtable::xtable(results, digits = 4)

# RAPORLAMA İÇİN SEÇİLENLER
results_rapor <- results |>
  dplyr::select(c(Method, n_hidden,
                  Train_RMSE_Mean, Train_RMSE_SD,
                  Test_RMSE_Mean, Test_RMSE_SD,
                  Test_RMSE_Reduce_Rate,
                  Lambda_Method, Lambda_Value_Mean,
                  Comp_Time_Mean)) |>
  dplyr::mutate(Method = rep(c("ELM","R-ELM","L-ELM","AUR-ELM",
                               "KL-ELM","KL-ELM",
                               "MKL-ELM","MKL-ELM"), each=2)) |>
  dplyr::mutate(Dataset = c("bodyfat", rep("", 15)), .before = Method)

writexl::write_xlsx(results_rapor, "report_bodyfat.xlsx")

# SEÇİLENLER
results_secilen <- results |>
  dplyr::select(c(Method, n_hidden, Lambda_Method,
                  Train_RMSE_Mean, Train_RMSE_SD,
                  Test_RMSE_Mean, Test_RMSE_SD,
                  Test_RMSE_Reduce_Rate))

# ÇOKLU KARŞILAŞTIRMALAR
df <- read.csv("individual_test_rmse.csv")
df$group <- as.factor(df$group)

df_secilen <- df |>
  mutate(group = factor(group))
levels(df_secilen$group) <- c("classic_elm_50_none",
                              "classic_elm_100_none",
                              "ridge_elm_50_khm",
                              "ridge_elm_100_khm",
                              "liu_elm_50_dopt",
                              "liu_elm_100_dopt",
                              "aur_elm_50_khm",
                              "aur_elm_100_khm",
                              "kl_elm_50_hkm",
                              "kl_elm_100_hkm",
                              "kl_elm_50_kmin",
                              "kl_elm_100_kmin",
                              "mkl_elm_50_hkm",
                              "mkl_elm_100_hkm",
                              "mkl_elm_50_kmin",
                              "mkl_elm_100_kmin")

df_secilen_50 <- df_secilen |> filter(grepl("50", group)) |>
  mutate(group = factor(group))

df_secilen_100 <- df_secilen |> filter(grepl("100", group)) |>
  mutate(group = factor(group))

### Non-Parametrik Testler ###

# Kruskal-Wallis
kruskal_50 <- kruskal.test(rmse ~ group, data = df_secilen_50)
print(kruskal_50)

kruskal_100 <- kruskal.test(rmse ~ group, data = df_secilen_100)
print(kruskal_100)

# Dunn testi
dunn_50 <- dunn.test::dunn.test(df_secilen_50$rmse, df_secilen_50$group,
                                method = "bonferroni")
dunn_results_50 <- data.frame(
  Comparison = dunn_50$comparisons,
  Z_Value = round(dunn_50$Z, 3),
  P_Value = round(dunn_50$P, 4),
  P_Adjusted = round(dunn_50$P.adjusted, 4),
  Significance = ifelse(dunn_50$P.adjusted < 0.001, "***",
                        ifelse(dunn_50$P.adjusted < 0.01, "**",
                               ifelse(dunn_50$P.adjusted < 0.05, "*", "ns")))
)
print(dunn_results_50)

dunn_100 <- dunn.test::dunn.test(df_secilen_100$rmse, df_secilen_100$group,
                                 method = "bonferroni", kw=T, label=T, table=T)
dunn_results_100 <- data.frame(
  Comparison = dunn_100$comparisons,
  Z_Value = round(dunn_100$Z, 3),
  P_Value = round(dunn_100$P, 4),
  P_Adjusted = round(dunn_100$P.adjusted, 4),
  Significance = ifelse(dunn_100$P.adjusted < 0.001, "***",
                        ifelse(dunn_100$P.adjusted < 0.01, "**",
                               ifelse(dunn_100$P.adjusted < 0.05, "*", "ns")))
)
print(dunn_results_100)

# Raporlanacak sonuçlar
source("funs_DUNN-gh.R")
dunn_50_rapor <- dunn.test.manual(df_secilen_50$rmse, df_secilen_50$group,
                                  c("kl_elm_50_hkm", "kl_elm_50_kmin",
                                    "mkl_elm_50_hkm","mkl_elm_50_kmin"))
dunn_50_rapordf <- dunn_50_rapor$dunn_test
dunn_50_rapordf <- tibble::add_column(dunn_50_rapordf,
                                      `n_Hidden`=c("50", rep(".", 8-1)),
                                      .before = 1)
dunn_50_rapordf <- tibble::add_column(dunn_50_rapordf,
                                      `Dataset`=c("bodyfat", rep(".", 8-1)),
                                      .before = 1)
dunn_50_kw <- round(unlist(dunn_50_rapor$kruskal_wallis[1:3]), 5)
dunn_50_rapordf[9,] <- c(".", "K-Wallis", dunn_50_kw, rep(".", 7))
dunn_100_rapor <- dunn.test.manual(df_secilen_100$rmse, df_secilen_100$group,
                                   c("kl_elm_100_hkm", "kl_elm_100_kmin",
                                     "mkl_elm_100_hkm","mkl_elm_100_kmin"))
dunn_100_rapordf <- dunn_100_rapor$dunn_test
dunn_100_rapordf <- tibble::add_column(dunn_100_rapordf,
                                       `n_Hidden`=c("100", rep(".", 8-1)),
                                       .before = 1)
dunn_100_rapordf <- tibble::add_column(dunn_100_rapordf,
                                       `Dataset`=c("bodyfat", rep(".", 8-1)),
                                       .before = 1)
dunn_100_kw <- round(unlist(dunn_100_rapor$kruskal_wallis[1:3]), 5)
dunn_100_rapordf[9,] <- c(".", "K-Wallis", dunn_100_kw, rep(".", 7))

writexl::write_xlsx(dunn_50_rapordf, "dunn_bodyfat_50.xlsx")
writexl::write_xlsx(dunn_100_rapordf, "dunn_bodyfat_100.xlsx")

# Boxplot'lar
{
  library(ggplot2)
  library(dplyr)
  
  veriad = "bodyfat"
  
  df_plot <- df_secilen %>%
    mutate(
      hidden = factor(ifelse(grepl("_50_", group), "#hidden: 50", "#hidden: 100"),
                      levels = c("#hidden: 50", "#hidden: 100")),
      
      lambda_method = case_when(
        grepl("none", group) ~ "-",
        grepl("khm", group) ~ "KHM",
        grepl("dopt", group) ~ "d(opt)",
        grepl("hkm", group) ~ "HKM",
        grepl("kmin", group) ~ "k(min)",
        TRUE ~ "none"
      ),
      lambda_method = factor(lambda_method,
                             levels = c("-", "KHM", "d(opt)", "HKM", "k(min)")),
      
      method = case_when(
        grepl("classic_elm", group) ~ "ELM",
        grepl("ridge_elm", group) ~ "Ridge-ELM",
        grepl("liu_elm", group) ~ "Liu-ELM",
        grepl("aur_elm", group) ~ "AUR-ELM",
        grepl("mkl_elm", group) ~ "MKL-ELM",
        grepl("kl_elm", group) ~ "KL-ELM",
        TRUE ~ as.character(group)
      ),
      method = factor(method, levels = c("ELM", "Ridge-ELM", "Liu-ELM", "AUR-ELM", "KL-ELM", "MKL-ELM"))
    )
  
  fill_scale <- scale_fill_manual(
    values = c("-" = "#f5fbff",
               "KHM" = "#b3dcff",
               "d(opt)" = "#b3dcff",
               "HKM" = "#71beff",
               "k(min)" = "#50afff"),
    breaks = c("KHM", "d(opt)", "HKM", "k(min)")
  )
  
  ortak_tema <- theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom"
    )
  
  # hidden = 50
  df_50 <- df_plot %>% filter(hidden == "#hidden: 50")
  
  test_rmse_kutu_50 <- ggplot(df_50, aes(x = method, y = rmse, fill = lambda_method)) +
    geom_boxplot() +
    fill_scale +
    labs(x = "Method", y = "RMSE", fill = "Lambda Method") +
    ortak_tema
  
  test_rmse_kutu_50 <- test_rmse_kutu_50 +
    analiz::gg.tema(xsize = 8.5, ysize = 8.5, legend = F, legendpos = "bottom") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5))
  
  analiz::gg.save(paste0(veriad, "_test_rmse_kutu_hidden50"), test_rmse_kutu_50, 7, 5)
  
  # hidden = 100
  df_100 <- df_plot %>% filter(hidden == "#hidden: 100")
  
  test_rmse_kutu_100 <- ggplot(df_100, aes(x = method, y = rmse, fill = lambda_method)) +
    geom_boxplot() +
    fill_scale +
    labs(x = "Method", y = "RMSE", fill = "Lambda Method") +
    ortak_tema
  
  test_rmse_kutu_100 <- test_rmse_kutu_100 +
    analiz::gg.tema(xsize = 8.5, ysize = 8.5, legend = F, legendpos = "bottom") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5))
  
  analiz::gg.save(paste0(veriad, "_test_rmse_kutu_hidden100"), test_rmse_kutu_100, 7, 5)
}