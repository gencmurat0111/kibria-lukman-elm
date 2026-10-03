dunn.test.manual <- function(x, groups, algorithms, alpha = 0.05) {
  
  # Grupları factor yap ve seviyelerini al
  groups <- as.factor(groups)
  group_levels <- levels(groups)
  group_names <- as.character(groups)
  
  # 1. Kruskal-Wallis test fonksiyonu
  kruskal.wallis.test <- function(x, g) {
    N <- length(x)
    Data <- data.frame(value = x, group = g)
    Data$rank <- rank(Data$value, ties.method = "average", na.last = NA)
    
    k <- length(unique(Data$group))
    
    # Bağlı değerler için düzeltme
    ranks <- Data$rank
    ties <- unique(ranks[duplicated(ranks)])
    tiesadj <- 0
    if (length(ties) > 0) {
      for (t in ties) {
        tau <- sum(ranks == t)
        tiesadj <- tiesadj + (tau^3 - tau)
      }
    }
    tiesadj <- ifelse(tiesadj > 0, 1 - (tiesadj/((N^3) - N)), 1)
    
    # H istatistiği
    ranksum <- 0
    for (i in unique(Data$group)) {
      ni <- sum(Data$group == i)
      ranksum <- ranksum + (sum(Data$rank[Data$group == i])^2) / ni
    }
    
    H <- ((12/(N * (N + 1))) * ranksum - 3 * (N + 1)) / tiesadj
    df <- k - 1
    p <- pchisq(H, df, lower.tail = FALSE)
    
    return(list(H = H, df = df, p = p, N = N, group_levels = levels(g)))
  }
  
  # 2. Dunn testi
  dunn.test <- function(x, g, algorithms) {
    # Tüm unique grupları al
    g <- as.factor(g)
    all_groups <- levels(g)
    k <- length(all_groups)
    
    # Veriyi hazırla
    Data <- data.frame(value = x, group = g)
    Data$rank <- rank(Data$value, ties.method = "average", na.last = NA)
    N <- nrow(Data)
    
    # Bağlı değerler için düzeltme
    ranks <- Data$rank
    ties <- unique(ranks[duplicated(ranks)])
    tiesadj <- 0
    if (length(ties) > 0) {
      for (t in ties) {
        tau <- sum(ranks == t)
        tiesadj <- tiesadj + (tau^3 - tau)
      }
    }
    tiesadj <- tiesadj / (12 * (N - 1))
    
    # Tüm ikili karşılaştırmalar için z ve p değerlerini hesapla
    z_matrix <- matrix(NA, nrow = k, ncol = k)
    p_matrix <- matrix(NA, nrow = k, ncol = k)
    rownames(z_matrix) <- all_groups
    colnames(z_matrix) <- all_groups
    rownames(p_matrix) <- all_groups
    colnames(p_matrix) <- all_groups
    
    for (i in 1:(k-1)) {
      for (j in (i+1):k) {
        group_i <- all_groups[i]
        group_j <- all_groups[j]
        
        ni <- sum(Data$group == group_i)
        nj <- sum(Data$group == group_j)
        
        mean_rank_i <- mean(Data$rank[Data$group == group_i])
        mean_rank_j <- mean(Data$rank[Data$group == group_j])
        
        # Z istatistiği
        z <- (mean_rank_i - mean_rank_j) / sqrt(((N * (N + 1)/12) - tiesadj) * (1/ni + 1/nj))
        
        # P değeri (two-tailed)
        p_value <- pnorm(-abs(z))
        
        z_matrix[i, j] <- z
        z_matrix[j, i] <- -z
        p_matrix[i, j] <- p_value
        p_matrix[j, i] <- p_value
      }
    }
    
    # Köşegenleri NA yap
    diag(z_matrix) <- NA
    diag(p_matrix) <- NA
    
    # Tüm algoritmaların listesini al (sütunlar için)
    all_algorithms <- all_groups  # Tüm grupları kullan
    
    # Tabloyu oluştur - sadece argüman olarak verilen algoritmalar için satır
    n_rows <- length(algorithms) * 2
    results_table <- data.frame(
      Algorithm = rep(algorithms, each = 2),
      Statistic = rep(c("z", "p"), length(algorithms)),
      stringsAsFactors = FALSE
    )
    
    # Başlangıçta tüm sütunları "--" ile doldur
    for (algo in all_algorithms) {
      results_table[[algo]] <- "--"
    }
    
    # Her bir algoritma için satırları doldur
    for (i in 1:length(algorithms)) {
      row_algo <- algorithms[i]
      
      if (row_algo %in% all_groups) {
        row_idx <- which(all_groups == row_algo)
        
        # z satırı (tek satır)
        z_row <- (i-1)*2 + 1
        # p satırı (çift satır)
        p_row <- (i-1)*2 + 2
        
        # Tüm sütunlar için değerleri doldur
        for (j in 1:length(all_algorithms)) {
          col_algo <- all_algorithms[j]
          
          if (col_algo == row_algo) {
            # Aynı algoritma - "--"
            results_table[z_row, col_algo] <- "--"
            results_table[p_row, col_algo] <- "--"
          } else {
            col_idx <- which(all_groups == col_algo)
            
            # z değeri
            z_val <- z_matrix[row_idx, col_idx]
            if (!is.na(z_val)) {
              results_table[z_row, col_algo] <- sprintf("%.4f", z_val)
            }
            
            # p değeri
            p_val <- p_matrix[row_idx, col_idx]
            if (!is.na(p_val)) {
              # PDF'deki gibi formatla: 0.0000 veya <0.0001
              if (p_val < 0.0001) {
                results_table[p_row, col_algo] <- "<0.0001"
              } else {
                results_table[p_row, col_algo] <- sprintf("%.4f", p_val)
              }
            }
          }
        }
      }
    }
    
    return(results_table)
  }
  
  # Ana işlemler
  kw_result <- kruskal.wallis.test(x, groups)
  dunn_result <- dunn.test(x, groups, algorithms)
  
  # Sonuç listesi
  results <- list(
    dunn_test = dunn_result,
    kruskal_wallis = kw_result
  )
  
  return(results)
}


# Ornek:
#results <- dunn.test.manual(df_secilen_50$rmse, df_secilen_50$group, 
#                            c("mkl_elm_50_hkm", "kl_elm_50_kmin", 
#                              "ridge_elm_50_khm"))



