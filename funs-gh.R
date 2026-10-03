# Yardımcı metrik hesaplama fonksiyonu
calculate_metrics <- function(y_true, y_pred, n_hidden, n_samples) {
  errors <- y_true - y_pred
  squared_errors <- errors^2
  
  mse <- mean(squared_errors)
  rmse <- sqrt(mse)
  
  return(list(
    rmse = rmse
  ))
}

# Klasik ELM Fonksiyonu
classic_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                        activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                        weight_init = c("normal", "uniform", "he"),
                        bias = TRUE,
                        seed = NULL,
                        verbose = FALSE) {
  
  # 1. Paket kontrolü
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil. Lütfen install.packages('MASS') ile yükleyin.")
  
  # 2. Argüman doğrulama
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  if (n_hidden < 1)
    stop("n_hidden en az 1 olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  # 3. Ağırlık başlatma
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  n_input <- ncol(X_train) + as.integer(bias)
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  # 4. Bias vektörü
  if (bias) {
    bias_vector <- runif(n_hidden, min = -1, max = 1)
  } else {
    bias_vector <- rep(0, n_hidden)
  }
  
  # 5. Aktivasyon fonksiyonu
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  # 6. Gizli katman çıktısı (eğitim)
  if (bias) {
    X_train_bias <- cbind(1, X_train)
  } else {
    X_train_bias <- X_train
  }
  H_train <- X_train_bias %*% input_weights
  H_train <- H_train + matrix(bias_vector, nrow = nrow(H_train), ncol = n_hidden, byrow = TRUE)
  H_train <- apply_activation(H_train, activation)
  
  # 7. Çıkış ağırlıkları
  output_weights <- MASS::ginv(H_train, tol = sqrt(.Machine$double.eps)) %*% y_train
  
  # 8. Gizli katman çıktısı (test)
  if (bias) {
    X_test_bias <- cbind(1, X_test)
  } else {
    X_test_bias <- X_test
  }
  H_test <- X_test_bias %*% input_weights
  H_test <- H_test + matrix(bias_vector, nrow = nrow(H_test), ncol = n_hidden, byrow = TRUE)
  H_test <- apply_activation(H_test, activation)
  
  # 9. Tahminler
  y_pred <- H_test %*% output_weights
  
  # 10. Train ve Test metrikleri
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  # 11. Sonuçları döndür
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    bias_vector = bias_vector,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = NA,
      lambda_method = "none",
      activation = activation,
      weight_init = weight_init,
      bias = bias
    )
  ))
}

# Ridge ELM Fonksiyonu
ridge_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                      lambda = 0.1,
                      activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                      weight_init = c("normal", "uniform", "he"),
                      bias = TRUE,
                      seed = NULL,
                      verbose = FALSE) {
  
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil.")
  
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (is.character(lambda) && !lambda %in% c("khm"))
    stop("lambda 'khm' veya sayısal bir değer olmalı")
  if (is.numeric(lambda) && lambda <= 0)
    stop("Sayısal lambda pozitif bir değer olmalı")
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  if (bias) {
    X_train_bias <- cbind(1, X_train)
    X_test_bias <- cbind(1, X_test)
  } else {
    X_train_bias <- X_train
    X_test_bias <- X_test
  }
  
  n_input <- ncol(X_train_bias)
  n_train <- nrow(X_train_bias)
  
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  H_train <- X_train_bias %*% input_weights
  H_train <- apply_activation(H_train, activation)
  
  HtH <- crossprod(H_train)
  beta_LS <- tryCatch(
    {
      MASS::ginv(HtH, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
    },
    error = function(e) {
      stop("H^T H matrisi için ginv başarısız: ", e$message)
    }
  )
  
  y_pred_LS <- H_train %*% beta_LS
  residuals <- y_train - y_pred_LS
  sigma2_hat <- sum(residuals^2) / (n_train - n_hidden)
  
  if (is.character(lambda)) {
    if (lambda == "khm") {
      eigen_decomp <- eigen(HtH, symmetric = TRUE)
      Q <- eigen_decomp$vectors
      alpha_hat <- t(Q) %*% beta_LS
      lambda_val <- n_hidden * sigma2_hat / sum(alpha_hat^2)
      lambda_method <- "khm"
    }
  } else {
    lambda_val <- lambda
    lambda_method <- "numeric"
  }
  
  I <- diag(n_hidden)
  output_weights <- MASS::ginv(HtH + lambda_val * I, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
  
  H_test <- X_test_bias %*% input_weights
  H_test <- apply_activation(H_test, activation)
  
  y_pred <- H_test %*% output_weights
  
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    beta_LS = beta_LS,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = lambda_val,
      lambda_method = lambda_method,
      activation = activation,
      weight_init = weight_init,
      bias = bias,
      sigma2_hat = sigma2_hat
    )
  ))
}

# Liu-ELM Fonksiyonu
liu_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                    lambda = 0.1,
                    activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                    weight_init = c("normal", "uniform", "he"),
                    bias = TRUE,
                    seed = NULL,
                    verbose = FALSE) {
  
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil.")
  
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (is.character(lambda) && !lambda %in% c("dopt"))
    stop("lambda 'dopt' veya sayısal bir değer olmalı")
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  if (bias) {
    X_train_bias <- cbind(1, X_train)
    X_test_bias <- cbind(1, X_test)
  } else {
    X_train_bias <- X_train
    X_test_bias <- X_test
  }
  
  n_input <- ncol(X_train_bias)
  n_train <- nrow(X_train_bias)
  
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  H_train <- X_train_bias %*% input_weights
  H_train <- apply_activation(H_train, activation)
  
  HtH <- crossprod(H_train)
  beta_LS <- tryCatch(
    {
      MASS::ginv(HtH, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
    },
    error = function(e) {
      stop("H^T H matrisi için ginv başarısız: ", e$message)
    }
  )
  
  y_pred_LS <- H_train %*% beta_LS
  residuals <- y_train - y_pred_LS
  sigma2_hat <- sum(residuals^2) / (n_train - n_hidden)
  
  if (is.character(lambda)) {
    if (lambda == "dopt") {
      eigen_decomp <- eigen(HtH, symmetric = TRUE)
      eigen_values <- eigen_decomp$values
      Q <- eigen_decomp$vectors
      if (any(eigen_values <= 0)) {
        eigen_values[eigen_values <= 0] <- 1e-6
      }
      
      alpha_hat <- t(Q) %*% beta_LS
      
      term1 <- sum(1 / (eigen_values * (eigen_values + 1)))
      term2 <- sum(alpha_hat^2 / (eigen_values + 1)^2)
      lambda_val <- 1 - sigma2_hat * (term1 / term2)
      lambda_val <- max(-1, min(1, lambda_val))
      lambda_method <- "dopt"
    }
  } else {
    lambda_val <- lambda
    lambda_method <- "numeric"
  }
  
  I <- diag(n_hidden)
  output_weights <- MASS::ginv(HtH + I, tol = sqrt(.Machine$double.eps)) %*%
    (crossprod(H_train, y_train) + lambda_val * beta_LS)
  
  H_test <- X_test_bias %*% input_weights
  H_test <- apply_activation(H_test, activation)
  
  y_pred <- H_test %*% output_weights
  
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    beta_LS = beta_LS,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = lambda_val,
      lambda_method = lambda_method,
      activation = activation,
      weight_init = weight_init,
      bias = bias,
      sigma2_hat = sigma2_hat
    )
  ))
}

# AUR-ELM Fonksiyonu
aur_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                    lambda = 0.1,
                    activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                    weight_init = c("normal", "uniform", "he"),
                    bias = TRUE,
                    seed = NULL,
                    verbose = FALSE) {
  
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil.")
  
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (is.character(lambda) && !lambda %in% c("khm"))
    stop("lambda 'khm' veya sayısal bir değer olmalı")
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  if (bias) {
    X_train_bias <- cbind(1, X_train)
    X_test_bias <- cbind(1, X_test)
  } else {
    X_train_bias <- X_train
    X_test_bias <- X_test
  }
  
  n_input <- ncol(X_train_bias)
  n_train <- nrow(X_train_bias)
  
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  H_train <- X_train_bias %*% input_weights
  H_train <- apply_activation(H_train, activation)
  
  HtH <- crossprod(H_train)
  beta_LS <- tryCatch(
    {
      MASS::ginv(HtH, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
    },
    error = function(e) {
      stop("H^T H matrisi için ginv başarısız: ", e$message)
    }
  )
  
  y_pred_LS <- H_train %*% beta_LS
  residuals <- y_train - y_pred_LS
  sigma2_hat <- sum(residuals^2) / (n_train - n_hidden)
  
  if (is.character(lambda)) {
    if (lambda == "khm") {
      eigen_decomp <- eigen(HtH, symmetric = TRUE)
      Q <- eigen_decomp$vectors
      alpha_hat <- t(Q) %*% beta_LS
      lambda_val <- n_hidden * sigma2_hat / sum(alpha_hat^2)
      lambda_method <- "khm"
    }
  } else {
    lambda_val <- lambda
    lambda_method <- "numeric"
  }
  
  I <- diag(n_hidden)
  inv_HtH_kI <- MASS::ginv(HtH + lambda_val * I, tol = sqrt(.Machine$double.eps))
  correction_term <- I - lambda_val^2 * inv_HtH_kI %*% inv_HtH_kI
  output_weights <- correction_term %*% beta_LS
  
  H_test <- X_test_bias %*% input_weights
  H_test <- apply_activation(H_test, activation)
  
  y_pred <- H_test %*% output_weights
  
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    beta_LS = beta_LS,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = lambda_val,
      lambda_method = lambda_method,
      activation = activation,
      weight_init = weight_init,
      bias = bias,
      sigma2_hat = sigma2_hat
    )
  ))
}

# KL-ELM Fonksiyonu
kl_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                   lambda = 0.1,
                   activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                   weight_init = c("normal", "uniform", "he"),
                   bias = TRUE,
                   seed = NULL,
                   verbose = FALSE) {
  
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil.")
  
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (is.character(lambda) && !lambda %in% c("hkm", "kmin"))
    stop("lambda 'hkm', 'kmin' veya sayısal bir değer olmalı")
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  if (bias) {
    X_train_bias <- cbind(1, X_train)
    X_test_bias <- cbind(1, X_test)
  } else {
    X_train_bias <- X_train
    X_test_bias <- X_test
  }
  
  n_input <- ncol(X_train_bias)
  n_train <- nrow(X_train_bias)
  
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  H_train <- X_train_bias %*% input_weights
  H_train <- apply_activation(H_train, activation)
  
  HtH <- crossprod(H_train)
  beta_LS <- tryCatch(
    {
      MASS::ginv(HtH, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
    },
    error = function(e) {
      stop("H^T H matrisi için ginv başarısız: ", e$message)
    }
  )
  
  y_pred_LS <- H_train %*% beta_LS
  residuals <- y_train - y_pred_LS
  sigma2_hat <- sum(residuals^2) / (n_train - n_hidden)
  
  if (is.character(lambda)) {
    if (lambda %in% c("hkm", "kmin")) {
      eigen_decomp <- eigen(HtH, symmetric = TRUE)
      eigen_values <- eigen_decomp$values
      Q <- eigen_decomp$vectors
      if (any(eigen_values <= 0)) {
        eigen_values[eigen_values <= 0] <- 1e-6
      }
      
      alpha_hat <- t(Q) %*% beta_LS
      
      if (lambda == "hkm") {
        terms <- sum(2 * alpha_hat^2 + sigma2_hat / eigen_values)
        lambda_val <- n_hidden * sigma2_hat / terms
      } else if (lambda == "kmin") {
        terms <- sigma2_hat / (2 * alpha_hat^2 + sigma2_hat / eigen_values)
        lambda_val <- min(terms)
      }
      lambda_method <- lambda
    }
  } else {
    lambda_val <- lambda
    lambda_method <- "numeric"
  }
  
  I <- diag(n_hidden)
  output_weights <- MASS::ginv(HtH + lambda_val * I, tol = sqrt(.Machine$double.eps)) %*%
    (crossprod(H_train, y_train) - lambda_val * beta_LS)
  
  H_test <- X_test_bias %*% input_weights
  H_test <- apply_activation(H_test, activation)
  
  y_pred <- H_test %*% output_weights
  
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    beta_LS = beta_LS,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = lambda_val,
      lambda_method = lambda_method,
      activation = activation,
      weight_init = weight_init,
      bias = bias,
      sigma2_hat = sigma2_hat
    )
  ))
}

# MKL-ELM Fonksiyonu
mkl_elm <- function(X_train, y_train, X_test, y_test = NULL, n_hidden,
                    lambda = 0.1,
                    activation = c("sigmoid", "relu", "leaky_relu", "tanh", "linear"),
                    weight_init = c("normal", "uniform", "he"),
                    bias = TRUE,
                    seed = NULL,
                    verbose = FALSE) {
  
  if (!requireNamespace("MASS", quietly = TRUE))
    stop("MASS paketi yüklü değil.")
  
  activation <- match.arg(activation)
  weight_init <- match.arg(weight_init)
  
  if (is.character(lambda) && !lambda %in% c("hkm", "kmin"))
    stop("lambda 'hkm', 'kmin' veya sayısal bir değer olmalı")
  
  if (!is.matrix(X_train) && !is.data.frame(X_train))
    stop("X_train bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(X_test) && !is.data.frame(X_test))
    stop("X_test bir matris veya veri çerçevesi olmalı")
  if (!is.matrix(y_train)) y_train <- as.matrix(y_train)
  
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  
  if (nrow(X_train) != nrow(y_train))
    stop("X_train ve y_train satır sayıları uyumsuz")
  if (ncol(X_train) != ncol(X_test))
    stop("X_train ve X_test sütun sayıları uyumsuz")
  if (!is.logical(bias))
    stop("bias TRUE veya FALSE olmalı")
  
  if (!is.null(seed)) set.seed(seed)
  
  if (bias) {
    X_train_bias <- cbind(1, X_train)
    X_test_bias <- cbind(1, X_test)
  } else {
    X_train_bias <- X_train
    X_test_bias <- X_test
  }
  
  n_input <- ncol(X_train_bias)
  n_train <- nrow(X_train_bias)
  
  initialize_weights <- function(n_input, n_hidden, method) {
    if (method == "normal") {
      return(matrix(rnorm(n_input * n_hidden, sd = 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "uniform") {
      return(matrix(runif(n_input * n_hidden, -1, 1),
                    nrow = n_input, ncol = n_hidden))
    } else if (method == "he") {
      return(matrix(rnorm(n_input * n_hidden, sd = sqrt(2 / n_input)),
                    nrow = n_input, ncol = n_hidden))
    }
  }
  
  input_weights <- initialize_weights(n_input, n_hidden, weight_init)
  
  apply_activation <- function(H, activation, leaky_alpha = 0.01) {
    switch(activation,
           sigmoid = plogis(H),
           relu = pmax(0, H),
           leaky_relu = ifelse(H > 0, H, leaky_alpha * H),
           tanh = tanh(H),
           linear = H,
           stop("Bilinmeyen aktivasyon fonksiyonu: ", activation))
  }
  
  H_train <- X_train_bias %*% input_weights
  H_train <- apply_activation(H_train, activation)
  
  HtH <- crossprod(H_train)
  beta_LS <- tryCatch(
    {
      MASS::ginv(HtH, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
    },
    error = function(e) {
      stop("H^T H matrisi için ginv başarısız: ", e$message)
    }
  )
  
  y_pred_LS <- H_train %*% beta_LS
  residuals <- y_train - y_pred_LS
  sigma2_hat <- sum(residuals^2) / (n_train - n_hidden)
  
  if (is.character(lambda)) {
    if (lambda %in% c("hkm", "kmin")) {
      eigen_decomp <- eigen(HtH, symmetric = TRUE)
      eigen_values <- eigen_decomp$values
      Q <- eigen_decomp$vectors
      if (any(eigen_values <= 0)) {
        eigen_values[eigen_values <= 0] <- 1e-6
      }
      
      I <- diag(n_hidden)
      default_lambda <- 0.1
      beta_RE <- MASS::ginv(HtH + default_lambda * I, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
      alpha_hat <- t(Q) %*% beta_RE
      
      if (lambda == "hkm") {
        terms <- sum(2 * alpha_hat^2 + sigma2_hat / eigen_values)
        lambda_val <- n_hidden * sigma2_hat / terms
      } else if (lambda == "kmin") {
        terms <- sigma2_hat / (2 * alpha_hat^2 + sigma2_hat / eigen_values)
        lambda_val <- min(terms)
      }
      lambda_method <- lambda
    }
  } else {
    lambda_val <- lambda
    lambda_method <- "numeric"
  }
  
  I <- diag(n_hidden)
  beta_RE <- MASS::ginv(HtH + lambda_val * I, tol = sqrt(.Machine$double.eps)) %*% crossprod(H_train, y_train)
  output_weights <- MASS::ginv(HtH + lambda_val * I, tol = sqrt(.Machine$double.eps)) %*%
    (HtH - lambda_val * I) %*% beta_RE
  
  H_test <- X_test_bias %*% input_weights
  H_test <- apply_activation(H_test, activation)
  
  y_pred <- H_test %*% output_weights
  
  y_pred_train <- H_train %*% output_weights
  train_metrics <- calculate_metrics(y_train, y_pred_train, n_hidden, nrow(X_train))
  
  test_metrics <- NULL
  if (!is.null(y_test)) {
    if (!is.matrix(y_test)) y_test <- as.matrix(y_test)
    if (nrow(y_test) != nrow(X_test)) stop("y_test ve X_test satır sayıları uyumsuz")
    test_metrics <- calculate_metrics(y_test, y_pred, n_hidden, nrow(X_test))
  }
  
  return(list(
    predictions = y_pred,
    output_weights = output_weights,
    input_weights = input_weights,
    beta_RE = beta_RE,
    H_train = H_train,
    H_test = H_test,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    model_info = list(
      n_hidden = n_hidden,
      lambda = lambda_val,
      lambda_method = lambda_method,
      activation = activation,
      weight_init = weight_init,
      bias = bias,
      sigma2_hat = sigma2_hat
    )
  ))
}