library(shapFlex)
library(dplyr)
library(ggplot2)
library(randomForest)
library(ggraph)


# Input data: Adult aka Census Income dataset.
data("data_adult", package = "shapFlex")
data <- data_adult
#------------------------------------------------------------------------------
# Train a machine learning model; currently limited to single outcome regression and binary classification. # nolint: line_length_linter.
outcome_name <- "income"
outcome_col <- which(names(data) == outcome_name)

model_formula <- formula(paste0(outcome_name,  "~ ."))

set.seed(1)
model <- randomForest::randomForest(model_formula, data = data, ntree = 300)
#------------------------------------------------------------------------------
# A user-defined prediction function that takes 2 positional arguments and returns # nolint: line_length_linter.
# a 1-column data.frame of predictions for each instance to be explained: (1) A trained
# ML model object and (2) a data.frame of model features; transformations of the input
# data such as converting the data.frame to a matrix should occur within this wrapper. # nolint: line_length_linter.
predict_function <- function(model, data) {
  
  # We'll predict the probability of the outcome being >50k.
  data_pred <- data.frame("y_pred" = predict(model, data, type = "prob")[, 2])
  return(data_pred)
}
#------------------------------------------------------------------------------
# shapFlex setup.
explain <- data[1:300, -outcome_col]  # Compute Shapley feature-level predictions for 300 instaces.

reference <- data[, -outcome_col]  # An optional reference population to compute the baseline prediction.

sample_size <- 60  # Number of Monte Carlo samples.

target_features <- c("marital_status", "education", "relationship",  "native_country",
                     "age", "sex", "race", "hours_per_week")  # Optional: A subset of features.

causal <- data.frame(
  "cause" = c(rep("age", 2), rep("marital_status",2), rep("education",2), "native_country"),
  "effect" = c("marital_status", "education", "relationship", "sex", "hours_per_week","native_country", "race")
)

set.seed(1)
causal_graph <- ggraph(causal, layout = "kk")
causal_graph <- causal_graph + geom_edge_link(aes(start_cap = label_rect(node1.name),
                            end_cap = label_rect(node2.name)),
                        arrow = arrow(length = unit(5, 'mm'), type = "closed"),
                        color = "grey25")
causal_graph <- causal_graph + geom_node_label(aes(label = name), fontface = "bold")
causal_graph <- causal_graph + scale_x_continuous(expand = expand_scale(0.2))
causal_graph <- causal_graph + theme_graph(foreground = 'white', fg_text_colour = 'white')
causal_graph

# 1: Non-causal symmetric Shapley values: ~10 seconds to run.
set.seed(1)
explained_non_causal <- shapFlex::shapFlex(explain = explain,
                                           reference = reference,
                                           model = model,
                                           predict_function = predict_function,
                                           target_features = target_features,
                                           sample_size = sample_size)
#------------------------------------------------------------------------------
# 2: Causal asymmetric Shapley values with full causal weights of 1: ~30 seconds to run.
set.seed(1)
explained_full <- shapFlex::shapFlex(explain = explain,
                                     reference = reference,
                                     model = model,
                                     predict_function = predict_function,
                                     target_features = target_features,
                                     causal = causal,
                                     causal_weights = rep(1, nrow(causal)),
                                     # Pure causal weights
                                     sample_size = sample_size)
#------------------------------------------------------------------------------
# 3: Causal asymmetric Shapley values with agnostic causal weights of .5: ~30 seconds to run.
set.seed(1)
explained_half <- shapFlex::shapFlex(explain = explain,
                                     reference = reference,
                                     model = model,
                                     predict_function = predict_function,
                                     target_features = target_features,
                                     causal = causal,
                                     causal_weights = rep(.5, nrow(causal)),  
                                     # Approximates symmetric calc.
                                     sample_size = sample_size)

explained_non_causal_sum <- explained_non_causal %>%
  dplyr::group_by(feature_name) %>%
  dplyr::summarize("shap_effect" = mean(shap_effect, na.rm = TRUE))
explained_non_causal_sum$type <- "Symmetric"

explained_full_sum <- explained_full %>%
  dplyr::group_by(feature_name) %>%
  dplyr::summarize("shap_effect" = mean(shap_effect, na.rm = TRUE))
explained_full_sum$type <- "Pure causal (1)"

explained_half_sum <- explained_half %>%
  dplyr::group_by(feature_name) %>%
  dplyr::summarize("shap_effect" = mean(shap_effect, na.rm = TRUE))
explained_half_sum$type <- "Agnostic causal (.5)"
#------------------------------------------------------------------------------
# Plot the Shapley feature effects for the target features.

data_plot <- dplyr::bind_rows(explained_non_causal_sum, explained_full_sum, explained_half_sum)

# Re-order the target features so the causal outcomes are first.
data_plot$feature_name <- factor(data_plot$feature_name, levels = target_features, ordered = TRUE)

p <- ggplot(data_plot, aes(feature_name, shap_effect, fill = ordered(type)))
p <- p + geom_col(position = position_dodge())
p <- p + theme_bw() + theme(
  plot.title = element_text(size = 14, face = "bold"),
  axis.title = element_text(size = 12, face = "bold"),
  axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5, size = 12),
  axis.text.y = element_text(size = 12)
)
p <- p + xlab(NULL) + ylab("Average Shapley effect (baseline is .23)") + labs(fill = "Algorithm") +
  ggtitle("Average Shapley Feature Effects Based on 3 Causal Assumptions")
p

