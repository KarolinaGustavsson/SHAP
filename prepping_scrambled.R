install.packages("haven")  # if not already installed
library(haven)

# Read the SAS file
df <- read_sas("/Users/kargus/SHAP_LUK/actual_amoris_shap/data/scrambled.sas7bdat")
# for comparison
nh3 <- read.csv("/Users/kargus/SHAP_LUK/amoris_shap/data/NH3.csv")

# Write to CSV
write.csv(df, "/Users/kargus/SHAP_LUK/actual_amoris_shap/data/scrambled.csv", row.names = FALSE)


# Baseline
# 1. Rename columns
df_renamed <- df
names(df_renamed)[names(df_renamed) == "IdS"] <- "sampleID"
names(df_renamed)[names(df_renamed) == "Alder"] <- "age"

# 2. Extract the first measurement per sampleID
library(dplyr)

df_baseline <- df_renamed %>%
  arrange(sampleID, age) %>%        # Ensure chronological order
  group_by(sampleID) %>%
  slice(1) %>%                      # Take the first row per group
  ungroup()

# 4. Adding mortality status
df_baseline <- df_baseline %>%
  mutate(status = ifelse(Event == 20, 1, 0))
df_baseline$status <- as.integer(df_baseline$status)

# 5. Complete obervations
required_cols <- c("sampleID", "TC", "TG", "fS_Gluk", "S_Krea", "S_Hapt",
                   "S_FAMN", "S_Urat", "S_Alb", "S_Alp", "fS_Jaern", "fS_TIBC",
                   "S_LD", "S_Ca", "S_Urea", "S_P", "Fe_maet", "S_K", "age", "status")


df_complete <- df_baseline %>%
  filter(if_all(all_of(required_cols), ~ !is.na(.) & . != ""))


# 6. Saving baseline
write.csv(df_complete, "/Users/kargus/SHAP_LUK/actual_amoris_shap/data/scrambled.csv", row.names = FALSE)