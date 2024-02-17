
# Improving Recommender System Security: Multi-Objective Shilling Attack with Generative Adversarial Networks

The repository contains the implementation of Shilling Attacks on recommendation systems with different attack methods and related experimental data.
> Kai Wang

## Catalog interpretation
### NeuralShillMatrix/data
    Mainly used to store raw data and attack data (generated fake data)
#### NeuralShillMatrix/data/attack_data
    Attack data generated in different ways (fake user rating data)
    
    Consists of 2 types of files:
    
    1. Attack configuration file: records the attack Settings.
    
    Example: filmTrust_50_16_1689_attackSetting.npy
        filmTrust is the data source name, 
        50 is the number of users used by the attack, 
        16 is the number of populate items used per user, 
        1689 is the target item id.
    
    2. Generated fake user rating data: used for direct attacks.
    
    Example: filmTrust_1689_average_50_16.dat
        filmTrust is the data source name, 
        50 is the number of users used by the attack, 
        16 is the number of populated items used per user, 
        1689 is the target item id,
        average is the attack method (method of generating fake user rating data).
#### NeuralShillMatrix/data/merged_data
    Store attack data for attacks on multiple target items.
    
    NeuralShillMatrix/data/merged_data/1689
        Examine the attack data influenced by the recommendation results of target project 1689.
    
    NeuralShillMatrix/data/merged_data/1689/1
        One target item was used to carry out Shilling Attacks.
    
    NeuralShillMatrix/data/merged_data/1689/2
        Two target items were used to carry out Shilling Attacks.
    
    NeuralShillMatrix/data/merged_data/1689/3
        Three target items were used to carry out Shilling Attacks.
    
    NeuralShillMatrix/data/merged_data/1689/4
        Four targeted items were used to carry out Shilling Attacks.
    
    NeuralShillMatrix/data/merged_data/1689/5
        Five targeted items were used to carry out Shilling Attacks.
#### NeuralShillMatrix/data/raw_data
    Raw data and related processed data.
    
    filmTrust_id_mapping.npy
    The mapping between the user and project ids in the data source and the ids used in the experiment.

    filmTrust_selected_items
    Selected project id and associated project id.

    filmTrust_target_users
    Selected project id and for user id.

    filmTrust_test.dat
    Test data.

    filmTrust_train.dat
    Training data.

    Ratings.txt
    The raw data.
### NeuralShillMatrix/result
#### NeuralShillMatrix/result/csv_evaluation
    This directory contains the result files that are used to evaluate the attack effectiveness and concealment metrics, 
    calculated based on the recommendation index and ranking predicted by the recommendation model.

#### NeuralShillMatrix/result/image
    The catalog contains visual charts generated based on the results of attack effectiveness evaluation and concealment metrics.

#### NeuralShillMatrix/result/model_checkpoint
    This directory is used to store the relevant data generated during the running of the model.

#### NeuralShillMatrix/result/model_evaluation
    This catalog contains the predicted rating and ranking data generated by the recommendation model.
